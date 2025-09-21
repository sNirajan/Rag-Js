// Tiny RAG API server for the demo User Interface

import "dotenv/config"; // Loads env variables

// Node built-ins for reading files and resolving paths
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

// Web Server framework  + CORS helper
import express from "express";
import cors from "cors";

// Langchain pieces: embeddings (vectors), chat model, and a simple vector store
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

// Resolves "where am I?" so we can find index.json and /public
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Snapshot created from ingest.mjs
const INDEX_JSON = path.join(__dirname, "..", "index.json");

// Heuristics: vagueness + low-confidence checks
const STOPWORDS = new Set([
  "the",
  "and",
  "for",
  "with",
  "you",
  "your",
  "are",
  "but",
  "not",
  "from",
  "this",
  "that",
  "it",
  "here",
  "there",
  "what",
  "is",
  "about",
  "please",
  "tell",
  "me",
  "a",
  "an",
  "of",
  "to",
  "in",
  "on",
  "at",
  "by",
  "be",
  "we",
  "they",
  "i",
]);

function tokenizeMeaningful(text) {
  return String(text)
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((w) => w.length > 2 && !STOPWORDS.has(w));
}

function isVagueQuestion(q) {
  const trimmed = q.trim();
  if (trimmed.length < 8) return true; // very short
  const lower = trimmed.toLowerCase();
  if (/\b(this|that|it|here|there|this one)\b/.test(lower)) return true; // deictic
  return false;
}

function hasKeywordOverlap(q, passage) {
  const qWords = new Set(tokenizeMeaningful(q));
  const pWords = new Set(tokenizeMeaningful(passage));
  let overlap = 0;
  for (const w of qWords) if (pWords.has(w)) overlap++;
  return overlap > 0; // at least 1 meaningful word in common
}

/*
 * This is NEW: micro synonym/alias expansion to improve recall on common asks
 * - Example: "open on sat?" -> expands to include "saturday weekend"
 * - Example: "bring my car" -> includes "vehicle parking"
 * This does not remove anything; it simply appends helpful terms.
 *  */
function expandQuestionSynonyms(q) {
  let out = q;
  const lower = q.toLowerCase();

  // weekend variants
  if (/\b(sat|saturday)\b/.test(lower)) out += " saturday weekend";
  if (/\b(sun|sunday)\b/.test(lower)) out += " sunday weekend";
  if (/\b(weekend|weekends)\b/.test(lower)) out += " saturday sunday";

  // parking / car
  if (/\b(car|cars|vehicle|vehicles|drive)\b/.test(lower))
    out += " parking lot";

  // hours / open / opening
  if (/\b(open|hours|opening|close|closing)\b/.test(lower))
    out += " hours open opening times";

  return out;
}

/**
 * bootstrapStore()
 * Builds the in-memory search index ONCE at server startup.
 * Reads index.json (your chunks + metadata)
 * Re-embeds them (turns text -> vectors)
 * Loads into MemoryVectorStore for fast semantic search
 *
 * Why at startup? Faster requests later. The first request
 * doesn't have to re-embed everything.
 */
async function bootstrapStore() {
  // ensures the snapshot exists, otherwise instructs user to run ingest
  if (!fs.existsSync(INDEX_JSON)) {
    throw new Error("index.json not found - run `npm run ingest` first.");
  }

  // reads the snapshot file (array of { pageContent, metadata })
  const snapshot = JSON.parse(fs.readFileSync(INDEX_JSON, "utf8"));

  // splits snapshot into parallel arrays for the vector store API
  const texts = snapshot.map((x) => x.pageContent);
  const metadatas = snapshot.map((x) => x.metadata);

  // creates the embeddings model  (uses the OPENAI_API_KEY)
  const embeddings = new OpenAIEmbeddings({ model: "text-embedding-3-small" });

  // Builds an in memory vector index so we can do semantic search
  const store = await MemoryVectorStore.fromTexts(texts, metadatas, embeddings);
  return store;
}

/**
 * buildSystemPrompt()
 * Returns the "rules" we give to the LLM. This keeps answers
 * concise, grounded in context, and cited. It also prevents
 * hallucinations.
 */

function buildSystemPrompt() {
  return [
    "You are a concise guest assistant.",
    "Use ONLY the context blocks provided.",
    "Do not add facts not present in the context.",
    'If the answer is not in the context, say: "I don\'t know."',
    "Cite the blocks you used with [1], [2], etc.",
    "If the answer naturally contains multiple items, present them as concise bullet points.",
    "If context is ambiguous or conflicting, briefly note that and ask for clarification.",
    "Keep answers short and professional. Use a friendly, neutral tone suitable for guest communications.",
  ].join(" ");
}

/**
 * main()
 * - Boot the store once
 * - Create the chat model
 * - Start an Express server
 * - Serve the static frontend from /public
 * - Expose POST /ask for the browser to call
 */

async function main() {
  // Builds the semantic search index once
  const store = await bootstrapStore();

  // Creates the chat model for generation (low temperature = factual)
  const llm = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0.2 });

  // Creates the web server
  const app = express();
  // Allows cross-origin requests (useful if hosting UI in different server)
  app.use(cors());

  // Parses JSON request bodies so req.body works
  app.use(express.json());

  // Serves static files from /public (index.html, CSS, client JS)
  app.use(express.static(path.join(__dirname, "..", "public"))); // serves frontend

  /**
   * POST /ask
   * Request body: { question: string }
   * Response: { answer: string, sources: [{id, source}] }
   */
  app.post("/ask", async (req, res) => {
    try {
      // Validating input
      const question = String(req.body?.question || "").trim();
      if (!question) return res.status(400).json({ error: "Missing question" });

      // Guardrail: vague question, asking for clarification
      if (isVagueQuestion(question)) {
        return res.json({
          answer:
            'Could you clarify your question? For example: "What are your hours?" or "Is outside food allowed?"',
          sources: [],
        });
      }

      // NEW: expand synonyms to help recall on common phrasings
      const expandedQuestion = expandQuestionSynonyms(question);

      // Retrieving (vector search + scores)
      const K = 3; // consider the top few
      const pairs = await store.similaritySearchWithScore(expandedQuestion, K);

      // Sorts by distance (defensive), take best (lower = closer)
      pairs.sort((a, b) => a[1] - b[1]);

      const [bestDoc, bestDistance] = pairs[0] || [];
      // RAISED THRESHOLD: allow near matches like 0.38 into context
      const COSINE_DISTANCE_MAX = 0.55; // tune 0.45–0.55 for  corpus

      if (!bestDoc || bestDistance > COSINE_DISTANCE_MAX) {
        return res.json({
          answer: "I don't know based on the provided documents.",
          sources: [],
        });
      }

      // Builds numbered context from the top few within threshold
      const results = pairs
        .filter(([, dist]) => dist <= COSINE_DISTANCE_MAX)
        .slice(0, K)
        .map(([doc]) => doc);
      if (results.length === 0 && pairs.length) {
        // fallback: use the best doc that passed the initial bestDoc check
        results.push(pairs[0][0]);
      }

      // Debug log (keep while tuning, then remove if you want)
      console.log(
        "Pairs + scores:",
        pairs.map(([doc, score]) => ({
          source: doc.metadata?.source,
          score,
        }))
      );

      // Builds numbered context blocks for citations [1], [2], ...
      const numbered = results.map((doc, i) => {
        const src = doc.metadata?.source || "unknown";
        return `[${i + 1}] (${src})\n${doc.pageContent}`;
      });

      // Builds the prompt: system rules + user with question & context
      const system = buildSystemPrompt();
      const user = `Question: ${question}
Context blocks:
${numbered.join("\n\n")}

Requirements:
- Answer using only the context.
- If missing, say "I don't know."
- Include citations like [1], [2].
`;

      // Asks the model to generate the answer
      const response = await llm.invoke([
        { role: "system", content: system },
        { role: "user", content: user },
      ]);

      // Returns a clean JSON shape to the browser
      let answer = (response.content || "").trim();

      // NEW: normalize rare "I don't now" typo to "I don't know"
      if (answer.startsWith("I don't now")) {
        answer = answer.replace(/^I don't now/, "I don't know");
      }

      /*
       * NEW: show ONLY the sources the model actually cited in the answer,
       * and deduplicate by filename (avoid showing faqs.txt twice).
       * - Citations look like [1], [2], [3] in the model output.
       * - We parse those indices and keep only those sources.
       *  */
      let sourcesToShow = [];
      if (!answer.startsWith("I don't know")) {
        // Builds the full list (as you had)
        const allSources = results.map((d, i) => ({
          id: i + 1,
          source: d.metadata?.source || "unknown",
        }));

        // Extracts cited indices from the answer: [1], [2], ...
        const citedIdx = new Set(
          (answer.match(/\[(\d+)\]/g) || [])
            .map((m) => Number(m.slice(1, -1)))
            .filter((n) => Number.isFinite(n))
        );

        // Keeps only cited sources; if none parsed, fall back to allSources
        const citedOnly = allSources.filter((s) => citedIdx.has(s.id));
        const base = citedOnly.length > 0 ? citedOnly : allSources;

        // Deduplicates by filename while preserving order
        const seen = new Set();
        sourcesToShow = base.filter((s) => {
          if (seen.has(s.source)) return false;
          seen.add(s.source);
          return true;
        });
      }

      // Final response
      return res.json({
        answer,
        sources: sourcesToShow, // will be [] if "I don't know"
      });
    } catch (err) {
      // If any error occurs, logs it and sends a safe error
      console.error(err);
      res.status(500).json({ error: "Server error" });
    }
  });

  // Starts the server on PORT or default 3000
  const port = process.env.PORT || 3000;
  app.listen(port, () => {
    console.log(` RAG server running at http://localhost:${port}`);
    console.log(`Open http://localhost:${port} in your browser.`);
  });
}

// Actually runs main()
main().catch((err) => {
  console.error("Failed to start server:", err);
  process.exit(1);
});
