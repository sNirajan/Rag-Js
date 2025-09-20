// Tiny RAG API server for the demo User INterface

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
      // VAlidating input
      const question = String(req.body?.question || "").trim();
      if (!question) return res.status(400).json({ error: "Missing question" });

      // Guardrail: vague question, asking for clarificaiton
      if (isVagueQuestion(question)) {
        return res.json({
          answer:
            'Could you clarify your question? For example: "What are your hours?" or "Is outside food allowed?"',
          sources: [],
        });
      }

      // Retrieving top-1 for crisp answer (can change to 3-5)
      const K = 1;
      const results = await store.similaritySearch(question, K);
      if (!results || results.length === 0) {
        return res.json({
          answer: "I don't know based on the provided documents.",
          sources: [],
        });
      }

      // 3) Low-confidence check: if no keyword overlap with top chunk, refuse
      const top = results[0];
      if (!hasKeywordOverlap(question, top.pageContent)) {
        return res.json({
          answer:
            "I don't know based on the provided documents. Try a more specific question, e.g., \“What are seasonal hours?\”",
          sources: [],
        });
      }

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

      const answer = (response.content || "").trim();

      // Hiding sources if we refused
      return res.json({
        answer,
        sources: answer.startsWith("I don't know")
          ? [] // no sources shown if model said "I don't know"
          : results.map((d, i) => ({
              id: i + 1,
              source: d.metadata?.source || "unknown",
            })),
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
