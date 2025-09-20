// (DONE) Lesson 2 (Memory): load snapshot, re-embed on startup, then semantic search top-k.

// Lesson 3: Retrieve - Generate (with citations)
// Loads snapshot from Lesson 2 memory path
// Retrieves top-k chunks (I have set k=1; can change to >1 anytime).
// Builds a safe prompt: "Use ONLY this context; otherwise say 'I don't know.'"
// Generates a concise answer and prints a clean "Sources" section.

import "dotenv/config";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const INDEX_JSON = path.join(__dirname, "..", "index.json");

// 0) Read the user's question from CLI
const question = process.argv.slice(2).join(" ").trim();

if (!question) {
  console.error(
    'Please pass a question, e.g.: npm run ask -- "What time does the zoo open?"'
  );
  process.exit(1);
}

// 1) Load snapshot (content + metadata)
if (!fs.existsSync(INDEX_JSON)) {
  console.error("index.json not found. Run: npm run ingest");
  process.exit(1);
}

const snapshot = JSON.parse(fs.readFileSync(INDEX_JSON, "utf8"));

// 2) Re-embed and build an in-memory store
const texts = snapshot.map((x) => x.pageContent);
const metadatas = snapshot.map((x) => x.metadata);

const embeddings = new OpenAIEmbeddings({ model: "text-embedding-3-small" });
const store = await MemoryVectorStore.fromTexts(texts, metadatas, embeddings);

// 3) Retrieve top-k content chunks (setting k = 3)
const K = 1; //
const results = await store.similaritySearch(question, K);

if (results.length === 0) {
  console.log(`\nQ: ${question}\n`);
  console.log("A: I don't know based on the provided documents");
  process.exit(0);
}

// 4) Build a number context block for citations like [1], [2]
const numberedBlocks = results.map((doc, i) => {
  const src = doc.metadata?.source || "unknown";
  return `[${i + 1}] (${src})\n${doc.pageContent}`;
});

const context = numberedBlocks.join("\n\n");

// 5) Safe prompt (prevents hallucinations)
const system = [
  "You are a concise guest assistant.",
  "Use ONLY the context blocks provided.",
  'If the answer is not in the context, say: "I don\'t know."',
  "Do not add facts not present in the context.",
  "Cite the blocks you used with [1], [2], etc.",
  "If the answer naturally contains multiple items, present them as concise bullet points.",
  "If context is ambiguous or conflicting, briefly note that and ask for clarification.",
  "Keep answers short and professional. Use a friendly, neutral tone suitable for guest communications.",
].join(" ");


const user = `Question: ${question}

Context blocks:
${context}

Requirements:
- Answer using only the context.
- If missing, say "I don't know."
- Include citations like [1], [2].
`;

// 6) Call the chat model
const llm = new ChatOpenAI({
  model: "gpt-4o-mini", // fast, clear; can swap anytime later
  temperature: 0.2, // low = focused and factual
  maxTokens: 250,
});

// 7) Generate
const response = await llm.invoke([
  {
    role: "system",
    content: system,
  },
  { role: "user", content: user },
]);

// 8) Nicely formatted output
console.log(`\nQ: ${question}\n`);
console.log("A:", response.content.trim(), "\n");

// 9) Print sources section (professional touch)
console.log("Sources:");
results.forEach((doc, i) => {
  const src = doc.metadata?.source || "unknown";
  console.log(`[${i + 1}] ${src}`);
});
