// Lesson 2 (Memory): load snapshot, re-embed on startup, then semantic search top-k.

import 'dotenv/config';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { OpenAIEmbeddings } from '@langchain/openai';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const INDEX_JSON = path.join(__dirname, '..', 'index.json');
const question = process.argv.slice(2).join(' ') || 'What time does the zoo open?';

if (!fs.existsSync(INDEX_JSON)) {
  console.error('Run `npm run ingest` first to create index.json.');
  process.exit(1);
}

// 1) Load snapshot (content + metadata)
const snapshot = JSON.parse(fs.readFileSync(INDEX_JSON, 'utf8'));

// 2) Re-embed and build an in-memory store
const texts = snapshot.map(x => x.pageContent);
const metadatas = snapshot.map(x => x.metadata);

const embeddings = new OpenAIEmbeddings({ model: 'text-embedding-3-small' });
const store = await MemoryVectorStore.fromTexts(texts, metadatas, embeddings);

// 3) Semantic search (top 3 chunks)
const results = await store.similaritySearch(question, 3);

// 4) Print context chunks (for Lesson 3 generation)
console.log('\nQuestion:', question);
if (results.length === 0) {
  console.log('\nNo relevant chunks found. Try rephrasing your question.');
  process.exit(0);
}

console.log(`\nTop ${results.length} chunk(s):\n`);
results.forEach((r, i) => {
  console.log(`[${i + 1}] (${r.metadata?.source || 'unknown'})\n${r.pageContent}\n`);
});


