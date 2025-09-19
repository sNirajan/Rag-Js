// This is Lesson 2 (Memory): load .txt -> chunk -> embed (OPENAI) -> hold in memory
// and save a lightweight snapshot so ask.mjs can rebuild the memory store

import 'dotenv/config';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { Document } from 'langchain/document';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { OpenAIEmbeddings } from '@langchain/openai';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const DATA_DIR = path.join(__dirname, '..', 'data');
const INDEX_JSON = path.join(__dirname, '..', 'index.json');

// 1) Load .txt files (manual, no @langchain/community - this gave me error)
const files = fs.readdirSync(DATA_DIR).filter(f => f.endsWith('.txt'));
if (files.length === 0) {
  console.error('No .txt files found in /data. Add one (e.g., zoo_faq.txt) and re-run.');
  process.exit(1);
}

let docs = [];
for (const f of files) {
  const text = fs.readFileSync(path.join(DATA_DIR, f), 'utf8');
  // Create one Document per file (good enough for our chunker)
  docs.push(new Document({ pageContent: text, metadata: { source: f } }));
}
console.log(`Loaded ${docs.length} document(s) from ${files.length} file(s).`);

// 2) Chunk (~500 chars with small overlap)
const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 500, chunkOverlap: 50 });
const chunks = await splitter.splitDocuments(docs);
console.log(`Split into ${chunks.length} chunk(s).`);

// 3) (Optional now) Create embeddings + memory store (proves it works end-to-end)
const embeddings = new OpenAIEmbeddings({ model: 'text-embedding-3-small' });
await MemoryVectorStore.fromDocuments(chunks, embeddings);

// 4) Save a lightweight snapshot (content + metadata)
//   Note: We don’t save numeric vectors here—keeps the file small.
//   ask.mjs will re-embed on load (OK for small projects).
const snapshot = chunks.map(c => ({ pageContent: c.pageContent, metadata: c.metadata }));
fs.writeFileSync(INDEX_JSON, JSON.stringify(snapshot, null, 2), 'utf8');

console.log(` Snapshot saved to ${INDEX_JSON}. (ask.mjs will re-embed on demand)`);