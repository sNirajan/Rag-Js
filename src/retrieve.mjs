// Goal: given a question, find the most relevant text chunks from our files
// I am building a tiny search tool that splits documents into pieces and picks the pieces whose word best match our questions

import fs from "node:fs"; // file system lets us read files from disk
import path from "node:path"; // paths helps build safe file/folder paths
import { fileURLToPath } from "node:url"; // converts a module URL into a normal file path

const __filename = fileURLToPath(import.meta.url); // absolute path to "this" file
const __dirname = path.dirname(__filename); // folder that contains this file

// Load all .txt documents
const DATA_DIR = path.join(__dirname, "..", "data");
// This makes a path to the "data" folder, one level up from src, where .txt files live

const files = fs.readdirSync(DATA_DIR).filter((f) => f.endsWith(".txt"));
// Reads all filenames in /data, then keeps only those ending in a ".txt"

if (files.length === 0) {
  console.error(
    "No .txt files found in /data. Please add one (e.g., zoo_faq.txt)."
  );
  process.exit(1);
} // if there are no .txt files, this error shows up and stops the program

// Ready every file into a big string
let corpus = []; // collects documents here: [{ source: 'zoo_faq.txt', text: '...' }, ...]
for (const f of files) {
  const text = fs.readFileSync(path.join(DATA_DIR, f), "utf8");
  // opens each file and read its entire content as a UTF-8 string
  corpus.push({ source: f, text });
  // saves both the filename (for citations) and file text
}

// Split into chunks ("pages") - small pieces help retrieval be precise
// Splitting by blank lines first as simple paragraphs
function chunkDocument(doc) {
  const rawParas = doc.text.split(/\n\s*\n/g);
  // Splits the document wherever there is a blank line
  // The regex /\n\s*\n/g means: a newline, optional spaces, then another newline

  // Clean and keep source info with each chunk
  return rawParas
    .map((p) => p.trim()) // removes extra spaces around each paragraph
    .filter((p) => p.length > 0) // ignores empty paragraphs
    .map((p) => ({ source: doc.source, content: p })); // converts each paragraph into {source: 'file.txt', content: 'paragraph text'}
}

// Build Chunk list

let chunks = []; // all chunks from all documents go here
for (const doc of corpus) {
  chunks.push(...chunkDocument(doc)); // adds this doc's chunks to the big list
}

// Simple retrieval by keyword overlap
// Idea: turn question into keywords: score each chunk by how many keywords it contains
function keywords(text) {
  return text
    .toLowerCase() // makes everything lowercase
    .replace(/[^a-z0-9\s]/g, " ") // replaces punctuation with spaces
    .split(/\s+/) // splits into words on whitespace
    .filter((w) => w.length > 2 && !STOPWORDS.has(w));
  // keeps only meaningful words; longer than 2 letters and not a stopword like "the", "and"
}

// A tiny list of stopwords; we ignore these (they're not meaningful)
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
  "will",
  "have",
  "has",
  "our",
  "any",
  "can",
  "may",
  "into",
  "over",
  "near",
  "per",
  "most",
]);
// words that are too common to be useful for matching

// Given a question and one chunk, produce a score = how many words overlap
function scoreChunk(question, chunkText) {
  const qwords = new Set(keywords(question)); // unique keywords from the question
  const cwords = new Set(keywords(chunkText)); // unique keywords from the chunk
  let overlap = 0;
  for (const w of qwords) if (cwords.has(w)) overlap++; // Count shared words
  return overlap; // higher = more relevant
}

// Main: get questions from CLI, rank chunks, print top hits
const question =
  process.argv.slice(2).join(" ") || "What time does the zoo open?";
// Reads the question from the command line arguments
// If none given, default to a sample question

// Score each chunk
const scored = chunks
  .map((ch) => ({ ...ch, score: scoreChunk(question, ch.content) })) // for each chunk, computes a score and attach it as { source, content, score }
  .filter((x) => x.score > 0);
// keeps only chunks that have at least one overlapping keyword

// Sorts descending by score
scored.sort((a, b) => b.score - a.score);
// Higher scores first, so best matches come first

// Takes top-k results
const k = 3;
const top = scored.slice(0, k);
// keeps just the top 3 matches (can adjust k)

// prints
console.log("\nQuestion:", question);
if (top.length === 0) {
  console.log("\nNo relevant chunks found. Try rephrasing your question");
} else {
  console.log(`\nTop ${top.length} context chunk'(s):\n`);
  top.forEach((t, i) => {
    console.log(`[${i + 1}] (${t.source}) score=${t.score}\n${t.content}\n`);
  });
  // shows each winning chunk with its source file and score, so you can see the evidence
}

console.log("---");
console.log("Now I would pass these chunk(s) + my question to an AI model.");
console.log(
  "In Lesson 2, I will replace keyword search with embeddings (better retrieval)."
);
