# RAG (Retrieval-Augmented Generation)

A basic RAG pipeline from scratch. No frameworks — just OpenAI + ChromaDB.

## What is RAG?

LLMs only know what they were trained on. RAG fixes that by **retrieving relevant documents** before generating an answer. It's how tools like ChatGPT with file uploads, Notion AI, and enterprise search actually work.

```
Your question → search vector DB → find relevant chunks → send chunks + question to LLM → answer
```

## How to run

Make sure you've done the setup in [SETUP.md](../SETUP.md), then:

```bash
cd rag
python main.py
```

## What the code does

1. **Loads** text files from `documents/`
2. **Splits** them into paragraphs (chunks)
3. **Embeds** each chunk using OpenAI's embedding model (turns text → numbers)
4. **Stores** embeddings in a local ChromaDB database
5. **Takes your question**, embeds it, and finds the most similar chunks
6. **Sends** those chunks + your question to GPT-4o-mini for an answer

## Things to try

- **Change the question** in `main.py` — ask something only the documents would know
- **Add your own documents** — drop any `.txt` file into `documents/` and re-run
- **Ask something NOT in the documents** — see how the AI says it doesn't have enough info
- **Try different `n_results`** values — retrieve more or fewer chunks and see how answers change
