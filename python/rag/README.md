# RAG (Retrieval-Augmented Generation)

A basic RAG pipeline from scratch using OpenAI and ChromaDB.

## What is RAG?

LLMs only know what they were trained on. RAG lets you give them access to your own documents by searching a vector database for relevant chunks before generating an answer.

```
question -> search vector db -> find relevant chunks -> send to LLM -> answer
```

## How to run

Make sure you've done the setup in the root [README](../README.md), then:

```bash
cd rag
python main.py
```

## What the code does

1. Reads text files from `documents/`
2. Splits them into paragraphs (chunks)
3. Embeds each chunk into a vector using OpenAI's embedding model
4. Stores the vectors in a local ChromaDB database
5. Embeds your question and finds the most similar chunks
6. Sends those chunks along with your question to GPT-4o-mini

## Things to try

- Change the question in `main.py`
- Add your own `.txt` files to `documents/` and re-run
- Ask something that isn't in the documents and see what happens
- Change `n_results` to retrieve more or fewer chunks
