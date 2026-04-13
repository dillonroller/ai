# Agent Memory

A chatbot that remembers you across sessions. After each exchange, it extracts personal facts and stores them in a vector database. On future conversations — even days later — it retrieves relevant memories and uses them to give context-aware responses.

## How to run

```bash
cd memory
python main.py
```

Commands inside the chat:
- `memories` — list everything the bot remembers about you
- `forget` — clear all stored memories
- `quit` — exit

## How it works

```
user message
    ↓
retrieve relevant memories from ChromaDB (semantic search)
    ↓
inject memories into system prompt
    ↓
generate response
    ↓
extract facts from the exchange (structured output)
    ↓
store new facts as embeddings in ChromaDB
```

### Memory lifecycle

1. **Recall** — before responding, embed the user's message and find the closest stored memories via cosine similarity. These get injected into the system prompt so the LLM can reference them naturally.

2. **Respond** — the LLM generates a response with those memories as context. It doesn't need special instructions to "use" them — they're just part of the prompt.

3. **Extract** — after the response, a second LLM call examines the exchange and pulls out any concrete personal facts (name, job, preferences, goals). This uses structured outputs to guarantee a clean list.

4. **Store** — new facts get embedded and added to ChromaDB. Duplicates are detected by content hash and skipped.

5. **Persist** — ChromaDB writes to disk (`memory_store/`), so memories survive across sessions. Close the terminal, come back tomorrow, and the bot still knows your name.

## Theory

### Why memory matters for agents

Without memory, every conversation starts from zero. The LLM has no idea who you are, what you've discussed before, or what you care about. This is fine for one-off questions but useless for anything ongoing — personal assistants, project copilots, customer support, coaching bots.

Memory turns a stateless function into something that feels like a relationship.

### Types of memory

This example implements **long-term semantic memory** — facts about the user stored as embeddings and retrieved by meaning. Production systems typically layer multiple types:

- **Short-term / buffer** — the conversation history within a single session. This example keeps the last 10 messages. Every chatbot has this.
- **Long-term / semantic** — facts extracted and stored across sessions. This is what this example builds. Retrieved by similarity to the current query.
- **Episodic** — full past conversations stored and retrievable. "What did we talk about last Tuesday?" Heavier to store, richer to retrieve.
- **Procedural** — learned patterns about *how* to behave. "The user prefers bullet points." Harder to implement, usually requires fine-tuning or explicit rules.

### The extraction step

The key insight is that you can't just dump the entire conversation history into a vector DB — it's too noisy. Instead, a second LLM call acts as a filter: it reads the exchange and extracts only the facts worth keeping. This is a form of **information compression** — turning a messy conversation into clean, retrievable knowledge.

Using structured outputs (`ExtractedFacts` with a `list[str]`) guarantees the LLM returns a parseable list, not free-form text. This is the same pattern from the `structured-outputs/` example applied to a real use case.

### Retrieval vs. stuffing

An alternative to semantic retrieval is to just stuff *all* memories into every prompt. This works when you have 5 facts but breaks at 500 — you'll blow the context window and dilute relevance. Embedding + similarity search ensures the LLM only sees the memories that matter for *this specific message*, regardless of how many are stored.

This is the same principle as RAG, applied to personal context instead of documents.

### Deduplication

Memories are hashed by content (`md5(fact)`). If the LLM extracts "User's name is Dillon" twice across sessions, only the first one gets stored. This prevents the memory store from filling up with redundant facts.

A more sophisticated system would detect *near-duplicates* (e.g., "User works at Acme" vs "User is employed at Acme Corp") via embedding similarity at write time, and merge or skip accordingly.

### Privacy and the right to forget

The `forget` command wipes all memories. In any production memory system, this is non-negotiable — users must be able to see what's stored and delete it. GDPR, CCPA, and basic trust all require it. This example makes it a first-class feature, not an afterthought.
