import hashlib
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv(dotenv_path="../.env")

CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
CHROMA_DIR = "memory_store"
COLLECTION_NAME = "memories"
MAX_MEMORIES = 5


class ExtractedFacts(BaseModel):
    facts: list[str]


def get_embedding(client: OpenAI, text: str) -> list[float]:
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
    return resp.data[0].embedding


def get_collection(db: chromadb.PersistentClient) -> chromadb.Collection:
    return db.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def store_memories(client: OpenAI, collection: chromadb.Collection, facts: list[str]) -> int:
    new = []
    for fact in facts:
        fact_id = hashlib.md5(fact.encode()).hexdigest()
        existing = collection.get(ids=[fact_id])
        if not existing["ids"]:
            new.append((fact_id, fact))
    if not new:
        return 0
    ids = [n[0] for n in new]
    texts = [n[1] for n in new]
    embeddings = [get_embedding(client, t) for t in texts]
    collection.add(ids=ids, documents=texts, embeddings=embeddings)
    return len(new)


def recall(client: OpenAI, collection: chromadb.Collection, query: str) -> list[str]:
    if collection.count() == 0:
        return []
    n = min(MAX_MEMORIES, collection.count())
    results = collection.query(
        query_embeddings=[get_embedding(client, query)],
        n_results=n,
    )
    return results["documents"][0]


def extract_facts(client: OpenAI, user_msg: str, assistant_msg: str) -> list[str]:
    resp = client.beta.chat.completions.parse(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": (
                "Extract facts worth remembering about the user from this exchange. "
                "Only include concrete, personal facts (name, preferences, job, goals, etc.). "
                "If there's nothing worth remembering, return an empty list. "
                "Keep each fact to one short sentence."
            )},
            {"role": "user", "content": f"User said: {user_msg}\nAssistant said: {assistant_msg}"},
        ],
        response_format=ExtractedFacts,
    )
    return resp.choices[0].message.parsed.facts


def build_system_prompt(memories: list[str]) -> str:
    base = "You are a helpful assistant with memory across conversations. Keep responses concise."
    if not memories:
        return base
    memory_block = "\n".join(f"- {m}" for m in memories)
    return f"{base}\n\nYou remember these things about the user:\n{memory_block}"


def stream_response(client: OpenAI, messages: list[dict]) -> str:
    stream = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.7,
        stream=True,
    )
    response = ""
    for chunk in stream:
        if content := chunk.choices[0].delta.content:
            print(content, end="", flush=True)
            response += content
    print("\n")
    return response


def main():
    client = OpenAI()
    db = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = get_collection(db)

    print(f"Memory chatbot ({collection.count()} memories stored)")
    print("Commands: 'memories' to list, 'forget' to clear, 'quit' to exit\n")

    messages = []

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue

        if user_input.lower() == "quit":
            break

        if user_input.lower() == "memories":
            all_memories = collection.get()
            if not all_memories["documents"]:
                print("No memories stored.\n")
            else:
                for doc in all_memories["documents"]:
                    print(f"  - {doc}")
                print()
            continue

        if user_input.lower() == "forget":
            ids = collection.get()["ids"]
            if ids:
                collection.delete(ids=ids)
            print(f"Cleared all memories.\n")
            continue

        memories = recall(client, collection, user_input)
        system_prompt = build_system_prompt(memories)

        messages = [{"role": "system", "content": system_prompt}] + [
            m for m in messages if m["role"] != "system"
        ][-10:]
        messages.append({"role": "user", "content": user_input})

        print("AI:  ", end="", flush=True)
        response = stream_response(client, messages)

        messages.append({"role": "assistant", "content": response})

        facts = extract_facts(client, user_input, response)
        if facts:
            stored = store_memories(client, collection, facts)
            if stored:
                print(f"  [remembered {stored} new fact(s)]\n")


if __name__ == "__main__":
    main()
