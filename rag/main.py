import hashlib
from dataclasses import dataclass
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path="../.env")

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
DOCS_DIR = Path("documents")
CHROMA_DIR = "chroma_data"


@dataclass
class Chunk:
    id: str
    text: str
    source: str


def read_documents(docs_dir: Path) -> list[Chunk]:
    """Read .txt files and split them into paragraph-sized chunks."""
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.glob("*.txt")):
        paragraphs = [p.strip() for p in path.read_text().split("\n\n") if p.strip()]
        for i, text in enumerate(paragraphs):
            chunk_id = content_hash(path.name, i, text)
            chunks.append(Chunk(id=chunk_id, text=text, source=path.name))
    return chunks


def content_hash(filename: str, index: int, text: str) -> str:
    """Hash based on content so edited documents get re-embedded."""
    return hashlib.md5(f"{filename}_{index}_{text}".encode()).hexdigest()


def get_embeddings(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Turn a list of strings into vectors using OpenAI's embedding model."""
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [e.embedding for e in resp.data]


def get_or_create_collection(db: chromadb.PersistentClient) -> chromadb.Collection:
    return db.get_or_create_collection("docs", metadata={"hnsw:space": "cosine"})


def find_new_chunks(chunks: list[Chunk], existing_ids: set[str]) -> list[Chunk]:
    return [c for c in chunks if c.id not in existing_ids]


def find_stale_ids(chunks: list[Chunk], existing_ids: set[str]) -> list[str]:
    current_ids = {c.id for c in chunks}
    return [id for id in existing_ids if id not in current_ids]


def sync_index(client: OpenAI, chunks: list[Chunk]) -> chromadb.Collection:
    """Sync chunks to the vector database. Only embeds new or changed chunks."""
    db = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = get_or_create_collection(db)
    existing_ids = set(collection.get()["ids"])

    stale = find_stale_ids(chunks, existing_ids)
    if stale:
        collection.delete(ids=stale)
        print(f"  Removed {len(stale)} stale chunks")

    new = find_new_chunks(chunks, existing_ids)
    if new:
        collection.add(
            ids=[c.id for c in new],
            documents=[c.text for c in new],
            embeddings=get_embeddings(client, [c.text for c in new]),
            metadatas=[{"source": c.source} for c in new],
        )
        print(f"  Embedded {len(new)} new chunks")
    else:
        print("  Already up to date")

    return collection


def retrieve(client: OpenAI, collection: chromadb.Collection, query: str, n: int = 3) -> list[str]:
    """Find the n chunks closest in meaning to the query."""
    results = collection.query(
        query_embeddings=get_embeddings(client, [query]),
        n_results=n,
    )
    return results["documents"][0]


def stream_response(client: OpenAI, question: str, context: list[str]) -> None:
    """Send the question and retrieved context to the LLM, stream the answer."""
    ctx = "\n\n".join(context)
    stream = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": (
                "Answer using ONLY the provided context. "
                "If the context doesn't cover it, say so. Be concise."
            )},
            {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}"},
        ],
        temperature=0.3,
        stream=True,
    )
    for chunk in stream:
        if content := chunk.choices[0].delta.content:
            print(content, end="", flush=True)
    print()


def main():
    client = OpenAI()
    question = "What's something interesting about Dillon?"

    chunks = read_documents(DOCS_DIR)
    print(f"Loaded {len(chunks)} chunks from {DOCS_DIR}/")

    collection = sync_index(client, chunks)
    context = retrieve(client, collection, question)

    print(f"\nQ: {question}\n")
    print("A: ", end="", flush=True)
    stream_response(client, question, context)


if __name__ == "__main__":
    main()
