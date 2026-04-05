from dataclasses import dataclass
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import chromadb

load_dotenv(dotenv_path="../.env")

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
DOCS_DIR = Path("documents")
CHROMA_DIR = "chroma_data"
COLLECTION = "my_documents"


@dataclass
class Chunk:
    id: str
    text: str
    source: str


def load_chunks(docs_dir: Path) -> list[Chunk]:
    """Load text files and split into paragraph-level chunks."""
    chunks: list[Chunk] = []
    for path in sorted(docs_dir.glob("*.txt")):
        paragraphs = [p.strip() for p in path.read_text().split("\n\n") if p.strip()]
        for i, text in enumerate(paragraphs):
            chunks.append(Chunk(id=f"{path.name}_{i}", text=text, source=path.name))
    return chunks


def embed(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Get embeddings for a list of texts."""
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [e.embedding for e in resp.data]


def build_index(client: OpenAI, chunks: list[Chunk]) -> chromadb.Collection:
    """Embed chunks and store them in a local ChromaDB collection."""
    db = chromadb.PersistentClient(path=CHROMA_DIR)
    # Reset collection on each run so we always reflect current documents
    db.delete_collection(COLLECTION) if COLLECTION in [c.name for c in db.list_collections()] else None
    collection = db.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    collection.add(
        ids=[c.id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embed(client, [c.text for c in chunks]),
        metadatas=[{"source": c.source} for c in chunks],
    )
    return collection


def retrieve(client: OpenAI, collection: chromadb.Collection, query: str, n: int = 3) -> list[str]:
    """Find the n most relevant chunks for a query."""
    results = collection.query(query_embeddings=embed(client, [query]), n_results=n)
    return results["documents"][0]


def ask(client: OpenAI, question: str, context: list[str]) -> None:
    """Send question + retrieved context to the LLM and stream the response."""
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

    chunks = load_chunks(DOCS_DIR)
    print(f"Loaded {len(chunks)} chunks from {DOCS_DIR}/\n")

    collection = build_index(client, chunks)
    context = retrieve(client, collection, question)

    print(f"Q: {question}\n")
    print("A: ", end="", flush=True)
    ask(client, question, context)


if __name__ == "__main__":
    main()
