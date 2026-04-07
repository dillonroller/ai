import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")


def make_openai_client() -> OpenAI:
    return OpenAI()


def make_ollama_client() -> OpenAI:
    return OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
    )


def stream_answer(client: OpenAI, model: str, question: str) -> float:
    """Stream the response and return how long it took."""
    start = time.time()
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": question}],
        temperature=0.7,
        stream=True,
    )
    for chunk in stream:
        if content := chunk.choices[0].delta.content:
            print(content, end="", flush=True)
    print()
    return time.time() - start


def compare(question: str) -> None:
    """Run the same question against OpenAI and a local Ollama model."""
    print(f"\nQuestion: {question}\n")

    openai_client = make_openai_client()
    ollama_client = make_ollama_client()

    print("-- OpenAI (gpt-4o-mini) --")
    try:
        elapsed = stream_answer(openai_client, "gpt-4o-mini", question)
        print(f"({elapsed:.1f}s)")
    except Exception as e:
        print(f"Error: {e}")

    print("\n-- Local (llama3.2 via Ollama) --")
    try:
        elapsed = stream_answer(ollama_client, "llama3.2", question)
        print(f"({elapsed:.1f}s)")
    except Exception as e:
        print(f"Error: {e}")
        print("Is ollama running? Try: ollama serve")


def main():
    questions = [
        "In one sentence, what is a vector database?",
        "Write a haiku about debugging code.",
    ]
    for question in questions:
        compare(question)


if __name__ == "__main__":
    main()
