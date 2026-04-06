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


def ask(client: OpenAI, model: str, question: str) -> tuple[str, float]:
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": question}],
        temperature=0.7,
    )
    elapsed = time.time() - start
    return response.choices[0].message.content, elapsed


def compare(question: str) -> None:
    """Run the same question against OpenAI and a local Ollama model."""
    print(f"\nQuestion: {question}\n")

    openai_client = make_openai_client()
    ollama_client = make_ollama_client()

    print("-- OpenAI (gpt-4o-mini) --")
    try:
        answer, elapsed = ask(openai_client, "gpt-4o-mini", question)
        print(f"({elapsed:.1f}s) {answer}")
    except Exception as e:
        print(f"Error: {e}")

    print("\n-- Local (llama3.2 via Ollama) --")
    try:
        answer, elapsed = ask(ollama_client, "llama3.2", question)
        print(f"({elapsed:.1f}s) {answer}")
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
