from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")

CHAT_MODEL = "gpt-4o-mini"


def ask(client: OpenAI, system: str, user: str) -> str:
    stream = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.7,
        stream=True,
    )
    response = ""
    for chunk in stream:
        if content := chunk.choices[0].delta.content:
            print(content, end="", flush=True)
            response += content
    print()
    return response


def researcher(client: OpenAI, topic: str) -> str:
    system = (
        "You are a research agent. Given a topic, produce a list of 5-8 key facts "
        "about it. Be concise and factual. No fluff."
    )
    return ask(client, system, f"Research topic: {topic}")


def writer(client: OpenAI, research: str) -> str:
    system = (
        "You are a writing agent. Turn the provided research notes into a short, "
        "engaging paragraph (3-4 sentences). Make it readable, not a list."
    )
    return ask(client, system, f"Research notes:\n{research}")


def critic(client: OpenAI, draft: str) -> str:
    system = (
        "You are a critic agent. Review the draft and point out any issues: "
        "unclear sentences, missing context, factual problems. Be direct and brief."
    )
    return ask(client, system, f"Draft:\n{draft}")


def revise(client: OpenAI, draft: str, feedback: str) -> str:
    system = (
        "You are a writing agent. Revise the draft based on the critic's feedback. "
        "Keep the changes focused on what the critic mentioned."
    )
    return ask(client, system, f"Draft:\n{draft}\n\nFeedback:\n{feedback}")


def run_pipeline(client: OpenAI, topic: str) -> None:
    """Research, draft, critique, revise."""
    print(f"Topic: {topic}\n")

    print("-- Researcher --")
    research = researcher(client, topic)

    print("\n-- Writer --")
    draft = writer(client, research)

    print("\n-- Critic --")
    feedback = critic(client, draft)

    print("\n-- Revised --")
    revise(client, draft, feedback)


def main():
    client = OpenAI()
    run_pipeline(client, "prediction markets")


if __name__ == "__main__":
    main()
