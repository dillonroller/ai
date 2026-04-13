from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")

CHAT_MODEL = "gpt-4o-mini"
SYSTEM_PROMPT = "You are a helpful assistant. Keep responses concise."


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
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("Chatbot (type 'quit' to exit)\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input or user_input.lower() == "quit":
            break

        messages.append({"role": "user", "content": user_input})

        print("AI:  ", end="", flush=True)
        response = stream_response(client, messages)

        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
