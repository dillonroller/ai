import json
import random
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")

CHAT_MODEL = "gpt-4o-mini"


def get_weather(city: str) -> dict:
    weather_data = {
        "new york": {"temp": 72, "condition": "sunny"},
        "london": {"temp": 58, "condition": "cloudy"},
        "tokyo": {"temp": 80, "condition": "humid"},
    }
    data = weather_data.get(city.lower(), {"temp": random.randint(50, 90), "condition": "unknown"})
    return {"city": city, **data}


def calculate(expression: str) -> dict:
    try:
        return {"expression": expression, "result": eval(expression)}
    except Exception as e:
        return {"expression": expression, "error": str(e)}


TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "calculate": calculate,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city name"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a math expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "A math expression like '2 + 2' or 'sqrt(144)'"},
                },
                "required": ["expression"],
            },
        },
    },
]


def call_tool(name: str, args: dict) -> str:
    return json.dumps(TOOL_FUNCTIONS[name](**args))


def stream_chat(client: OpenAI, messages: list, tools: list, print_prefix: str = "") -> tuple[str, list[dict]]:
    """Stream a chat completion, accumulating content and tool calls as they arrive."""
    stream = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        tools=tools,
        stream=True,
    )
    content = ""
    tool_calls: dict[int, dict] = {}

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            if not content and print_prefix:
                print(print_prefix, end="", flush=True)
            print(delta.content, end="", flush=True)
            content += delta.content
        if delta.tool_calls:
            for tc in delta.tool_calls:
                if tc.index not in tool_calls:
                    tool_calls[tc.index] = {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                if tc.id:
                    tool_calls[tc.index]["id"] = tc.id
                if tc.function and tc.function.name:
                    tool_calls[tc.index]["function"]["name"] += tc.function.name
                if tc.function and tc.function.arguments:
                    tool_calls[tc.index]["function"]["arguments"] += tc.function.arguments

    if content:
        print()
    return content, list(tool_calls.values())


def run_conversation(client: OpenAI, user_message: str) -> None:
    """Send a message, let the LLM call tools if it wants, then get the final answer."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided tools when needed."},
        {"role": "user", "content": user_message},
    ]

    while True:
        content, tool_calls = stream_chat(client, messages, TOOLS, print_prefix="  Answer: ")

        if not tool_calls:
            return

        messages.append({"role": "assistant", "content": content or None, "tool_calls": tool_calls})

        for tool_call in tool_calls:
            name = tool_call["function"]["name"]
            args = json.loads(tool_call["function"]["arguments"])
            result = call_tool(name, args)
            print(f"  Tool call: {name}({args}) -> {result}")
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": result,
            })


def main():
    client = OpenAI()

    questions = [
        "What's the weather like in Tokyo?",
        "What's 1234 * 5678?",
        "Is it warmer in New York or London right now?",
    ]

    for question in questions:
        print(f"\nQ: {question}")
        run_conversation(client, question)


if __name__ == "__main__":
    main()
