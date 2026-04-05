"""
Function calling lets the LLM use tools you define. Instead of just generating text,
it can decide to call a function, get the result, and use that to answer your question.

This is how ChatGPT plugins, web browsing, and code execution work under the hood.
"""

import json
import random
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")

CHAT_MODEL = "gpt-4o-mini"


# -- Tools the LLM can call --------------------------------------------------
# These are just regular Python functions. The LLM doesn't run them directly,
# it just decides WHEN to call them and with WHAT arguments. We run them ourselves.

def get_weather(city: str) -> dict:
    """Fake weather API for demo purposes."""
    weather_data = {
        "new york": {"temp": 72, "condition": "sunny"},
        "london": {"temp": 58, "condition": "cloudy"},
        "tokyo": {"temp": 80, "condition": "humid"},
    }
    data = weather_data.get(city.lower(), {"temp": random.randint(50, 90), "condition": "unknown"})
    return {"city": city, **data}


def calculate(expression: str) -> dict:
    """Evaluate a math expression."""
    try:
        result = eval(expression)  # fine for a demo, never do this in production
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"expression": expression, "error": str(e)}


# Map of function names to actual functions
TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "calculate": calculate,
}

# This is the schema that tells the LLM what tools exist and how to call them.
# It's just JSON describing the function name, parameters, and types.
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
    fn = TOOL_FUNCTIONS[name]
    result = fn(**args)
    return json.dumps(result)


def run_conversation(client: OpenAI, user_message: str) -> None:
    """Send a message, let the LLM call tools if it wants, then get the final answer."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Use the provided tools when needed."},
        {"role": "user", "content": user_message},
    ]

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        tools=TOOLS,
    )
    message = response.choices[0].message

    # The LLM might call one or more tools before answering
    while message.tool_calls:
        messages.append(message)

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            result = call_tool(name, args)
            print(f"  Tool call: {name}({args}) -> {result}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

        # Send the tool results back so the LLM can generate a final answer
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            tools=TOOLS,
        )
        message = response.choices[0].message

    print(f"  Answer: {message.content}")


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
