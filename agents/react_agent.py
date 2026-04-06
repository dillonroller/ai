import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")

CHAT_MODEL = "gpt-4o-mini"
MAX_STEPS = 10


def search_web(query: str) -> str:
    fake_results = {
        "polymarket": "Polymarket is a decentralized prediction market built on Polygon. It uses USDC for trading.",
        "ethereum price": "Ethereum is currently trading around $3,200 USD.",
        "bitcoin price": "Bitcoin is currently trading around $67,000 USD.",
        "rust language": "Rust is a systems programming language focused on safety and performance. Created by Graydon Hoare at Mozilla.",
    }
    for key, result in fake_results.items():
        if key in query.lower():
            return result
    return "No results found."


def calculator(expression: str) -> str:
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


TOOL_FUNCTIONS = {
    "search_web": search_web,
    "calculator": calculator,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for information",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a math expression",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a ReAct agent. For every step:

1. First, think out loud about what you need to do next
2. Then call a tool OR give the final answer

Always explain your reasoning before taking an action. Be explicit about why
you're choosing each step."""


def call_tool(name: str, args: dict) -> str:
    return TOOL_FUNCTIONS[name](**args)


def run_react(client: OpenAI, task: str) -> None:
    """Run a ReAct loop where the agent reasons out loud before each action."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]

    for step in range(1, MAX_STEPS + 1):
        response = client.chat.completions.create(model=CHAT_MODEL, messages=messages, tools=TOOLS)
        message = response.choices[0].message

        if message.content:
            print(f"\n[Step {step}] Thought: {message.content}")

        if not message.tool_calls:
            return

        messages.append(message)

        for tool_call in message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            result = call_tool(name, args)
            print(f"[Step {step}] Action: {name}({args})")
            print(f"[Step {step}] Observation: {result}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })


def main():
    client = OpenAI()
    task = "If I bought 1 Bitcoin and 2 Ethereum today, how much would I have spent in USD?"
    print(f"Task: {task}")
    run_react(client, task)


if __name__ == "__main__":
    main()
