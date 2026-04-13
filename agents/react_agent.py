import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")

CHAT_MODEL = "gpt-4o-mini"
MAX_STEPS = 10


def search_web(query: str) -> str:
    q = query.lower()
    if "bitcoin" in q or "btc" in q:
        return "Bitcoin is currently trading around $67,000 USD."
    if "ethereum" in q or "eth" in q:
        return "Ethereum is currently trading around $3,200 USD."
    if "polymarket" in q:
        return "Polymarket is a decentralized prediction market built on Polygon. It uses USDC for trading."
    if "rust" in q:
        return "Rust is a systems programming language focused on safety and performance. Created by Graydon Hoare at Mozilla."
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


def stream_chat(client: OpenAI, messages: list) -> tuple[str, list[dict]]:
    """Stream a chat completion, accumulating content and tool calls as they arrive."""
    stream = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        tools=TOOLS,
        stream=True,
    )
    content = ""
    tool_calls: dict[int, dict] = {}

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
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

    return content, list(tool_calls.values())


def run_react(client: OpenAI, task: str) -> None:
    """Run a ReAct loop where the agent reasons out loud before each action."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]

    for step in range(1, MAX_STEPS + 1):
        print(f"\n[Step {step}] Thought: ", end="", flush=True)
        content, tool_calls = stream_chat(client, messages)
        print()

        if not tool_calls:
            return

        messages.append({
            "role": "assistant",
            "content": content or None,
            "tool_calls": tool_calls,
        })

        for tool_call in tool_calls:
            name = tool_call["function"]["name"]
            args = json.loads(tool_call["function"]["arguments"])
            result = call_tool(name, args)
            print(f"[Step {step}] Action: {name}({args})")
            print(f"[Step {step}] Observation: {result}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": result,
            })


def main():
    client = OpenAI()
    task = "If I bought 1 Bitcoin and 2 Ethereum today, how much would I have spent in USD?"
    print(f"Task: {task}")
    run_react(client, task)


if __name__ == "__main__":
    main()
