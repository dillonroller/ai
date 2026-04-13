import json
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")

CHAT_MODEL = "gpt-4o-mini"
MAX_STEPS = 10


def read_file(filename: str) -> str:
    try:
        with open(filename) as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: {filename} not found"


def write_file(filename: str, content: str) -> str:
    with open(filename, "w") as f:
        f.write(content)
    return f"Wrote {len(content)} characters to {filename}"


def list_files(directory: str) -> str:
    path = Path(directory)
    if not path.exists():
        return f"Error: {directory} not found"
    files = [f.name for f in path.iterdir() if f.is_file()]
    return json.dumps(files)


TOOL_FUNCTIONS = {
    "read_file": read_file,
    "write_file": write_file,
    "list_files": list_files,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {"filename": {"type": "string"}},
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["filename", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List all files in a directory",
            "parameters": {
                "type": "object",
                "properties": {"directory": {"type": "string"}},
                "required": ["directory"],
            },
        },
    },
]


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


def run_agent(client: OpenAI, task: str) -> None:
    """Run the agent loop until it stops calling tools or hits the step limit."""
    messages = [
        {"role": "system", "content": (
            "You are an agent that completes tasks using the provided tools. "
            "Work step by step. When you're done, respond with your final answer."
        )},
        {"role": "user", "content": task},
    ]

    for step in range(1, MAX_STEPS + 1):
        print(f"\n[Step {step}] ", end="", flush=True)
        content, tool_calls = stream_chat(client, messages)

        if not tool_calls:
            print()
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
            print(f"\n  -> {name}({args}): {result[:100]}")
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": result,
            })

    print("\nAgent stopped: hit max steps")


def main():
    client = OpenAI()
    task = (
        "Look at the files in the 'workspace' directory. "
        "Read each one, then write a summary.txt that summarizes everything you found."
    )

    print(f"Task: {task}\n")
    run_agent(client, task)


if __name__ == "__main__":
    main()
