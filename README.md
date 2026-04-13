# Getting Started

Everything you need to set up before running any of the projects.

## 1. Install Python

Check if you already have it:

```bash
python3 --version
```

If you see `Python 3.10` or higher, you're good. If not:

- **Mac**: `brew install python` (if you have Homebrew) or download from https://www.python.org/downloads/
- **Windows**: Download from https://www.python.org/downloads/ and check "Add to PATH" during install

## 2. Set up a virtual environment

A virtual environment keeps this project's packages separate from everything else on your computer. You only need to do this once.

Open a terminal, navigate to this repo, and run:

```bash
python3 -m venv .venv
```

This creates a `.venv/` folder. Now activate it:

- **Mac/Linux**:
  ```bash
  source .venv/bin/activate
  ```
- **Windows**:
  ```bash
  .venv\Scripts\activate
  ```

You'll see `(.venv)` at the start of your terminal prompt when it's active. You need to activate it each time you open a new terminal.

## 3. Install dependencies

With the virtual environment active:

```bash
pip install openai python-dotenv chromadb pydantic
```

## 4. Get an OpenAI API key

1. Go to https://platform.openai.com/api-keys and sign in (or create an account)
2. Click "Create new secret key" and give it a name (e.g. `ai-learning`)
3. Copy the key, you won't be able to see it again
4. Go to Settings > Billing and add a payment method + load some credits ($5 is plenty)

## 5. Set up your .env file

Copy the example and paste in your key:

```bash
cp .env.example .env
```

Then open `.env` and replace the placeholder with your actual key:

```
OPENAI_API_KEY=sk-your-key-here
```

## 6. Run a project

| Project | What it does |
|---------|-------------|
| [`chatbot/`](chatbot/) | Conversational AI in the terminal |
| [`rag/`](rag/) | Answer questions using your own documents |
| [`function-calling/`](function-calling/) | Let the LLM call tools (APIs, calculators, etc.) |
| [`structured-outputs/`](structured-outputs/) | Force the LLM to return typed JSON you can use in code |
| [`agents/`](agents/) | LLM in a loop that decides its own steps (basic, ReAct, multi-agent) |
| [`evaluation/`](evaluation/) | Measure whether your AI is actually good |
| [`fine-tuning/`](fine-tuning/) | Train a model on your own examples |
| [`local-models/`](local-models/) | Run open-source models locally with Ollama |
| [`memory/`](memory/) | A chatbot that remembers you across sessions |
| [`mcp-server/`](mcp-server/) | Expose tools and data to any LLM via the Model Context Protocol |

Pick one and run it:

```bash
cd chatbot
python main.py
```

## Troubleshooting

`python3: command not found` - Python isn't installed or not in your PATH. Try `python` instead of `python3`, or reinstall.

`No module named openai` - You forgot to activate the venv. Run `source .venv/bin/activate` (Mac) or `.venv\Scripts\activate` (Windows) first.

`AuthenticationError` - The API key in `.env` is missing or wrong. Make sure it starts with `sk-`.

`insufficient_quota / 429 error` - You need to add credits at https://platform.openai.com/settings/organization/billing/overview.
