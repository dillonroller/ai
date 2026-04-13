# Local Models

Run open-source LLMs on your own machine with Ollama. No API key, no costs, no internet required after the download.

Companies care about this because it lets them avoid OpenAI dependency, keep data private, and cut costs at scale.

## Setup

1. Install Ollama: https://ollama.com/download (or `brew install ollama` on Mac)
2. Pull a model:
   ```bash
   ollama pull llama3.2
   ```
3. Make sure the Ollama service is running:
   ```bash
   ollama serve
   ```

Then run the project:

```bash
cd local-models
python main.py
```

## How it works

Ollama exposes an OpenAI-compatible API at `http://localhost:11434/v1`. That means you can use the same `openai` Python library you already know, just with a different base URL:

```python
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)
```

Any project in this repo can run on local models by changing those two lines.

## Models to try

- `llama3.2` - Meta's model, 3B params, fast on any laptop
- `qwen2.5` - very strong for code
- `mistral` - fast and capable
- `deepseek-r1` - reasoning model, thinks before answering
- `nomic-embed-text` - for embeddings (drop-in replacement in the RAG project)

## Tradeoffs vs OpenAI

- Free and private
- Works offline
- Slower than OpenAI's datacenters
- Smaller models are noticeably worse at complex tasks (function calling, structured outputs, long reasoning)

## Things to try

- Swap the RAG project to use `nomic-embed-text` and `llama3.2` instead of OpenAI
- Pull a larger model (`ollama pull qwen2.5:14b`) and see if it's noticeably better
- Time the same prompt across 3 different local models
