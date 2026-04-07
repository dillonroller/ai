# Function Calling

Let the LLM call tools you define. Instead of just generating text, the model can decide to call a function, get the result, and use it to answer your question.

This is how ChatGPT plugins, web browsing, and code execution work.

## How to run

Make sure you've done the setup in the root [README](../README.md), then:

```bash
cd function-calling
python main.py
```

## How it works

1. You define tools as regular Python functions (e.g. `get_weather`, `calculate`)
2. You describe them as JSON schemas so the LLM knows what's available
3. When you ask a question, the LLM decides whether it needs a tool
4. If it does, it returns a tool call with arguments instead of text
5. You run the function yourself and send the result back
6. The LLM uses that result to write its final answer

The key insight is that the LLM never runs code. It just decides what to call and with what arguments. You control the execution.

## Things to try

- Add a new tool (e.g. a stock price lookup, a unit converter)
- Ask a question that requires multiple tool calls
- Ask something that doesn't need any tools and see it respond normally
