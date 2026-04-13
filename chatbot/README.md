# Chatbot

A simple conversational AI in the terminal. It remembers what you've said within the session.

## How to run

Make sure you've done the setup in the root [README](../README.md), then:

```bash
cd chatbot
python main.py
```

Type your messages and press Enter. Type `quit` to exit.

## How it works

Each message gets appended to a list alongside the AI's responses. The entire conversation history is sent with every request, which is how the model remembers context. This is the simplest possible chatbot architecture.

## Things to try

- Change the `SYSTEM_PROMPT` to give it a different personality
- Have a long conversation and notice it remembers earlier messages
- Ask it to recall something you said earlier
