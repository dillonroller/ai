# Chatbot

A simple conversational AI in the terminal. It remembers what you've said within the session.

## How to run

Make sure you've done the setup in [SETUP.md](../SETUP.md), then:

```bash
cd chatbot
python main.py
```

Type your messages and press Enter. Type `quit` to exit.

## How it works

Each message you send gets appended to a list alongside the AI's responses. The entire conversation history is sent with every request — that's how the model "remembers" context. This is the simplest architecture for a chatbot and the foundation for everything more complex.

## Things to try

- **Change the `SYSTEM_PROMPT`** — make it a pirate, a tutor, a debate opponent
- **Have a long conversation** — notice it remembers earlier messages
- **Ask it to recall something you said earlier** — see where context limits kick in
