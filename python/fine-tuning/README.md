# Fine-Tuning

Fine-tuning takes an existing model and trains it on your own examples. Instead of explaining what you want in a prompt every time, you teach the model by showing it examples of input/output pairs.

## How to run

Make sure you've done the setup in the root [README](../README.md), then:

```bash
cd fine-tuning
python main.py
```

This will upload training data, start a fine-tuning job, wait for it to finish (5-15 minutes), then test the result. It costs a few cents.

## How it works

1. You write training examples as conversations (question + ideal answer)
2. The script formats them as JSONL and uploads to OpenAI
3. OpenAI trains a new version of the model on your examples
4. You get back a model ID you can use just like `gpt-4o-mini`

The example here trains a sarcastic tech support bot. With only 10 examples the model picks up the tone and style.

## When to fine-tune vs when to prompt

- **Prompting**: good enough for most things, no training cost, easy to change
- **Fine-tuning**: better when you need a consistent style/format across thousands of calls, or when your prompt would be too long

## Things to try

- Add more training examples and see if the style gets more consistent
- Train a model for a completely different persona
- Compare the fine-tuned model's responses vs the base model with the same prompt
