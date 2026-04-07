# Evaluation

How do you know if your AI is actually good? You write test cases with expected outputs, run them through the model, and measure how many pass.

This is one of the most underrated skills in AI right now. Companies are drowning in LLM products they can't measure.

## How to run

Make sure you've done the setup in the root [README](../README.md), then:

```bash
cd evaluation
python main.py
```

## Three ways to evaluate

1. **Exact match** - the answer has to exactly match the expected string. Brittle but simple.
2. **Contains match** - the answer has to contain the expected substring. Looser.
3. **LLM-as-judge** - use another LLM to decide if the answer is good. Flexible, handles open-ended questions, but more expensive and has its own biases.

Real eval setups usually combine all three depending on the question type.

## Why this matters

Without evaluation you're flying blind. Every time you tweak a prompt or swap a model, you have no idea if it got better or worse. A solid eval suite lets you:

- Compare models (is gpt-4o-mini good enough or do you need gpt-4o?)
- Catch regressions when prompts change
- Measure progress on improvements
- Give stakeholders real numbers instead of vibes

## Things to try

- Add your own test cases
- Run the same tests against different models and compare scores
- Change a prompt and see if scores improve
- Add a new eval type (semantic similarity using embeddings)
