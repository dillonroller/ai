# Structured Outputs

Force the LLM to return JSON matching a schema you define. Instead of parsing text responses yourself, you get back a typed Python object.

This is the foundation of turning LLMs into data pipelines: give them messy text, get back clean structured data.

## How to run

Make sure you've done the setup in the root [README](../README.md), then install pydantic:

```bash
pip install pydantic
```

Then run:

```bash
cd structured-outputs
python main.py
```

## How it works

1. Define a Pydantic model describing the shape you want
2. Pass it as `response_format` to the API call
3. OpenAI forces the model to return JSON that exactly matches your schema
4. You get back a typed Python object, no parsing needed

The key word is "forces". This isn't hoping the model returns JSON. The API won't let it return anything else. No more regex parsing, no more "please return JSON" in the prompt.

## Why this matters

Structured outputs are the backbone of real AI products:

- Extracting data from emails, resumes, invoices, receipts
- Classifying support tickets into categories
- Turning natural language queries into database filters
- Generating test cases, form fields, API calls

Any time you need the LLM to produce something a program will consume (not a human will read), you want structured outputs.

## Things to try

- Add a new schema (event details, product listing, recipe, etc.)
- Extract multiple items at once with `list[Person]`
- Add enums using `Literal["option1", "option2"]`
- Nest schemas inside each other
