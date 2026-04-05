# Agents

An agent is an LLM in a loop that decides what to do next on its own. You give it tools and a goal, and it figures out the steps.

This is the architecture behind things like ChatGPT deep research, Devin, and Cursor.

## How to run

Make sure you've done the setup in the root [README](../README.md), then:

```bash
cd agents
python main.py
```

## How it works

1. You give the agent a task and a set of tools (read file, write file, list files)
2. The LLM decides which tool to call and with what arguments
3. You run the tool and send the result back
4. The LLM decides the next step based on what it learned
5. This loops until the LLM responds with text instead of a tool call

The difference from function calling: in function calling, the LLM makes one decision. An agent makes many decisions in a row, building on the results of each step.

## Things to try

- Add more files to `workspace/` and see how the agent handles them
- Change the task to something more complex
- Add new tools (e.g. search within a file, rename a file, run a shell command)
- Lower `MAX_STEPS` and see what happens when the agent runs out of steps
