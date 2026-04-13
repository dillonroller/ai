# Agents

An agent is an LLM in a loop that decides what to do next on its own. You give it tools and a goal, and it figures out the steps.

This is the architecture behind things like ChatGPT deep research, Devin, and Cursor.

## Three examples in this folder

1. **`main.py`** - Basic agent that uses file tools to complete a task
2. **`react_agent.py`** - ReAct pattern: the agent explicitly writes out its reasoning before every action
3. **`multi_agent.py`** - Multiple agents with different roles (researcher, writer, critic) working together

## How to run

Make sure you've done the setup in the root [README](../README.md), then:

```bash
cd agents
python main.py           # basic agent
python react_agent.py    # ReAct pattern
python multi_agent.py    # multi-agent collaboration
```

## How it works

### Basic agent (`main.py`)
1. You give the agent a task and a set of tools
2. The LLM decides which tool to call
3. You run the tool and send the result back
4. The LLM decides the next step based on what it learned
5. Loops until the LLM responds with text instead of a tool call

### ReAct (`react_agent.py`)
Same loop, but the agent writes out its reasoning in natural language before every tool call. This makes it better at complex problems and way easier to debug (you can literally see what it's thinking).

### Multi-agent (`multi_agent.py`)
Instead of one agent, you have several with different roles and system prompts. They pass work between each other like coworkers. Used for things that benefit from specialization: research + writing + review, planning + execution, etc.

## Things to try

- Add more files to `workspace/` and see how the basic agent handles them
- Give the ReAct agent a harder multi-step task and watch its reasoning
- Add a fourth "fact-checker" agent to the multi-agent pipeline
- Add new tools (search within a file, rename a file, run a shell command)
