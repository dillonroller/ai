# Getting Started

Everything you need to set up before running any of the projects.

## 1. Install Python

Check if you already have it:

```bash
python3 --version
```

If you see `Python 3.10` or higher, you're good. If not:

- **Mac**: `brew install python` (if you have Homebrew) or download from https://www.python.org/downloads/
- **Windows**: Download from https://www.python.org/downloads/ — **check "Add to PATH"** during install

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

You'll see `(.venv)` at the start of your terminal prompt — that means it's active. You need to activate it each time you open a new terminal.

## 3. Install dependencies

With the virtual environment active:

```bash
pip install openai python-dotenv
```

We'll add more packages in later projects, but this is all you need for now.

## Troubleshooting

**`python3: command not found`** — Python isn't installed or not in your PATH. Try `python` instead of `python3`, or reinstall Python.

**`No module named openai`** — You probably forgot to activate the venv. Run `source .venv/bin/activate` (Mac) or `.venv\Scripts\activate` (Windows) first.

**`AuthenticationError`** — The API key in `.env` is missing or wrong. Make sure it starts with `sk-`.
