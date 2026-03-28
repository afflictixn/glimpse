"""
Local coding agent powered by Gemma 3 1B via Ollama.

Uses a minimal agent loop designed for small models — no heavy framework
protocol to follow. The model gets a simple system prompt and can call tools
via ACTION blocks or write Python in CODE blocks.

Usage:
    conda_ai
    python agent.py "Your task here"
    python agent.py  # interactive mode
"""

import argparse
import json
import os
import re
import subprocess
import sys

import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma3:12b"
MAX_STEPS = 6

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

def tool_list_files(directory: str = ".") -> str:
    try:
        entries = os.listdir(directory)
        lines = []
        for e in sorted(entries):
            p = os.path.join(directory, e)
            lines.append(f"{'[DIR] ' if os.path.isdir(p) else '      '}{e}")
        return "\n".join(lines) or "(empty directory)"
    except Exception as exc:
        return f"Error: {exc}"


def tool_read_file(filepath: str) -> str:
    try:
        with open(filepath) as f:
            return f.read()
    except Exception as exc:
        return f"Error: {exc}"


def tool_write_file(filepath: str, content: str) -> str:
    try:
        with open(filepath, "w") as f:
            f.write(content)
        return f"OK — wrote {len(content)} chars to {filepath}"
    except Exception as exc:
        return f"Error: {exc}"


def tool_run_shell(command: str) -> str:
    try:
        r = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
        out = r.stdout
        if r.stderr:
            out += f"\nSTDERR:\n{r.stderr}"
        if r.returncode != 0:
            out += f"\n(exit code {r.returncode})"
        return out.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: timed out after 30s"
    except Exception as exc:
        return f"Error: {exc}"


def tool_run_python(code: str) -> str:
    try:
        r = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=30,
        )
        out = r.stdout
        if r.stderr:
            out += f"\nSTDERR:\n{r.stderr}"
        if r.returncode != 0:
            out += f"\n(exit code {r.returncode})"
        return out.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: timed out after 30s"
    except Exception as exc:
        return f"Error: {exc}"


TOOLS = {
    "list_files": tool_list_files,
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "run_shell": tool_run_shell,
    "run_python": tool_run_python,
}

SYSTEM_PROMPT = """\
You are a coding assistant with access to the local filesystem and shell.

When you need to do something, write Python code inside a ```python block.
I will execute it and show you the output. Examples:

To list files:
```python
import os
print(os.listdir("."))
```

To read a file:
```python
with open("main.py") as f:
    print(f.read())
```

To run a shell command:
```python
import subprocess
print(subprocess.run(["ls", "-la"], capture_output=True, text=True).stdout)
```

Rules:
- Write ONE ```python block per reply when you need to run code.
- After seeing the output, continue reasoning or write more code.
- When done, give your final answer in plain text with NO code block.
- Be concise.
"""

# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

CODE_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)


def chat(messages: list[dict]) -> str:
    resp = requests.post(
        OLLAMA_URL,
        json={"model": MODEL, "messages": messages, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def run_agent(task: str, verbose: bool = True) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]

    for step in range(1, MAX_STEPS + 1):
        if verbose:
            print(f"\n{'━'*60} Step {step} {'━'*60}")

        reply = chat(messages)
        if verbose:
            print(f"[Model]:\n{reply}")

        messages.append({"role": "assistant", "content": reply})

        match = CODE_RE.search(reply)
        if not match:
            if verbose:
                print("(no code block — treating as final answer)")
            return reply

        code = match.group(1).strip()
        if verbose:
            print(f"[Executing]:\n{code}")

        result = tool_run_python(code)
        if verbose:
            preview = result[:800] + ("..." if len(result) > 800 else "")
            print(f"[Output]:\n{preview}")

        messages.append({"role": "user", "content": f"Output:\n{result}"})

    return messages[-1]["content"] if messages else "(no answer)"


def main():
    parser = argparse.ArgumentParser(description="Local coding agent (Gemma 3 1B)")
    parser.add_argument("task", nargs="?", help="Task to run (omit for interactive mode)")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress step output")
    args = parser.parse_args()

    verbose = not args.quiet
    print(f"Agent ready — {MODEL} via Ollama\n")

    if args.task:
        result = run_agent(args.task, verbose=verbose)
        print(f"\n{'='*60}")
        print(f"Final answer:\n{result}")
    else:
        print("Interactive mode — type 'quit' to exit.\n")
        while True:
            try:
                task = input(">>> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if not task or task.lower() in ("quit", "exit", "q"):
                print("Bye!")
                break
            result = run_agent(task, verbose=verbose)
            print(f"\nFinal answer:\n{result}\n")


if __name__ == "__main__":
    main()
