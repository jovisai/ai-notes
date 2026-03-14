---
title: "Code Writing Agents and Program Synthesis Teaching AI to Build Its Own Tools"
date: 2026-02-25
draft: false
tags: ["ai-agents", "program-synthesis", "code-generation", "repl", "swe-agent", "tool-use"]
description: "How AI agents generate, execute, and refine code as a reasoning medium, from classical program synthesis to modern REPL-based agent loops and SWE-bench architectures"
---

Code is the most powerful tool an AI agent can wield. Unlike a database query or an API call, code is universal: it can transform data, automate tasks, call other tools, and even modify the agent's own environment. When an agent learns to write and run code as a first-class reasoning step, it is no longer limited to pre-defined tools. This article explores how code-writing agents work, where they came from, and how to build one yourself.

## Concept Introduction

## Concept Introduction

A code-writing agent works in a loop:

1. Receive a task in natural language
2. Write code that would solve it
3. Execute the code in a sandboxed environment
4. Observe the output (or the error)
5. Revise the code until it works
6. Return the result

This is the **REPL loop** (Read-Eval-Print Loop) applied to agent cognition.

Code-writing agents combine two capabilities: natural language to code translation (like GitHub Copilot) and agentic execution loops (like ReAct). The key insight is that code provides a *verifiable, executable intermediate representation*. Instead of reasoning in unstructured text that might be wrong, the agent produces code that either runs correctly or fails with a precise error message. That's a much tighter feedback signal.

The agent's context window typically contains:
- The task description
- Prior code attempts and their outputs/errors
- Tool definitions (APIs it can call within the code)
- Accumulated observations

## Historical & Theoretical Context

Program synthesis (automatically generating programs from specifications) is a decades-old field. Early milestones include:

- **1969: Waldinger & Lee**: first formal synthesis system using resolution theorem proving
- **1986: Manna & Waldinger**: deductive synthesis from logical specifications
- **2006: Sketch (Armando Solar-Lezama)**: syntax-guided synthesis with human-provided program skeletons
- **2015: DeepCoder (Microsoft)**: neural networks predicting which library functions a solution uses
- **2021: Codex (OpenAI)**: GPT-3 fine-tuned on GitHub code, passing HumanEval benchmarks
- **2022: AlphaCode (DeepMind)**: competitive programming at median human level on Codeforces

The modern shift is that synthesis is no longer purely offline. Agents execute, observe, and revise in a live loop rather than generating once and hoping for the best, closer to how human programmers actually work.

## Algorithms & Math

### The REPL Agent as a Search Problem

Formally, a code-writing agent solves a search problem over program space. Let $P$ be the space of all programs, $o(p)$ the observed output of executing program $p$, and $\text{goal}(o)$ a binary signal of task success. The agent seeks:

$$p^* = \arg\max_{p \in P} \text{goal}(o(p))$$

At each step $t$, the LLM generates a candidate program conditioned on history:

$$p_t \sim \pi_\theta(\cdot \mid x, p_1, o_1, \ldots, p_{t-1}, o_{t-1})$$

where $x$ is the task, $p_i$ are prior attempts, and $o_i$ are their outputs.

This is an **online search** guided by error signals rather than a blind generation. The history of (attempt, outcome) pairs is the "trace" that accumulates in context.

### Self-Debugging Update Rule

When $o_t$ contains a traceback, the agent uses the error as a negative signal. The implicit update is:

$$p_{t+1} \sim \pi_\theta(\cdot \mid x, p_t, \text{error}(o_t))$$

Researchers have shown that simply including the error message in context improves fix rates dramatically, an emergent form of **gradient-free policy improvement** driven by execution feedback.

### Pseudocode: REPL Agent Loop

```python
def repl_agent(task: str, max_attempts: int = 5) -> str:
    history = []
    for attempt in range(max_attempts):
        code = llm_generate(task, history)
        output, error = sandbox_execute(code)
        history.append((code, output, error))
        if not error and task_success(output, task):
            return output
        # LLM sees the error on next iteration
    return "Max attempts reached"
```

## Design Patterns & Architectures

### The Code-Act Pattern

Popularized by the **CodeAct** paper (Wang et al., 2024), this approach replaces JSON tool calls with Python code as the action medium:

```
Traditional agent:  think → call tool(name="search", args={...}) → observe
CodeAct agent:      think → execute code(import search; result = search(...)) → observe
```

Code-Act agents are more flexible because they can compose tools, use loops, and transform intermediate results within a single action step.

### Architecture Overview

```
┌─────────────────────────────────────────────────┐
│                   LLM Planner                   │
│  (task → reasoning → code generation)           │
└────────────────────┬────────────────────────────┘
                     │ code string
                     ▼
┌─────────────────────────────────────────────────┐
│              Sandboxed Executor                 │
│  (Docker / E2B / modal.com / subprocess)        │
│  - stdout/stderr capture                        │
│  - timeout + memory limits                      │
│  - filesystem isolation                         │
└────────────────────┬────────────────────────────┘
                     │ output + error
                     ▼
┌─────────────────────────────────────────────────┐
│             Observation Formatter               │
│  (truncate long output, format tracebacks)      │
└────────────────────┬────────────────────────────┘
                     │
                     └──► back to LLM Planner
```

### Integration with Memory

Code-writing agents benefit enormously from episodic memory. If an agent successfully solved a file-parsing problem last week, it can retrieve that code snippet as a starting point. This connects directly to the **Skill Libraries** pattern, but now skills are discovered dynamically by writing code rather than pre-programmed.

## Practical Application

A minimal code-writing agent using Claude with the Anthropic SDK:

```python
import anthropic
import subprocess
import textwrap

client = anthropic.Anthropic()

SYSTEM = """You are a code-writing agent. When given a task:
1. Write Python code to solve it
2. Wrap the code in ```python ... ``` blocks
3. The code will be executed and results shown to you
4. Revise if needed"""

def extract_code(text: str) -> str | None:
    import re
    match = re.search(r"```python\n(.*?)```", text, re.DOTALL)
    return match.group(1) if match else None

def run_code(code: str, timeout: int = 10) -> tuple[str, str]:
    try:
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True, text=True, timeout=timeout
        )
        return result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return "", "TimeoutError: execution exceeded limit"

def code_agent(task: str, max_turns: int = 6) -> str:
    messages = [{"role": "user", "content": task}]

    for turn in range(max_turns):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=SYSTEM,
            messages=messages,
        )
        reply = response.content[0].text
        messages.append({"role": "assistant", "content": reply})

        code = extract_code(reply)
        if not code:
            return reply  # No code = final answer

        stdout, stderr = run_code(code)
        observation = f"Output:\n{stdout}" if stdout else f"Error:\n{stderr}"
        print(f"[Turn {turn+1}] {observation[:200]}")

        messages.append({"role": "user", "content": observation})

        if stdout and not stderr:
            return stdout  # Success

    return "Agent exhausted attempts"

# Example
result = code_agent("Download the top 5 HackerNews stories and print their titles")
print(result)
```

### SWE-Agent Style: Repository-Aware Code Agents

For software engineering tasks, agents need repository context. The **SWE-agent** pattern adds:

```python
# Tools available to the agent as callable Python functions
def read_file(path: str) -> str: ...
def write_file(path: str, content: str) -> None: ...
def run_tests(test_file: str) -> str: ...
def search_codebase(query: str) -> list[str]: ...

# Agent sees repo structure in context and calls these in generated code
```

SWE-agent achieved ~12% on SWE-bench (2024), later improved to ~43% by Devin-style systems combining long-horizon planning with code execution.

## Latest Developments & Research

### SWE-bench: The Software Engineering Benchmark (2023–2025)

SWE-bench (Princeton, 2023) tests agents on real GitHub issues from popular Python repositories. Progress has been dramatic:

- **2023**: ~3% resolution rate (naive GPT-4)
- **2024**: ~12% (SWE-agent), ~18% (Devin)
- **2025**: ~43–50% (best closed-source systems like Claude 3.7 in agentic mode)

This benchmark crystallized the research agenda around **long-horizon software engineering** with multi-file edits, test execution, and iterative debugging.

### CodeAct (Wang et al., 2024)

Demonstrated that replacing JSON tool calls with executable Python code as the action medium improves performance on 17 out of 20 agent benchmarks while requiring fewer turns. The key insight: code is a richer action representation than key-value parameter dicts.

### OpenDevin / All-Hands (2024)

An open-source platform for software development agents. Introduces **sandboxed workspaces** with persistent file systems, browser access, and terminal, giving agents a full developer environment rather than a single code execution cell.

### Program of Thoughts (Chen et al., 2022)

A prompting technique where the LLM writes Python code to perform mathematical and symbolic reasoning, then executes it to get exact answers. Outperforms chain-of-thought on arithmetic benchmarks by separating language reasoning from numerical computation.

### Frontier: Multi-Agent Code Review

Recent work (2025) explores multi-agent setups where one agent writes code, another reviews it, and a third writes tests, mirroring software engineering team structures. Early results show significant quality improvements over single-agent loops.

## Cross-Disciplinary Insight

Code-writing agents parallel the **scientific method**: hypothesize (write code), experiment (execute), observe (read output), revise (update code). Each execution either confirms or refutes the agent's model of the problem, applying Karl Popper's falsificationism to programming.

There's also a connection to **constructivist learning theory** (Piaget): learners construct knowledge through active experimentation rather than passive reception. Code-writing agents don't just predict answers. They *build* things and discover truth through action.

The REPL loop also mirrors **cybernetic control systems**: the error signal (stderr, failed tests) drives corrective action in a negative feedback loop. The agent is a controller minimizing the distance between current behavior and desired behavior, measured in executable test cases rather than reward scalars.

## Daily Challenge

**Exercise: Build a Self-Healing Data Pipeline**

Create a code-writing agent that:

1. Receives a messy CSV file path and a natural language description of the desired output (e.g., "sum sales by region, sorted descending")
2. Generates pandas code to process it
3. Executes the code
4. If it fails, reads the error and generates a fix
5. Repeats until success or 5 attempts

Starter scaffold:

```python
def data_agent(csv_path: str, goal: str) -> str:
    """
    Example goal: "Group by 'region' column, sum 'revenue', sort descending"
    The agent should:
    - Read the CSV schema on first attempt
    - Generate pandas transformation code
    - Handle errors like missing columns, wrong dtypes
    - Return the final output as a string
    """
    # Your implementation here
    pass

# Test with intentionally messy data
import pandas as pd
df = pd.DataFrame({
    "Region": ["North", "South", "North", "East"],
    "Revenue ": ["1000", "2000", "1500", "$500"],  # Note trailing space, $ sign
})
df.to_csv("/tmp/messy_sales.csv", index=False)

result = data_agent("/tmp/messy_sales.csv", "sum Revenue by Region, sort descending")
print(result)
```

**Bonus**: Add a memory layer that stores successful code patterns by task type. On the next similar task, inject the relevant snippet into the agent's context.

## References & Further Reading

### Papers
- **"Executable Code Actions Elicit Better LLM Agents"** (Wang et al., 2024), the CodeAct paper: https://arxiv.org/abs/2402.01030
- **"SWE-bench: Can Language Models Resolve Real-World GitHub Issues?"** (Jimenez et al., 2023): https://arxiv.org/abs/2310.06770
- **"Program of Thoughts Prompting: Disentangling Computation from Reasoning"** (Chen et al., 2022): https://arxiv.org/abs/2211.12588
- **"Self-Debugging: Teaching LLMs to Debug Their Predicted Programs"** (Chen et al., 2023): https://arxiv.org/abs/2304.05128
- **"InterCode: Standardizing and Benchmarking Interactive Coding with Execution Feedback"** (Yang et al., 2023): https://arxiv.org/abs/2306.14898

### Tools & Frameworks
- **E2B Code Interpreter**: https://e2b.dev (secure cloud sandboxes for AI-generated code)
- **OpenDevin / All-Hands**: https://github.com/All-Hands-AI/OpenHands (open-source software engineering agent platform)
- **SWE-agent**: https://github.com/princeton-nlp/SWE-agent (agent for resolving GitHub issues)
- **Modal Labs**: https://modal.com (serverless sandboxed Python execution at scale)

### Blog Posts
- **"Coding Agents Are Getting Really Good"** (Simon Willison, 2024): practical overview of the SWE-bench landscape
- **"The unreasonable effectiveness of just executing code"** (Anthropic blog, 2024): why code execution matters for agent reliability

---
