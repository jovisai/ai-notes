#!/usr/bin/env python3
"""
Transform all AI agent articles:
1. Remove ## Historical & Theoretical Context sections
2. Remove ## Daily Challenge sections
3. Replace Practical Application code blocks with a short description + Try it prompt
"""

import re
import subprocess
import sys
from pathlib import Path

ARTICLES_DIR = Path("/home/leopard/development/blogs/ai-notes/content/ai-agents")
CLAUDE_BIN = "/home/leopard/.local/bin/claude"


def remove_section(content: str, heading_pattern: str) -> str:
    """Remove a ## section whose heading matches heading_pattern, up to the next ## heading."""
    lines = content.split('\n')
    result = []
    skip = False

    for line in lines:
        if re.match(heading_pattern, line):
            skip = True
            continue
        if skip and re.match(r'^## ', line):
            skip = False
        if not skip:
            result.append(line)

    return '\n'.join(result)


def transform_practical_application(content: str, filepath: Path) -> str:
    """Replace code blocks in Practical Application with a Try it prompt via claude CLI."""

    pa_match = re.search(
        r'(## Practical Application\n)(.*?)(?=\n## |\Z)',
        content,
        re.DOTALL
    )
    if not pa_match:
        print(f"  [skip] no Practical Application section")
        return content

    pa_body = pa_match.group(2)

    if '```' not in pa_body:
        print(f"  [skip] no code blocks found")
        return content

    topic = filepath.stem.replace('-', ' ')

    prompt = f"""You are editing a technical blog article about: {topic}

Here is the current "Practical Application" section body (the ## heading is NOT included):

{pa_body}

Rewrite this section body with:
1. A 3-5 sentence description of what a minimal implementation would do — name the key classes/functions, the data flow, and the best-fit framework (LangGraph, CrewAI, AutoGen, raw Anthropic SDK, etc.).
2. A **Try it** heading followed by a plain fenced block (no language tag) containing a single tight 4-8 line prompt the reader can paste directly into a coding agent (Claude Code, Cursor, Copilot, etc.). The prompt must name the framework, describe the core behavior, and ask for runnable code with inline comments.

Rules:
- Do NOT include any code listings of your own.
- Output ONLY the replacement section body (no ## heading, no preamble, no trailing commentary).
- End with the Try it block in exactly this format:

**Try it**

```
[the prompt text here]
```"""

    result = subprocess.run(
        [CLAUDE_BIN, "--dangerously-skip-permissions", "-p", prompt],
        capture_output=True,
        text=True,
        timeout=60
    )

    if result.returncode != 0:
        print(f"  [error] claude CLI failed: {result.stderr[:200]}")
        return content

    new_body = result.stdout.strip()

    if not new_body:
        print(f"  [error] empty response from claude CLI")
        return content

    before = content[:pa_match.start()]
    after = content[pa_match.end():]   # starts with \n## or is empty
    new_content = before + '## Practical Application\n\n' + new_body + '\n' + after

    return new_content


def process_file(filepath: Path) -> None:
    print(f"Processing {filepath.name} ...")
    content = filepath.read_text(encoding='utf-8')
    original = content

    # 1. Remove Historical & Theoretical Context
    content = remove_section(content, r'^## Historical')

    # 2. Remove Daily Challenge
    content = remove_section(content, r'^## Daily Challenge')

    # 3. Transform Practical Application
    content = transform_practical_application(content, filepath)

    # Normalise excessive blank lines (max 2 consecutive newlines)
    content = re.sub(r'\n{3,}', '\n\n', content)

    if content != original:
        filepath.write_text(content, encoding='utf-8')
        print(f"  Saved.")
    else:
        print(f"  No changes.")


def main():
    md_files = sorted(f for f in ARTICLES_DIR.glob("*.md") if f.name != '_index.md')
    print(f"Found {len(md_files)} articles.\n")

    errors = []
    for filepath in md_files:
        try:
            process_file(filepath)
        except Exception as e:
            print(f"  ERROR: {e}")
            errors.append((filepath.name, str(e)))

    print(f"\nDone. {len(errors)} errors.")
    for name, err in errors:
        print(f"  {name}: {err}")


if __name__ == '__main__':
    main()
