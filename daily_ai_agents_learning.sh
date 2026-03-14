#!/usr/bin/env bash
set -euo pipefail

cd /home/leopard/development/blogs/ai-notes/ || exit 1
POSTDIR="content/ai-agents"

# Build list of already-covered topics from existing filenames
COVERED_TOPICS=$(ls "$POSTDIR"/*.md 2>/dev/null | xargs -I{} basename {} .md | sed 's/-/ /g' | sed 's/^/- /' | sort)

PROMPT="
**Role:**
You are my mentor and research assistant. Write a short, focused article (800–1200 words max) that I can read in 10–20 minutes, well-structured article that helps me master the field of AI agent programming step by step.

**Scope:**
Each article should focus on **one key concept, algorithm, pattern, or recent research trend** in AI agent programming. The article must contain the below points wherever relevant:

- **Concept Introduction**: explain the concept simply, then expand into technical detail for a practitioner.
- **Historical & Theoretical Context**: origin of the idea (who/when/why) and relation to core AI principles.
- **Algorithms & Math (if relevant)**: pseudocode or formulas where useful, broken down step by step.
- **Design Patterns & Architectures**: how the concept fits into real agent architectures and known patterns.
- **Practical Application**: a small coding example (Python preferred), shown in a real agent framework (LangGraph, CrewAI, AutoGen, Swarm, etc).
- **Latest Developments & Research**: recent papers, benchmarks, or breakthroughs (past 2–3 years) and open problems.
- **Cross-Disciplinary Insight**: relate the idea to another field (systems theory, neuroscience, distributed computing, economics, etc).
- **Daily Challenge / Thought Exercise**: a short problem or thought experiment completable in under 30 minutes.
- **References & Further Reading**

* Cite papers by **title, authors, venue, and year** — do not generate URLs.
* Only include URLs for well-known, major repos where you are certain the URL is correct (e.g. \`github.com/openai/...\`, \`github.com/google-deepmind/...\`). When in doubt, omit the URL entirely.

**Tone & Style:**

* Write in a clear, engaging, human style — like a knowledgeable colleague explaining something at a whiteboard, not a textbook.
* Use short paragraphs, lists, and mermaid diagrams when useful.
* Vary sentence length deliberately. Mix short punchy sentences with longer ones. Monotone rhythm is an AI tell.
* Do NOT use em-dashes (—) anywhere in the article. Use commas, periods, or parentheses instead.
* Do NOT open with a metaphor or hook paragraph designed to "engage the reader." Start with the substance of the topic directly.
* Do NOT include a "Key Takeaways" or summary section at the end. End with the References section.
* Do NOT reproduce the section framework labels (e.g. "Simple Explanation", "Technical Detail") as sub-headers — just write the content flowing naturally within each section.
* Do NOT label design patterns as "Pattern: X". Describe them in prose.
* Bold sparingly — at most one or two terms per section that are truly critical. Do not bold every technical term on first mention.
* Avoid triple parallel lists ("X, Y, and Z" constructions used repeatedly). Vary your sentence structures.
* Do not wrap up paragraphs with tidy bow phrases like "exactly the architecture proposed in..." or "this is X in action." Just make the point and move on.

**Example Topics for Rotation:**

* Classical algorithms: A*, Minimax, Reinforcement Learning basics
* Agent reasoning loops: Planner–Executor–Memory patterns
* Multi-agent coordination: Auctions, Game theory, Swarm intelligence
* Modern frameworks: LangGraph, CrewAI, AutoGen, OpenAI Swarm
* Tool use & orchestration: Function calling, memory, retrieval
* Advanced topics: Self-play, Autoformalization, AI Safety in agents
* Research updates: new benchmarks, emerging architectures

**Topics already covered — do NOT write about these or close variants:**

$COVERED_TOPICS

- Look at the format of the article /home/leopard/development/blogs/ai-notes/content/ai-agents/agent-debugging-and-observability.md to understand how the blogs are structured in this blog.
- Ensure the title is a single sentence without special characters.
- Do not repeat or copy existing articles inside $POSTDIR.
- Save the file as {article-title}.md in all lowercase inside $POSTDIR.
- **Math & LaTeX formatting:** This blog uses KaTeX for math rendering. NEVER put math expressions inside code blocks (\`\`\` or backticks). Use \$...\$ for inline math (e.g. \$O(T^2)\$, \$\\pi(a|s)\$) and \$\$...\$\$ on their own lines for display math. Pseudocode and code examples stay in code blocks — only mathematical notation uses LaTeX."

# Snapshot existing files before generation
BEFORE=$(ls "$POSTDIR"/*.md 2>/dev/null | sort)

/home/leopard/.local/bin/claude --dangerously-skip-permissions --verbose -p "$PROMPT"

# Detect the newly created file
AFTER=$(ls "$POSTDIR"/*.md 2>/dev/null | sort)
NEW_FILE=$(comm -13 <(echo "$BEFORE") <(echo "$AFTER") | head -1)

if [[ -z "$NEW_FILE" ]]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: No new article was created. Aborting." >&2
  exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] New article detected: $NEW_FILE"

/home/leopard/.local/bin/claude --dangerously-skip-permissions --verbose -p "Review the file $NEW_FILE and fix any errors:
1. **Math/LaTeX**: Ensure all math uses KaTeX syntax (\$...\$ inline, \$\$...\$\$ display). No math inside code blocks or backticks.
2. **Code blocks**: Ensure all code blocks have correct language tags and valid syntax.
3. **Mermaid diagrams**: Ensure all mermaid blocks use valid Mermaid.js syntax (correct node/edge definitions, no missing brackets or arrows).
4. **Markdown structure**: Fix broken links, unclosed formatting, or malformed tables.
Only edit the file if there are actual errors. Do not rewrite or restyle the content."

bash render_publish.sh

echo "[$(date '+%Y-%m-%d %H:%M:%S')] AI Agents Learning article generated successfully: $NEW_FILE"
