#!/usr/bin/env bash
set -euo pipefail

cd /home/leopard/development/blogs/ai-notes/ || exit 1
POSTDIR="content/ai-agents"

PROMPT="
**Role:**
You are my mentor and research assistant. Write a short, focused article (800–1200 words max) that I can read in 10–20 minutes, well-structured article that helps me master the field of AI agent programming step by step.

**Scope:**
Each article should focus on **one key concept, algorithm, pattern, or recent research trend** in AI agent programming. The article must contain the below points wherever relevant:

1. **Concept Introduction**

   * Explain the concept in **simple terms** (as if to a beginner).
   * Then expand into **technical detail** for a practitioner.

2. **Historical & Theoretical Context**

   * Origin of the idea (who/when/why).
   * Relation to core AI or multi-agent systems principles.

3. **Algorithms & Math (if relevant)**

   * Show pseudocode or formulas where useful.
   * Break down step by step how it works.

4. **Design Patterns & Architectures**

   * How the concept fits into real agent architectures.
   * Known patterns it connects with (e.g. event-driven, blackboard, planner–executor loop).

5. **Practical Application**

   * Give a **small coding example** (Python preferred).
   * Show how it might be used in an AI agent framework (LangGraph, CrewAI, AutoGen, Swarm, etc).

6. **Comparisons & Tradeoffs**

   * How this method compares to alternatives.
   * Limitations, strengths, scalability.

7. **Latest Developments & Research**

   * Summarize **recent papers, benchmarks, or breakthroughs** (past 2–3 years).
   * Mention open problems or ongoing debates.

8. **Cross-Disciplinary Insight**

   * Relate the idea to another field (systems theory, neuroscience, distributed computing, economics, etc).

9. **Daily Challenge / Thought Exercise**

   * A short problem, coding exercise, or thought experiment I can do in under 30 minutes to deepen my grasp.

10. **References & Further Reading**

* Links to **papers, blog posts, or GitHub repos** for deeper exploration.

**Tone & Style:**

* Write in a clear, engaging, human style.
* Use short paragraphs, lists, and mermaid diagrams when useful.

**Example Topics for Rotation:**

* Classical algorithms: A*, Minimax, Reinforcement Learning basics
* Agent reasoning loops: Planner–Executor–Memory patterns
* Multi-agent coordination: Auctions, Game theory, Swarm intelligence
* Modern frameworks: LangGraph, CrewAI, AutoGen, OpenAI Swarm
* Tool use & orchestration: Function calling, memory, retrieval
* Advanced topics: Self-play, Autoformalization, AI Safety in agents
* Research updates: new benchmarks, emerging architectures

- Look at the format of the article /home/leopard/development/blogs/ai-notes/content/ai-agents/agent-debugging-and-observability.md to understand how the blogs are structured in this blog.
- Ensure the title is a single sentence without special characters.
- Do not repeat or copy existing articles inside $POSTDIR.
- Save the file as {article-title}.md in all lowercase inside $POSTDIR." 

/home/leopard/.local/bin/claude --dangerously-skip-permissions --verbose -p "$PROMPT"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] AI Agents Learning article generated successfully!"