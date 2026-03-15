---
title: "Working Memory Compression and Context Distillation in Long Horizon Agents"
date: 2026-03-15
draft: false
tags: ["ai-agents", "memory", "context-engineering", "langgraph", "multi-agent"]
description: "How long-running agents compress, distill, and selectively retain working memory to operate effectively within finite context windows"
---

Long-running agents break. Not because of bugs in the usual sense, but because they run out of room to think. An agent investigating a codebase, synthesizing research over dozens of documents, or managing a multi-day workflow eventually fills its context window with accumulated history. At that point, either the oldest observations get silently truncated (and the agent loses track of what it already tried), or the whole run crashes. Both outcomes are unacceptable for production systems.

## Concept Introduction

The core problem is that LLMs have fixed context windows, but agent tasks are often open-ended. Every tool call, observation, and reasoning step appends to the agent's working context. A task that seemed simple at step 3 looks very different at step 30, when the context is packed with partially-relevant tool outputs, retried subtasks, and obsolete intermediate conclusions.

**Working memory compression** refers to the set of techniques for actively managing what an agent keeps in its immediate context: what gets summarized, what gets evicted, what gets archived to external storage, and what gets retrieved on demand. The goal is to preserve the agent's ability to reason coherently over long horizons without blowing the context budget.

Three broad strategies emerge in practice. First, sliding window truncation: keep only the last N messages. Simple, but it amputates history arbitrarily, and the agent may re-attempt steps it already tried. Second, periodic summarization: after every K steps, an LLM call compresses the recent history into a compact summary. Third, hierarchical distillation: the agent maintains multiple memory tiers, with different retention policies at each level.

In real systems, you usually combine all three.

## Historical and Theoretical Context

The challenge mirrors how working memory is theorized in cognitive science. Miller's 1956 observation that human working memory holds roughly seven items (plus or minus two) sparked decades of research into how the brain manages limited attentional resources. The solution wasn't just truncation: humans chunk information, consolidate short-term memories into longer-term schemas, and selectively recall relevant context based on current goals.

The earliest practical analog in AI was paging in symbolic planners. SOAR (Laird, Newell, and Rosenbloom, 1987) introduced chunking as a mechanism to compile frequently-used reasoning patterns into long-term production rules, reducing the load on working memory. HTN planners similarly compressed sub-plan results into abstract plan steps that were referenced without being re-expanded.

For LLM agents, the problem became acute once people moved beyond single-turn use. Early AutoGPT experiments in 2023 showed agents spiraling into incoherence after 20-30 steps, repeatedly trying the same failed approaches because their context had been truncated. This sparked practical work on memory management that continues to evolve.

## Algorithms and Distillation Patterns

The simplest workable approach is the rolling summary pattern. After every K steps:

```
summary = LLM("Summarize the key findings, decisions, and
               outstanding questions from the following steps:
               {recent_steps}")
context = [initial_task, summary] + recent_steps[-K:]
```

This preserves the original task and a growing compressed history, while keeping the raw detail of recent steps available for reasoning.

A more sophisticated approach introduces relevance-scored retention. When the context nears capacity, each stored observation is scored by the agent for relevance to the current subtask, and low-scoring observations are either evicted or archived to a vector store for retrieval later. This lets the agent maintain a leaner working context while still being able to pull back specific memories when needed.

The key design decision is who does the relevance scoring. You can use the same LLM (expensive but accurate), a smaller classifier (cheap but coarser), or a similarity search against the current task embedding (fast but shallow). In practice, a hybrid works best: use embedding similarity for coarse filtering, then LLM scoring for the borderline cases.

## Design Patterns in Practice

Long-horizon agents typically compose several memory tiers:

```
Tier 1: Active context  (last 5-10 steps, always in prompt)
Tier 2: Episode buffer  (compressed summaries of earlier steps)
Tier 3: Archive store   (vector DB, retrieved on demand)
Tier 4: Permanent facts (task description, constraints, invariants)
```

Tier 4 is often underestimated. The agent's original task description, any constraints given by the user, and factual anchors (like "we are targeting Python 3.11") should never be evicted. Keeping these pinned at the front of the prompt prevents the agent from drifting off-task in ways that are hard to debug.

One pattern that works well is the checkpoint-resume loop. At regular intervals, the agent writes a structured checkpoint: a snapshot of what it has accomplished, what it still needs to do, and any dead ends it has ruled out. If the context fills before the task is done, the agent can start a fresh context window, reading the checkpoint rather than the full history. This mirrors how humans pause a complex task, write down where they are, and pick it up the next day.

## Practical Application

Here is a LangGraph implementation that demonstrates rolling summarization with a pinned task context and an archival store for evicted observations:

```python
import os
from typing import Annotated, TypedDict
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_anthropic import AnthropicEmbeddings
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# How many raw steps to keep before forcing a compression
COMPRESSION_THRESHOLD = 6
# How many recent steps remain uncompressed after a compression
RECENT_WINDOW = 3

class AgentState(TypedDict):
    task: str                          # Pinned, never evicted
    messages: Annotated[list, add_messages]
    episode_summary: str               # Rolling compressed history
    step_count: int
    archive: list[str]                 # Observations moved out of context

llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=2048)

def build_archive_store():
    """Vector store for archived observations, retrieved on demand."""
    # In production, replace with a persistent store
    store = {}
    return store

archive_store = build_archive_store()

def compress_episode(state: AgentState) -> AgentState:
    """Summarize older messages and evict them from active context."""
    messages = state["messages"]

    # Keep the most recent steps raw; compress everything older
    to_compress = messages[:-RECENT_WINDOW]
    to_keep = messages[-RECENT_WINDOW:]

    if not to_compress:
        return state

    # Build text for the summarizer
    history_text = "\n".join(
        f"{m.type.upper()}: {m.content}" for m in to_compress
    )

    prior_summary = state.get("episode_summary", "")

    summary_prompt = f"""You are summarizing an agent's work history.

Prior summary (if any):
{prior_summary}

New steps to incorporate:
{history_text}

Produce a concise but complete summary covering:
- What has been accomplished so far
- Key findings or decisions
- What has been ruled out or tried unsuccessfully
- What remains to be done

Keep it under 300 words."""

    response = llm.invoke([SystemMessage(content=summary_prompt)])
    new_summary = response.content

    # Archive the raw observations in case we need to retrieve them
    archive_entries = [m.content for m in to_compress]

    return {
        **state,
        "messages": to_keep,
        "episode_summary": new_summary,
        "archive": state.get("archive", []) + archive_entries,
    }

def should_compress(state: AgentState) -> str:
    """Decide whether to compress before the next reasoning step."""
    if len(state["messages"]) >= COMPRESSION_THRESHOLD:
        return "compress"
    return "reason"

def reason(state: AgentState) -> AgentState:
    """Main agent reasoning step with pinned task + episode summary in context."""
    task = state["task"]
    summary = state.get("episode_summary", "")

    # Build the system prompt, always including the pinned task
    system_parts = [f"Your task: {task}"]
    if summary:
        system_parts.append(f"\nWork done so far:\n{summary}")
    system_parts.append(
        "\nContinue working on the task. Think step by step. "
        "When done, say TASK COMPLETE followed by your final answer."
    )

    system = "\n".join(system_parts)
    messages = [SystemMessage(content=system)] + state["messages"]

    response = llm.invoke(messages)

    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "step_count": state.get("step_count", 0) + 1,
    }

def is_done(state: AgentState) -> str:
    """Check if the agent has declared the task complete."""
    last = state["messages"][-1] if state["messages"] else None
    if last and "TASK COMPLETE" in last.content:
        return "done"
    if state.get("step_count", 0) >= 20:  # hard cap
        return "done"
    return "continue"

# Build the graph
graph = StateGraph(AgentState)
graph.add_node("compress", compress_episode)
graph.add_node("reason", reason)

graph.set_entry_point("reason")
graph.add_conditional_edges("reason", is_done, {
    "done": END,
    "continue": should_compress,
})
graph.add_conditional_edges(
    # reuse the router as the branch from the compression check
    # (when coming fresh from entry we skip compress)
    "compress", lambda s: "reason", {"reason": "reason"}
)

# Patch: add direct edge from compress to reason
graph.add_edge("compress", "reason")

agent = graph.compile()

if __name__ == "__main__":
    result = agent.invoke({
        "task": "List 5 key differences between MARL and single-agent RL, "
                "then explain which matters most for cooperative robotics.",
        "messages": [HumanMessage(content="Begin the task.")],
        "episode_summary": "",
        "step_count": 0,
        "archive": [],
    })

    print("=== Episode Summary ===")
    print(result["episode_summary"])
    print(f"\n=== Steps taken: {result['step_count']} ===")
    print(f"Archived {len(result['archive'])} observations")

    last_msg = result["messages"][-1]
    if "TASK COMPLETE" in last_msg.content:
        answer = last_msg.content.split("TASK COMPLETE")[-1].strip()
        print(f"\n=== Final Answer ===\n{answer}")
```

The key architectural decision here is that `task` is a separate field in state, never mixed into the message list where it might get truncated. The `episode_summary` grows incrementally rather than being recomputed from scratch each time, which keeps compression costs proportional to recent history rather than total history.

## Latest Developments and Research

MemGPT (Packer et al., MemGPT: Towards LLMs as Operating Systems, NeurIPS 2023) formalized the idea of treating context management as a memory hierarchy problem explicitly inspired by OS virtual memory. The agent has a "main context" (analogous to RAM) and a "external storage" (analogous to disk), and manages page-ins and page-outs explicitly. MemGPT showed this could support conversations and agent tasks far longer than raw context limits.

Cognitive Architectures for Language Agents (CoALA, Sumers et al., TMLR 2024) provides a unifying taxonomy of memory in LLM agents, distinguishing working memory, episodic memory, semantic memory, and procedural memory. It is the most rigorous framework currently available for reasoning about memory design decisions.

More recent work examines compression quality. A compressed summary that omits a critical dead-end means the agent may try the same failed approach again. Evaluation metrics for compression fidelity (does the summary preserve the information needed for future steps?) are an active research gap.

## Cross-Disciplinary Insight

Operating systems solved a structurally similar problem decades ago with virtual memory and demand paging. The CPU's registers are a tiny, fast working memory; RAM is a larger but slower tier; disk is vast but slow. The OS moves pages between tiers based on recency and access patterns, giving processes the illusion of infinite memory.

Agent memory management is demand paging for cognition. The "page fault" equivalent is a retrieval call: when the agent needs context that has been evicted, it fetches it from the archive. The key insight borrowed from OS design is that you do not need to keep everything in fast memory simultaneously, only what is actively needed. The hard part is predicting what will be needed next, which in agent systems depends on the task structure in ways that LRU caching (the OS default) does not capture well.

## Daily Challenge

Take any multi-step agent you have built or can build quickly. Deliberately make it run 15-20 steps on a complex task and observe where it starts to degrade: does it re-attempt things it already tried? Does it lose track of the original task? Does the context fill and truncate silently?

Then implement one of: (a) a rolling summarization step after every 5 messages, or (b) a checkpoint write after every 8 steps. Measure whether the agent's final output quality improves by running both versions on the same task and comparing the outputs. Pay attention to whether the compressed version makes mistakes the full-context version avoids, and vice versa.

## References and Further Reading

- Packer, C., Fang, V., Patil, S. G., Moon, K., Kaufman, S., and Gonzalez, J. E. "MemGPT: Towards LLMs as Operating Systems." NeurIPS 2023 Workshop on Interactive Learning from Implicit Human Feedback.
- Sumers, T. R., Yao, S., Narasimhan, K., and Griffiths, T. L. "Cognitive Architectures for Language Agents." Transactions on Machine Learning Research, 2024.
- Park, J. S., O'Brien, J. C., Cai, C. J., Morris, M. R., Liang, P., and Bernstein, M. S. "Generative Agents: Interactive Simulacra of Human Behavior." UIST 2023.
- Wang, L., Ma, C., Feng, X., Zhang, Z., Yang, H., Zhang, J., Chen, Z., Tang, J., Chen, X., Lin, Y., Zhao, W. X., Wei, Z., and Wen, J.-R. "A Survey on Large Language Model based Autonomous Agents." Frontiers of Computer Science, 2024.
- Laird, J. E., Newell, A., and Rosenbloom, P. S. "SOAR: An Architecture for General Intelligence." Artificial Intelligence, 33(1), 1987.
- Miller, G. A. "The Magical Number Seven, Plus or Minus Two: Some Limits on Our Capacity for Processing Information." Psychological Review, 63(2), 1956.
