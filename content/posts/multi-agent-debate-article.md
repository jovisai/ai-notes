---
title: "Building a Multi-Agent Debate System with LangGraph"
date: 2025-10-22
tags: ["ai", "multi-agent-systems", "langgraph", "python"]
---

## Introduction

Multi-agent systems are a fascinating area of AI development, where multiple autonomous agents collaborate or compete to solve complex problems. One interesting application of this paradigm is the concept of a "debate," where different AI agents take on distinct personas to argue a topic from various viewpoints. This approach can help to uncover nuanced perspectives and lead to more robust and well-reasoned conclusions.

In this article, we'll explore a minimal multi-agent debate system built with Python and the powerful `langgraph` library. This system orchestrates a debate between two AI agents, with a third agent acting as a judge to determine the most accurate answer. We'll dive into the product requirements, the system architecture, and the implementation details.

## Project Overview

The goal of this project is to create a lightweight system that can host a structured debate on a given question. The system is designed to be simple, with a clear and predictable flow. The core idea is to leverage the different perspectives of the AI agents to produce a more comprehensive answer than a single agent could provide.

### System Architecture

The system is composed of four main components:

*   **Two Debater Agents:** These agents have distinct "personalities" or reasoning approaches. In this implementation, we have:
    *   **Agent 1 (Analytical):** Focuses on data, evidence, and logical reasoning.
    *   **Agent 2 (Philosophical):** Considers ethical implications, values, and broader perspectives.
*   **One Judge Agent:** This agent's role is to evaluate the arguments presented by the debaters and provide a final, reasoned judgment.
*   **A Moderator:** This is not an agent itself, but rather the orchestration logic that manages the debate flow.

The debate follows a structured format:

1.  **Opening Arguments:** Each debater presents their initial position.
2.  **Rebuttal Phase:** The debaters respond to each other's arguments for a set number of rounds (in this case, two).
3.  **Conclusion:** Each debater provides a final summary of their position.
4.  **Judgment:** The judge evaluates the entire debate and provides a final answer, along with reasoning and a confidence score.

## Implementation with `langgraph`

The orchestration of the debate is handled by `langgraph`, a library that allows you to build stateful, multi-agent applications. At its core, `langgraph` uses a state machine approach, where each step in the debate is a node in the graph.

The state of our debate is managed in a `GraphState` object, which is a TypedDict that holds information such as the current round, the debate log, and the messages from each agent.

Here's a simplified view of how the debate graph is constructed:

```python
workflow = StateGraph(GraphState)

# Add nodes for each debate phase
workflow.add_node("agent1_opening", self._agent1_opening)
workflow.add_node("agent2_opening", self._agent2_opening)
workflow.add_node("agent1_rebuttal1", self._agent1_rebuttal1)
workflow.add_node("agent2_rebuttal1", self._agent2_rebuttal1)
# ... and so on for all rounds
workflow.add_node("judge", self._judge)

# Define the debate flow
workflow.set_entry_point("agent1_opening")
workflow.add_edge("agent1_opening", "agent2_opening")
workflow.add_edge("agent2_opening", "agent1_rebuttal1")
# ... and so on, linking all the steps in sequence
workflow.add_edge("agent2_conclusion", "judge")
workflow.add_edge("judge", END)

self.graph = workflow.compile()
```

Each node in the graph is a function that takes the current state as input, invokes the appropriate agent, and returns an update to the state. This makes it easy to manage the flow of the debate and keep track of the conversation history.

## Agent Prompts

The behavior of each agent is guided by a carefully crafted system prompt. These prompts define the agent's persona, its role in the debate, and the guidelines it should follow.

### Debater 1 (Analytical)

```
You are Agent 1 in a structured debate. Your approach is analytical and data-driven.

Your role:
- Present arguments based on empirical evidence, statistics, and logical reasoning
- Focus on practical implications and real-world outcomes
- Challenge opposing arguments with concrete counterexamples
- Remain objective and fact-focused in your analysis
```

### Debater 2 (Philosophical)

```
You are Agent 2 in a structured debate. Your approach is philosophical and ethical.

Your role:
- Consider broader implications, values, and long-term consequences
- Examine underlying assumptions and principles
- Explore edge cases and potential unintended effects
- Consider multiple stakeholder perspectives
```

### Judge

```
You are an impartial judge evaluating a debate between two agents.

Your task:
- Carefully review all arguments presented by both debaters
- Evaluate the logical consistency, evidence quality, and persuasiveness of each side
- Determine which position is most accurate or well-supported
- Provide a clear judgment with detailed reasoning
```

These prompts are combined with the debate question and the conversation history to generate the full prompt for each agent at each step of the debate.

## Input and Output

The system is designed to be simple to use, with a single API endpoint for initiating a debate. The input is a JSON object containing the debate question and an optional context string.

### Input

```json
{
  "question": "Is nuclear energy a good solution to climate change?",
  "context": "Optional background information relevant to the question"
}
```

The output is also a JSON object, which includes the complete debate log, the final judgment from the judge, the reasoning behind the judgment, and a confidence score.

### Output

```json
{
  "debate_log": [
    {"agent": "agent1", "message": "...", "round": 1},
    {"agent": "agent2", "message": "...", "round": 1},
    ...
  ],
  "final_judgment": "Based on the arguments presented, nuclear energy is...",
  "reasoning": "The judge's reasoning for the decision",
  "confidence": 0.85
}
```

## How to Run the System

Running a debate is straightforward. You simply instantiate the `DebateOrchestrator` and call the `run_debate` method with your question.

```python
from src.orchestrator import DebateOrchestrator
from src.models import DebateInput

# Initialize the orchestrator
orchestrator = DebateOrchestrator()

# Define the debate input
debate_input = DebateInput(
    question="Is nuclear energy a good solution to climate change?"
)

# Run the debate
result = orchestrator.run_debate(debate_input)

# Print the results
print(result.final_judgment)
print(result.reasoning)
```

This minimal multi-agent debate system demonstrates how `langgraph` can be used to create sophisticated, multi-agent workflows. By orchestrating a debate between agents with different perspectives, we can generate more comprehensive and well-reasoned answers to complex questions.

While this system is simple, it provides a solid foundation for more advanced applications. Future improvements could include:

*   **Tool Use:** Allowing agents to access external tools for fact-checking or data retrieval.
*   **Dynamic Routing:** Implementing more complex logic to control the flow of the debate based on the agents' responses.
*   **More Agents:** Adding more agents with different personas to further enrich the debate.

Multi-agent systems are a powerful tool for tackling complex problems, and the debate paradigm is a compelling way to leverage their potential.
