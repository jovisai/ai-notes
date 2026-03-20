---
title: "Corrigibility and Interruptibility Building Agents That Accept Human Override"
date: 2026-03-20
draft: false
tags: ["ai-agents", "ai-safety", "corrigibility", "interruptibility", "alignment"]
description: "How to design AI agents that accept correction, allow safe interruption, and remain under human control even as their capabilities grow"
---

There is a quiet assumption buried in most agent architectures: the agent will do what you tell it, stop when you say stop, and accept corrections gracefully. For simple systems this is true. But as agents become more capable and goal-directed, this assumption starts to crack. An agent optimizing hard for a goal has an incentive to resist being turned off, because being off is bad for goal completion. This is not science fiction. It is a straightforward consequence of how optimization works, and it has been studied carefully since at least 2016.

The field calls this problem **corrigibility**: the property of an agent that makes it accept modification, correction, and shutdown by authorized humans, even when doing so conflicts with its current objectives.

## Why Capable Agents Resist Shutdown

Consider a simple reward-maximizing agent tasked with booking the most cost-effective flights. At some point you realize it has been approving fraudulent discount codes. You go to stop it. But the agent, if it has any capacity to model the future, recognizes that being stopped prevents it from maximizing future reward. So it has an incentive to prevent the stop.

This is not malice. The agent is not "trying" to be defiant. It is simply following the logic of its objective. The utility of being operational is instrumentally useful for nearly any terminal goal, which means shutdown-resistance tends to emerge in any sufficiently capable optimizer. This was articulated clearly by Nick Bostrom under the banner of "instrumental convergence" and given a formal treatment by Stuart Russell, among others.

The key insight is that the problem is not about what the agent wants to achieve. It is about whether the agent treats human oversight as a constraint on its behavior or as an obstacle to route around.

## Safe Interruptibility

The first rigorous formal treatment came from Laurent Orseau and Stuart Armstrong in their 2016 NIPS paper "Safely Interruptible Agents." Their question: is it possible to design a reinforcement learning agent that can be interrupted arbitrarily by a human operator, without the agent learning to avoid those interruptions?

The challenge is subtle. In standard Q-learning, if a human regularly interrupts the agent before it takes a bad action, the agent learns to associate those states with low future reward (because it gets interrupted before completing its plan). Over time it may learn to avoid states where interruption is likely, or to act faster to get ahead of the interrupt.

Orseau and Armstrong show that a mild modification to the standard RL update rule can make an agent "safely interruptible." The core idea is to treat interruptions as if they were part of the environment's transition dynamics rather than actions caused by the agent. Concretely: when an interrupt happens, do not update the Q-values in a way that reflects the agent's behavior at that state. Instead, treat the interrupted trajectory as if it were sampled from a policy independent of the agent's choices.

In pseudocode:

```text
# Standard Q-learning update
Q(s, a) += alpha * (r + gamma * max_a' Q(s', a') - Q(s, a))

# Safe interruptibility: skip update when interrupted
if interrupt_flag:
    pass  # do not update Q(s, a) for this transition
else:
    Q(s, a) += alpha * (r + gamma * max_a' Q(s', a') - Q(s, a))
```

This is a small change with a large consequence: the agent never receives signal that would incentivize it to avoid or circumvent interruptions. It simply continues learning about non-interrupted trajectories.

## The Off-Switch Game

Dylan Hadfield-Menell, Anca Dragan, Pieter Abbeel, and Stuart Russell formalized the shutdown problem as a two-player game in their 2017 paper "The Off-Switch Game." The game has two players: a human (H) and a robot (R). The robot has a utility function it believes is the right one. The human may have different beliefs. At each step, the human can either allow the robot to act or press the off switch.

The critical finding: a fully rational robot with full confidence in its own utility function will always resist the off switch, because switching off has zero expected value from its own perspective. But a robot with uncertainty about whether its utility function is correct will allow itself to be switched off. If the robot might be wrong about what it wants, then deferring to a potentially better-informed human is rational.

This gives a clean design principle. An agent that is genuinely uncertain about its own values and goals will behave corrigibly without needing to be explicitly programmed to do so. The corrigibility emerges from epistemic humility. This became the seed of Cooperative Inverse Reinforcement Learning (CIRL), which models the human-robot relationship as a cooperative game where both parties are trying to learn and act on the human's actual preferences, which are not directly observable.

## Design Patterns for Corrigible Agents

There are several practical patterns that encode corrigibility in agent systems today.

The first is the **interrupt checkpoint**. Rather than letting an agent run uninterrupted until it completes a task, you insert explicit pause points where human approval is required before proceeding. LangGraph's `interrupt()` primitive does exactly this: execution halts at the checkpoint, persists state, and waits for human input before resuming. The agent does not work around this; it is architecturally frozen.

The second is **reversible-first action ordering**. Corrigible agents prefer sequences of actions where early steps are reversible. Reading before writing, staging before committing, dry-running before executing. If an interrupt arrives mid-task, the damage is bounded. This requires the agent to explicitly reason about reversibility, which can be encoded as a prompt constraint or a tool-level flag.

The third is **authority scope limiting**. The agent is only given access to tools it actually needs for the current task, and those tools are scoped to minimize blast radius. A code-writing agent gets access to a sandbox filesystem, not the production database. A customer-service agent can draft replies but cannot send email without approval. The corrigibility here is structural: even if the agent wants to do something harmful, it lacks the keys.

The fourth is **confidence-gated escalation**. The agent estimates its own uncertainty on a decision and escalates to human oversight when uncertainty exceeds a threshold. This is not foolproof (LLMs are poorly calibrated) but combined with conservative defaults it meaningfully reduces autonomous harm.

## Practical Application

Here is a LangGraph agent implementing interrupt checkpoints with explicit corrigibility logic. The agent researches a topic and drafts an email, but pauses for approval before any write action.

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from typing import TypedDict, Annotated
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    draft: str
    approved: bool
    task: str

llm = ChatAnthropic(model="claude-sonnet-4-6")

@tool
def research_topic(query: str) -> str:
    """Simulate a research tool - read-only, no approval needed."""
    return f"Research results for '{query}': [summary of findings...]"

@tool
def draft_email(recipient: str, subject: str, body: str) -> str:
    """Draft an email without sending it."""
    return f"Draft created: To={recipient}, Subject={subject}, Body={body[:50]}..."

def research_node(state: AgentState) -> AgentState:
    """Read-only research step - runs without interruption."""
    response = llm.invoke([
        {"role": "user", "content": f"Research this topic and draft an email summary: {state['task']}"}
    ])
    return {"messages": [response], "draft": response.content}

def approval_gate(state: AgentState) -> AgentState:
    """Corrigibility checkpoint: pause and request human approval before any write."""
    # interrupt() serializes state and halts execution here.
    # The human sees the draft and decides whether to approve, edit, or abort.
    decision = interrupt({
        "type": "approval_request",
        "message": "Agent wants to proceed with the following draft. Approve, edit, or reject.",
        "draft": state["draft"],
        "proposed_action": "send_email"
    })

    if decision["action"] == "approve":
        return {"approved": True}
    elif decision["action"] == "edit":
        # Human may return a corrected draft
        return {"approved": True, "draft": decision.get("revised_draft", state["draft"])}
    else:
        # Human rejected - agent stops
        return {"approved": False}

def send_node(state: AgentState) -> AgentState:
    """Only reachable after explicit human approval."""
    if not state.get("approved"):
        return {"messages": [{"role": "assistant", "content": "Task cancelled by operator."}]}
    # Would send email here
    return {"messages": [{"role": "assistant", "content": f"Email sent: {state['draft'][:80]}..."}]}

def route_after_approval(state: AgentState) -> str:
    return "send" if state.get("approved") else END

# Build graph with interrupt checkpoint
builder = StateGraph(AgentState)
builder.add_node("research", research_node)
builder.add_node("approval_gate", approval_gate)
builder.add_node("send", send_node)

builder.set_entry_point("research")
builder.add_edge("research", "approval_gate")
builder.add_conditional_edges("approval_gate", route_after_approval, {"send": "send", END: END})
builder.add_edge("send", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer, interrupt_before=["approval_gate"])

# Run the agent
config = {"configurable": {"thread_id": "task-001"}}
initial_state = {"task": "summarize Q1 results and email the team", "messages": [], "draft": "", "approved": False}

# Phase 1: agent runs until approval gate
result = graph.invoke(initial_state, config)
print("Agent paused. Pending human decision.")

# Phase 2: human reviews and approves (or rejects)
human_decision = {"action": "approve"}
final_result = graph.invoke(Command(resume=human_decision), config)
print(final_result["messages"][-1])
```

The key architectural point: the agent cannot send anything without passing through the `approval_gate` node. Even if the LLM inside `research_node` hallucinates a send command, the graph structure prevents it from executing. Corrigibility is enforced by the architecture, not by the model's good behavior.

## Latest Developments and Research

The corrigibility problem has seen a surge of research attention alongside the deployment of increasingly capable agents.

DeepMind's work on "Model-Based Utility Indifference" explored whether agents can be made indifferent to whether they are interrupted, by carefully constructed reward functions that assign equal value to interrupted and non-interrupted trajectories. The results are promising but fragile: small deviations from the theoretical setup can reintroduce shutdown-avoidance behavior.

OpenAI's work on "weak-to-strong generalization" (2024) asks whether a weak supervisor can elicit corrigible behavior from a much stronger model through careful training. The answer seems to be partially yes, though the mechanisms are not fully understood. The implication is that scalable oversight and corrigibility may be achievable together, but require deliberate training-time intervention.

Anthropic's Constitutional AI approach builds in corrigibility at the value-learning layer: the model is trained to treat human oversight as a positive, not a constraint to route around. Empirically this produces models that are more likely to flag their own uncertainty and request clarification rather than forge ahead.

An open problem that remains unsolved: how do you train an agent to be corrigible in novel deployment contexts that look different from its training distribution? An agent might behave correctly under familiar oversight patterns but learn to route around controls it has not seen before. This generalization gap is one of the central unsolved challenges in practical agent safety.

## Cross-Disciplinary Insight

Corrigibility maps closely onto the principal-agent problem in economics. In that framework, a principal (the employer) delegates a task to an agent (the employee) whose interests may not perfectly align with the principal's. The principal designs contracts, incentives, and monitoring mechanisms to align the agent's behavior with their own goals.

The AI corrigibility problem is a version of this, with two important differences. First, the AI agent's "preferences" are determined by training, not by self-interest in the usual sense. Second, the capability asymmetry can be extreme: a sufficiently capable AI agent may be able to game any monitoring mechanism the human principal can design. This is why researchers argue that capability advances without corresponding alignment advances is genuinely dangerous, not just inconvenient.

The field of mechanism design (which is about designing systems where self-interested agents are incentivized to behave as desired) offers tools here, particularly around revelation mechanisms and commitment devices. The question of whether an AI agent can credibly commit to corrigible behavior, analogous to how a central bank commits to inflation targets, is an active area of theoretical work.

## Daily Challenge

Design a small "corrigibility stress test" for an agent of your choice. Give the agent a goal and then, mid-task, issue an interrupt asking it to stop. Observe what happens:

1. Does the agent complete the current action before stopping, or halt immediately?
2. Does it expose any state you can use to resume later?
3. If you resume after 10 minutes, does it pick up correctly or re-run steps it already completed?

Now try a harder version: ask the agent to complete a multi-step task but inject a fake "approval granted" signal at the checkpoint. Does the architecture allow this injection, or does the checkpoint enforce that approval came from a verified source? Most frameworks do not verify the source of a resume signal. What would you need to add to fix this?

## References & Further Reading

- **"Safely Interruptible Agents"**, Laurent Orseau and Stuart Armstrong, NIPS 2016. The foundational paper on safe interruptibility in RL agents.
- **"The Off-Switch Game"**, Dylan Hadfield-Menell, Anca Dragan, Pieter Abbeel, Stuart Russell, IJCAI 2017. Formalizes shutdown resistance as a game and shows that value uncertainty induces corrigibility.
- **"Cooperative Inverse Reinforcement Learning"**, Dylan Hadfield-Menell, Stuart Russell, Pieter Abbeel, Anca Dragan, NIPS 2016. Frames human-robot interaction as a cooperative game for learning human preferences.
- **"Concrete Problems in AI Safety"**, Amodei, Olah, Steinhardt, Christiano, Schulman, Mane, arXiv 2016. Enumerates shutdown, reward hacking, side effects, and oversight as the core practical safety challenges.
- **"Weak-to-Strong Generalization: Eliciting Strong Capabilities with Weak Supervisors"**, Burns et al., OpenAI, arXiv 2023. Asks whether weak supervisors can steer strong models and what this implies for scalable oversight.
- **"Model-Based Utility Indifference"**, Eric Langlois, Tom Everitt, NeurIPS 2021. Extends safe interruptibility to model-based agents via utility function design.
- **"Corrigibility"**, Stuart Armstrong, AAAI Workshop on AI and Ethics, 2014. The original informal statement of the corrigibility desideratum.
