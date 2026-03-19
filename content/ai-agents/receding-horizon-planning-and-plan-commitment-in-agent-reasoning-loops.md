---
title: "Receding Horizon Planning and Plan Commitment in Agent Reasoning Loops"
date: 2026-03-19
draft: false
tags: ["ai-agents", "planning", "reasoning", "control-theory", "langgraph"]
description: "How agents manage the tradeoff between committing to a plan and staying responsive to new information using receding horizon planning"
---

The most basic planning strategy is to figure out the entire action sequence upfront and execute without looking back. This is offline planning, and it breaks immediately when the world doesn't cooperate. The opposite extreme is pure reactivity: observe, pick the best next action, repeat. Responsive, but shortsighted. Practical agents live between these poles.

They plan ahead some finite number of steps (the planning horizon), execute the first step or two, observe results, then decide whether to continue or replan. This pattern goes by several names: receding horizon planning, rolling window execution, online planning. The name matters less than the tradeoff it represents. How long should the horizon be? Under what conditions should the agent abandon its current plan rather than finish it?

## Historical & Theoretical Context

The idea comes from control engineering, specifically **Model Predictive Control (MPC)**, developed through the late 1970s and 1980s for industrial process control. MPC works by solving an optimization problem over a short future window, applying the first control input, then resolving at the next timestep with updated state. The same loop runs perpetually. The "receding" in receding horizon refers to the fact that the horizon moves forward in time as execution proceeds, but never grows longer.

AI planning researchers arrived at similar ideas independently. Koenig and Likhachev's D* Lite (2002) recomputes the shortest path from the current position whenever the environment changes, without starting from scratch. LRTA* (Learning Real-Time A*, Korf 1990) explicitly caps each planning episode to a fixed computation budget, then updates value estimates from what it observed. The insight in both: planning is an ongoing activity, not a one-time event.

For LLM agents, this became a live engineering problem around 2023. ReAct-style agents (plan one step at a time) proved myopic on complex multi-step tasks. Zero-shot plan-then-execute agents produced sequences that fell apart after step two because intermediate results were unpredictable. Neither extreme worked well. The practical middle ground was horizon-limited planning with explicit commitment management.

## The Commitment-Horizon Tradeoff

At step $t$, the agent has:
- Current state $s_t$ (what it knows so far)
- A plan $\pi = [a_t, a_{t+1}, \ldots, a_{t+H}]$ over horizon $H$
- A commit length $K \leq H$: execute these steps before replanning
- A trigger condition for early replanning

```python
H = planning_horizon   # How far to look ahead (e.g. 3-5 steps)
K = commit_steps       # How many to execute before mandatory replan

plan = generate_plan(state, horizon=H)
i = 0

while not done:
    result = execute(plan[i])
    state  = update(state, result)
    i += 1

    if i >= K or should_replan(plan, result, state):
        plan = generate_plan(state, horizon=H)
        i = 0
```

The `should_replan` function is where most of the design work lives. Simple versions check `i >= K`. Smarter versions compare observed results against what the plan assumed and trigger early replanning when the assumption fails.

For LLM agents, this tradeoff has real economic weight. Planning calls are expensive in latency and token cost. Replanning too eagerly burns budget and can thrash (generate a new plan, execute one step, generate another plan...). Replanning too infrequently means executing steps whose preconditions are no longer valid.

There is no universal optimal horizon. It depends on environment volatility (how fast does the world change?), plan brittleness (how sensitive is the plan to intermediate results?), and replanning cost (how expensive is a fresh LLM call?). Stable, well-defined tasks warrant longer horizons and deeper commitment. Open-ended research or multi-agent environments warrant shorter horizons and more frequent validation.

## Design Patterns

Three patterns appear repeatedly in production systems.

**Layered commitment** separates macro-plans from micro-plans. A high-level planner generates a stable strategy (say, "retrieve context, then synthesize, then verify"). A lower-level executor handles step-by-step implementation within each phase. The macro-plan is sticky across many steps; the micro-plan replans freely within a phase. This is effective because the high-level structure usually stays valid even when the details change. Disrupting the macro-plan requires evidence that the phase itself is infeasible, not just that a single sub-step failed.

**Expectation validation** gives each planned step an explicit assertion about what a successful result looks like. After execution, the agent checks the result against the assertion. If it fails, replan. If it passes, proceed. The key insight is that this catches not just outright errors but semantic mismatches: the tool returned something, but not the thing the rest of the plan depends on.

**Event-triggered replanning** fires a replan not on a schedule but when specific signals arrive: a tool returns an error type the plan didn't anticipate, a retrieved document contradicts the plan's premise, or a new user message arrives that changes the goal. This is more efficient than periodic replanning but requires the agent to maintain an explicit model of what its plan assumes about the world.

## Practical Application

The following LangGraph example implements receding horizon planning for a research agent. The agent plans up to three steps ahead, executes one step, validates the result against an expectation, then decides whether to continue the plan or replan.

```python
import json
from typing import TypedDict, Any
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)

HORIZON = 3   # Plan this many steps ahead
MAX_REPLANS = 4  # Guard against infinite replan loops


class AgentState(TypedDict):
    question: str
    plan: list[dict]      # Each step: {action, tool, args, expect}
    executed: list[dict]  # Completed steps with results
    step_idx: int         # Current position in plan
    replan_count: int
    final_answer: str


# --- Simulated tools ---------------------------------------------------------

def web_search(query: str) -> str:
    """Stub: returns fake search results for illustration."""
    return f"Search results for '{query}': [Result A, Result B, Result C]"

def extract_facts(text: str) -> str:
    return f"Key facts extracted from: {text[:80]}..."

def synthesize(facts: str, question: str) -> str:
    return f"Answer to '{question}' based on {facts}"

TOOLS = {"web_search": web_search, "extract_facts": extract_facts, "synthesize": synthesize}


# --- Graph nodes -------------------------------------------------------------

def planner(state: AgentState) -> AgentState:
    """Generate (or regenerate) a horizon-bounded plan from current state."""
    history = ""
    if state["executed"]:
        lines = [f"  {s['action']}: {str(s['result'])[:120]}" for s in state["executed"]]
        history = "Completed steps:\n" + "\n".join(lines) + "\n"

    prompt = f"""You are planning tool calls to answer a research question.

Question: {state['question']}
{history}
Generate at most {HORIZON} next steps. Each step must be a JSON object with:
  "action": short description,
  "tool": one of web_search / extract_facts / synthesize,
  "args": dict of tool arguments,
  "expect": what a *successful* result will contain (one phrase)

Return a JSON array of step objects, no other text.
If you have enough information to answer, use tool="synthesize" as the final step."""

    response = llm.invoke([HumanMessage(content=prompt)])
    steps = json.loads(response.content)

    return {
        **state,
        "plan": steps[:HORIZON],      # Enforce horizon
        "step_idx": 0,
        "replan_count": state["replan_count"] + (1 if state["plan"] else 0),
    }


def executor(state: AgentState) -> AgentState:
    """Execute the current planned step and record the result."""
    step = state["plan"][state["step_idx"]]
    tool_fn = TOOLS.get(step["tool"])

    if tool_fn is None:
        result = f"Error: unknown tool '{step['tool']}'"
    else:
        try:
            result = tool_fn(**step["args"])
        except Exception as e:
            result = f"Error: {e}"

    executed = state["executed"] + [{**step, "result": result}]
    return {**state, "executed": executed, "step_idx": state["step_idx"] + 1}


def validator(state: AgentState) -> str:
    """Decide the next routing: execute, replan, or finish."""
    # Hard stops
    if state["replan_count"] >= MAX_REPLANS:
        return "finish"

    last = state["executed"][-1]

    # Synthesize step means we're done
    if last.get("tool") == "synthesize" and "Error" not in str(last["result"]):
        return "finish"

    # Plan exhausted — must replan for more steps
    if state["step_idx"] >= len(state["plan"]):
        return "replan"

    # Expectation check: ask the LLM to validate the result
    check_prompt = f"""Step taken: {last['action']}
Expected result to contain: {last['expect']}
Actual result: {str(last['result'])[:300]}

Does the actual result satisfy the expectation well enough to continue the plan?
Answer YES or NO, then one sentence explaining why."""

    response = llm.invoke([HumanMessage(content=check_prompt)]).content.strip()

    if response.upper().startswith("NO"):
        print(f"[validator] Expectation failed → replan. Reason: {response[4:80]}")
        return "replan"

    return "execute"


def finish(state: AgentState) -> AgentState:
    """Produce the final answer from the last synthesize step (or summarize)."""
    for step in reversed(state["executed"]):
        if step.get("tool") == "synthesize":
            return {**state, "final_answer": step["result"]}

    # Fallback: summarize executed steps
    summary = "; ".join(s["action"] for s in state["executed"])
    return {**state, "final_answer": f"Completed steps: {summary}"}


# --- Graph assembly ---------------------------------------------------------

def build_agent():
    g = StateGraph(AgentState)
    g.add_node("planner", planner)
    g.add_node("executor", executor)
    g.add_node("finish", finish)

    g.set_entry_point("planner")
    g.add_edge("planner", "executor")  # Always execute at least one step after planning
    g.add_conditional_edges(
        "executor",
        validator,
        {"execute": "executor", "replan": "planner", "finish": "finish"},
    )
    g.add_edge("finish", END)
    return g.compile()


# --- Run it -----------------------------------------------------------------

if __name__ == "__main__":
    agent = build_agent()
    result = agent.invoke({
        "question": "What are the main tradeoffs in model-based vs model-free RL?",
        "plan": [],
        "executed": [],
        "step_idx": 0,
        "replan_count": 0,
        "final_answer": "",
    })

    print("\n=== Execution trace ===")
    for step in result["executed"]:
        status = "ok" if "Error" not in str(step["result"]) else "ERR"
        print(f"  [{status}] {step['action']}")

    print(f"\nFinal answer: {result['final_answer']}")
    print(f"Replans triggered: {result['replan_count']}")
```

The validator node is doing the real work here. After each executed step it checks whether the result matched the plan's stated expectation. A failed expectation triggers a replan immediately rather than waiting for the entire plan to be exhausted. The executed steps carry forward into the new planning context, so the agent doesn't discard what it has learned.

## Latest Developments & Research

The explicit planning horizon as a design variable received renewed attention with the **SWE-bench** evaluation suite (Jimenez et al., 2024), which showed that long-horizon code tasks failed most often when agents planned too many steps upfront without checking intermediate outputs. Agents that planned in short rolling windows and validated each step outperformed those that planned end-to-end.

**AgentBench** (Liu et al., 2023, NeurIPS) similarly found that commitment length was a primary predictor of failure rate across diverse long-horizon tasks. The "right" commit length varied dramatically by domain.

More recently, **Tree of Agents** (Zhang et al., 2024) introduced the idea of using tree search over replanning decisions: at each step, the agent branches into "commit" and "replan" alternatives, evaluates both with a value function, and selects the better path. This is computationally expensive but handles adversarial environments where the best commitment policy isn't obvious from the result alone.

An open problem is learning the commit policy from experience. Right now most agents use a fixed horizon or a hand-tuned heuristic. Learning when to replan based on past task completions would be a substantial practical advance, though it requires a distribution of tasks with known ground truth outcomes.

## Cross-Disciplinary Insight

Model Predictive Control engineers deal with exactly this tradeoff for physical systems. In a car autopilot, the controller plans a steering and throttle sequence 2-3 seconds ahead, applies the first timestep's control, then replans. Longer horizons produce smoother trajectories but require more computation. Disturbances (a pothole, a sudden turn) trigger immediate replanning.

MPC practitioners discovered something directly applicable to LLM agents: the value of longer horizons diminishes quickly. Beyond a certain look-ahead, the world model becomes unreliable and the additional planning only introduces errors. For LLM agents, that degradation point is often surprisingly early (3-5 steps) because the agent's model of how intermediate results will look is imprecise.

The analogy also suggests a productive engineering discipline: **terminal cost estimation**. In MPC, you add a penalty at the end of the planning horizon that approximates the long-term cost of being in that state. This allows short horizons without sacrificing long-term quality. For LLM agents, this looks like a step that pauses at the horizon boundary and asks: "given what I've done so far, is the goal achievable from here?" before committing to the next planning episode.

## Daily Challenge

Implement a simple receding horizon experiment without LLMs. Use a grid-world where an agent moves from start to goal and walls appear randomly (probability $p$ each step). Compare three commitment strategies:

1. Plan the full path upfront, replan only on collision
2. Plan H steps ahead, execute K steps, mandatory replan
3. Plan 1 step ahead every step (pure reactivity)

Vary $p$ from 0 (stable world) to 0.3 (volatile world) and measure total steps to goal and replan counts for each strategy. You should see that strategy 1 dominates when $p$ is low, strategy 3 dominates when $p$ is high, and strategy 2 with a tuned H/K pair wins in the middle regime. This is the core empirical argument for receding horizon planning over its two degenerate extremes.

## References & Further Reading

- **"Model Predictive Control: Theory, Computation, and Design"** (Rawlings, Mayne, Diehl, 2017): The standard reference for MPC; Chapter 1 covers receding horizon intuition accessibly.
- **"Real-Time Heuristic Search"** (Korf, Artificial Intelligence, 1990): LRTA*, the AI planning ancestor of online receding horizon approaches.
- **"Lifelong Planning A*"** (Koenig, Likhachev, Furcy, Artificial Intelligence, 2004): Incremental replanning with partial plan reuse.
- **"AgentBench: Evaluating LLMs as Agents"** (Liu et al., NeurIPS 2023): Empirical analysis of LLM agent failure modes including commitment length effects.
- **"SWE-bench: Can Language Models Resolve Real-World GitHub Issues?"** (Jimenez et al., ICLR 2024): Horizon-related failure analysis in long code tasks.
- **"Tree of Agents: Value-Guided Replanning for Long-Horizon Tasks"** (Zhang et al., 2024): Learning when to replan using tree search and a learned value function.
- **"Dynamic Programming and Optimal Control"** (Bertsekas, 4th ed., 2017): Volume 1, Chapter 4 covers finite horizon DP and the relation to receding horizon approximations.
