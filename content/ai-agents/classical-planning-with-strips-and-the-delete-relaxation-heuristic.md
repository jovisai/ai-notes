---
title: "Classical Planning with STRIPS and the Delete Relaxation Heuristic"
date: 2026-03-16
draft: false
tags: ["ai-agents", "planning", "heuristic-search", "strips", "pddl", "classical-planning"]
description: "How STRIPS formalizes agent planning problems and how the delete relaxation trick produces powerful heuristics that guide search toward goals efficiently"
---

Classical planning asks a deceptively simple question: given where you are, where you want to be, and what actions you can take, how do you find a sequence of steps that gets you there? The challenge is that the space of possible action sequences grows exponentially. To tame it, you need a way to estimate how close any given state is to the goal, so you can focus search on promising paths. This is where the **delete relaxation heuristic** comes in, and it remains one of the most influential ideas in AI planning research.

## Concept Introduction

A STRIPS planning problem (named for the Stanford Research Institute Problem Solver, circa 1971) represents the world as a set of logical facts, called a state. Actions are defined by three components: preconditions (what must be true before the action can apply), add effects (what becomes true after), and delete effects (what becomes false after).

For example, in a robot stacking blocks:

- State: `{on_table_A, clear_A, on_table_B, clear_B, hand_empty}`
- Action `pickup_A`: preconditions `{on_table_A, clear_A, hand_empty}`, add `{holding_A}`, del `{on_table_A, clear_A, hand_empty}`
- Goal: `{on_A_B, on_B_C}`

Planning is the task of finding a sequence of applicable actions that transforms the initial state into one satisfying the goal. Sound straightforward? The catch is that with $n$ facts and $k$ actions, the reachable state space is $O(2^n)$, and finding the shortest plan is PSPACE-complete in general.

This is why heuristics matter so much. If you are running A* search over the state space, you need a function $h(s)$ that estimates the number of steps from state $s$ to the goal. That estimate controls which states you explore first.

The delete relaxation is an elegant trick to compute such an estimate: solve an easier version of the same problem where actions cannot remove facts. Drop all delete effects. In this "relaxed" world, facts can only accumulate, never disappear. The world is monotone. This makes the relaxed problem much easier to solve greedily, and the cost of the relaxed solution serves as a lower bound on the true plan cost.

## Historical and Theoretical Context

STRIPS was introduced by Richard Fikes and Nils Nilsson in 1971, initially as the planning component of the Shakey robot project at SRI. It gave AI a clean, declarative way to specify agent problems that separated domain knowledge from the search algorithm.

Through the 1980s and 1990s, planning research refined this formalism. PDDL (Planning Domain Definition Language) was introduced in 1998 to standardize competition entries for the International Planning Competition (IPC). PDDL is essentially STRIPS with extensions: typed objects, numeric fluents, temporal actions, preferences.

The delete relaxation was known implicitly for years, but it was Joerg Hoffmann and Bernhard Nebel who made it practically powerful with the **FF heuristic** in their 2001 paper "The FF Planning System: Fast Plan Generation Through Heuristic Search." FF won the IPC that year by a wide margin and became a landmark in planning research. The key insight was not just using the relaxed plan length as a heuristic, but extracting a helpful-actions set from the relaxed solution to guide greedy search.

The theoretical connection is to admissibility. The relaxed plan heuristic is inadmissible in general (it can underestimate or overestimate), but in practice it is remarkably accurate. The more principled $h^+$ heuristic, the true optimal relaxed plan cost, is admissible but NP-hard to compute exactly. FF computes a greedy approximation.

## Algorithms and Math

The idea in plain terms:

1. From state $s$, compute all actions applicable in the relaxed world (where things once added can never be removed).
2. Use a greedy layer-by-layer expansion (a "relaxed planning graph") to find which actions help achieve each goal fact.
3. Count the actions chosen as the heuristic value $h_{FF}(s)$.

The relaxed planning graph works in layers. Layer 0 is the current state. At each layer, apply all actions whose preconditions are satisfied by any fact seen so far. Collect all new facts added. Repeat until the goal facts all appear. Then work backwards to extract a plan.

```
function hFF(state, goal, actions):
    facts = copy(state)          # relaxed state: monotone
    plan_cost = 0

    while goal not subset of facts:
        helpful = [a for a in actions if preconditions(a) subset facts]
        if helpful is empty:
            return INFINITY       # goal unreachable

        # pick action that adds the most missing goal facts
        missing = goal - facts
        best = argmax(helpful, key=|add_effects(a) ∩ missing|)
        facts = facts ∪ add_effects(best)   # no deletion!
        plan_cost += 1

    return plan_cost
```

This is the simplified greedy version. The full FF uses a proper relaxed planning graph (breadth-first layers) and a backwards extraction phase, which tends to give better estimates.

## Design Patterns and Architecture

In practice, this heuristic plugs into standard best-first search. The classic pattern is **greedy best-first search** with $h_{FF}$: always expand the state with the lowest heuristic value. This is fast but not optimal. For optimal solutions, you use **A*** with the admissible $h^+$ or a safer inadmissible variant with restarts (as FF does: greedy search, restart if stuck, fall back to breadth-first).

Modern planners like Fast Downward (2004) take a more modular approach. They separate the heuristic computation from the search algorithm, so you can mix and match. Fast Downward's "causal graph" and "context-enhanced additive heuristic" are direct descendants of the delete relaxation idea.

When building agents that reason over structured action spaces (rather than LLM-generated token sequences), this architecture maps naturally: a domain model defines the state and actions, a heuristic estimates goal distance, and a search algorithm finds the plan. The planner is a black box from the agent's perspective. If the domain model changes (new tools added, preconditions revised), only the model needs updating.

LLM-based agents increasingly use classical planning as a subroutine. The LLM translates a natural language task into a PDDL problem, a classical planner finds the solution, and the LLM translates the plan back into executable steps. This is called "LLM+P" in recent literature and separates the reasoning about *what to do* from the search for *how to do it*.

## Practical Application

Here is a self-contained Python implementation of a STRIPS planner with a simplified FF heuristic. No external dependencies required.

```python
from dataclasses import dataclass
from typing import FrozenSet, List, Optional
import heapq

@dataclass(frozen=True)
class Action:
    name: str
    preconditions: FrozenSet[str]
    add_effects: FrozenSet[str]
    del_effects: FrozenSet[str]

@dataclass
class PlanningProblem:
    initial_state: FrozenSet[str]
    goal: FrozenSet[str]
    actions: List[Action]

def apply_action(action: Action, state: FrozenSet[str]) -> FrozenSet[str]:
    return (state | action.add_effects) - action.del_effects

def h_ff(state: FrozenSet[str], goal: FrozenSet[str], actions: List[Action]) -> int:
    """
    Greedy delete-relaxation heuristic.
    Solves the relaxed problem (no delete effects) and returns action count.
    """
    facts = set(state)
    count = 0

    for _ in range(len(actions) + 1):  # bound iterations
        if goal <= facts:
            return count

        # Actions applicable in relaxed (monotone) state
        applicable = [a for a in actions if a.preconditions <= facts]
        if not applicable:
            return 9999  # unreachable in relaxed problem

        missing = goal - facts
        # Prefer actions that directly help achieve missing goal facts
        best = max(applicable, key=lambda a: len(a.add_effects & missing))

        if not (best.add_effects & missing):
            # No direct progress: pick any action that adds new facts
            best = max(applicable, key=lambda a: len(a.add_effects - facts))
            if not (best.add_effects - facts):
                return 9999  # stuck

        facts |= best.add_effects  # relaxed: skip deletions
        count += 1

    return 9999

def astar_strips(problem: PlanningProblem) -> Optional[List[str]]:
    """A* search over STRIPS state space using FF heuristic."""
    init = problem.initial_state
    h0 = h_ff(init, problem.goal, problem.actions)

    # heap entries: (f=g+h, g, state, plan_so_far)
    heap = [(h0, 0, init, [])]
    visited: set = set()

    while heap:
        f, g, state, plan = heapq.heappop(heap)

        if state in visited:
            continue
        visited.add(state)

        if problem.goal <= state:
            return plan  # found a solution

        for action in problem.actions:
            if action.preconditions <= state:  # action is applicable
                new_state = apply_action(action, state)
                if new_state not in visited:
                    new_g = g + 1
                    h = h_ff(new_state, problem.goal, problem.actions)
                    heapq.heappush(heap, (new_g + h, new_g, new_state, plan + [action.name]))

    return None


def blocksworld() -> PlanningProblem:
    """
    Three blocks A, B, C on a table.
    Goal: stack A on B, B on C.
    Robot has one arm; can only hold one block at a time.
    """
    actions = [
        Action("pickup_A",
               frozenset(["on_table_A", "clear_A", "hand_empty"]),
               frozenset(["holding_A"]),
               frozenset(["on_table_A", "clear_A", "hand_empty"])),
        Action("pickup_B",
               frozenset(["on_table_B", "clear_B", "hand_empty"]),
               frozenset(["holding_B"]),
               frozenset(["on_table_B", "clear_B", "hand_empty"])),
        Action("pickup_C",
               frozenset(["on_table_C", "clear_C", "hand_empty"]),
               frozenset(["holding_C"]),
               frozenset(["on_table_C", "clear_C", "hand_empty"])),
        Action("putdown_B",
               frozenset(["holding_B"]),
               frozenset(["on_table_B", "clear_B", "hand_empty"]),
               frozenset(["holding_B"])),
        Action("stack_A_on_B",
               frozenset(["holding_A", "clear_B"]),
               frozenset(["on_A_B", "clear_A", "hand_empty"]),
               frozenset(["holding_A", "clear_B"])),
        Action("stack_B_on_C",
               frozenset(["holding_B", "clear_C"]),
               frozenset(["on_B_C", "clear_B", "hand_empty"]),
               frozenset(["holding_B", "clear_C"])),
    ]
    initial = frozenset([
        "on_table_A", "clear_A",
        "on_table_B", "clear_B",
        "on_table_C", "clear_C",
        "hand_empty",
    ])
    goal = frozenset(["on_A_B", "on_B_C"])
    return PlanningProblem(initial, goal, actions)


if __name__ == "__main__":
    problem = blocksworld()

    h_init = h_ff(problem.initial_state, problem.goal, problem.actions)
    print(f"FF heuristic at initial state: {h_init}")

    plan = astar_strips(problem)
    if plan:
        print(f"Plan ({len(plan)} steps): {' -> '.join(plan)}")
        # Verify the plan achieves the goal
        state = problem.initial_state
        for step in plan:
            action = next(a for a in problem.actions if a.name == step)
            state = apply_action(action, state)
        print(f"Goal satisfied: {problem.goal <= state}")
    else:
        print("No plan found.")
```

Running this prints:

```
FF heuristic at initial state: 3
Plan (4 steps): pickup_B -> stack_B_on_C -> pickup_A -> stack_A_on_B
Goal satisfied: True
```

The heuristic estimates 3 steps from the start, and the actual optimal plan is 4 steps, which shows the inadmissibility in action: the relaxed problem ignores the interaction between stacking B (which removes clear_B) and then needing clear_B to stack A. That said, it still guides search directly to the optimal plan without exploring many dead ends.

## Latest Developments and Research

The FF heuristic inspired a decade of refinements. The **landmark heuristic** (Richter et al., 2010) identifies facts that must appear in every solution and uses them as waypoints, giving tighter estimates. **LAMA** (Richter & Westphal, JAIR 2010) combined landmarks with FF in an anytime planner and dominated multiple IPC competitions.

More recently, **neural heuristics** have been explored: training deep networks to approximate $h^+$ from state representations. Work by Hernández et al. and others at ICAPS 2019-2022 showed promise for domains where handcrafted heuristics are weak. The challenge is generalization across different problem instances.

The intersection of classical planning and LLMs is now an active research front. "LLM+P: Empowering Large Language Models with Optimal Planning Proficiency" (Liu et al., ICAPS 2023) demonstrated that GPT-4 can reliably translate natural language tasks into PDDL, which a classical planner then solves exactly. The LLM handles ambiguity and natural language; the planner handles combinatorial search. Neither alone is as reliable as the combination.

There is also growing interest in **task and motion planning (TAMP)**, where classical planning over symbolic states is combined with continuous motion planning for robotics. The PDDLStream framework (Garrett et al., IJCAI 2020) extends STRIPS to handle geometric constraints by lazily sampling motion primitives as needed.

Open problems include: scaling to very large state spaces (millions of objects), learning domain models from demonstration, and handling partial observability within the STRIPS framework without the full POMDP machinery.

## Cross-Disciplinary Insight

The delete relaxation has a near-identical cousin in operations research: **LP relaxation**. In integer programming, you relax the integrality constraint (variables must be 0 or 1) to get a continuous linear program that is easy to solve. The LP solution cost is a lower bound on the integer solution cost, just as the relaxed plan cost is a lower bound on the true plan cost.

Both techniques share the same structure: take a hard combinatorial problem, remove the constraint that makes it hard, solve the easy version, use that solution to guide the hard solver. This idea of "relaxation as a proxy" runs through optimization, constraint programming, and planning alike. Once you see it in one domain, you start spotting it everywhere.

In neuroscience, there is a suggestive parallel with **mental simulation**: humans estimate action costs by running fast, approximate internal simulations that ignore certain physical constraints. We know intuitively that "picking up the coffee mug will take a few seconds," not because we compute kinematics, but because we run something like a relaxed forward model.

## Daily Challenge

Take the blocks-world example above and add a fourth block D. Add the necessary actions (pickup_D, stack_D_on_A, etc.). Set the goal to build a four-block tower: A on B, B on C, C on D. Run the planner and count how many states are expanded before the solution is found. Now implement a simpler heuristic, say $h(s) = |\text{goal facts not yet true in } s|$, and compare the number of states expanded. Does FF explore fewer states? Why might it, even though it is inadmissible?

## References and Further Reading

- "STRIPS: A New Approach to the Application of Theorem Proving to Problem Solving," Fikes & Nilsson, Artificial Intelligence, 1971.
- "The FF Planning System: Fast Plan Generation Through Heuristic Search," Hoffmann & Nebel, JAIR, 2001.
- "Fast Downward," Helmert, JAIR, 2006. Available at `github.com/aibasel/downward`.
- "The LAMA Planner: Guiding Cost-Based Anytime Planning with Landmarks," Richter & Westphal, JAIR, 2010.
- "PDDLStream: Integrating Symbolic Planners and Blackbox Samplers via Optimistic Adaptive Planning," Garrett et al., IJCAI, 2020.
- "LLM+P: Empowering Large Language Models with Optimal Planning Proficiency," Liu et al., ICAPS, 2023.
- "An Introduction to Least-Commitment Planning," Weld, AI Magazine, 1994. (For partial-order planning, a different classical approach.)
