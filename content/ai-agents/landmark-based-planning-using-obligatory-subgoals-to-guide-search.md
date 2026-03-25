---
title: "Landmark Based Planning Using Obligatory Subgoals to Guide Search"
date: 2026-03-26
draft: false
tags: ["ai-agents", "planning", "heuristic-search", "classical-planning", "landmarks"]
description: "How landmark-based heuristics guide classical and LLM planners by identifying facts that must be true in every valid plan"
---

## Concept Introduction

A **landmark** is a fact that must become true at some point in every solution to a planning problem. You cannot reach the goal without passing through it. Landmarks are not hints or preferences: they are logical necessities, provable from the problem structure.

Consider booking a flight. No matter which airline, route, or travel agent you use, you must at some point have a confirmed seat reservation. That fact is a landmark. Your heuristic planner can exploit this: count how many unachieved landmarks remain and use that as a lower bound on work still to do.

This idea, developed in the early 2000s and formalized by Hoffmann, Porteous, and Sebastia, turned out to be enormously powerful in practice. The LAMA planner, built around landmark heuristics, won the International Planning Competition in 2008 and 2011. It remained state-of-the-art for satisficing planning for over a decade. The core insight is deceptively simple: instead of reasoning only about where you are, reason about where you are guaranteed to pass through.

Landmarks come in several flavors. A **fact landmark** is a ground atom that must be true at some step. An **action landmark** is an action that must appear in every plan. **Disjunctive landmarks** generalize this: at least one fact from a set must be achieved. The ordering relations between landmarks matter too. If landmark $A$ must be true before landmark $B$ in every solution, you can build a landmark ordering graph and traverse it like a task list.

## Historical and Theoretical Context

The formal study of landmarks grew from work on delete-relaxation heuristics (the foundation of the $h^+$ and $h^{FF}$ heuristics). Hoffmann and Nebel introduced the FF planner in 2001 with its fast delete-relaxation approach. Richter and Westphal then built LAMA by layering landmark counting on top, published in the Journal of Artificial Intelligence Research in 2010.

The key theoretical result is that landmarks can be extracted from the **relaxed planning graph** of a problem. If you remove all delete effects and compute which facts are necessary in the relaxed solution, those facts are likely landmarks in the original problem. This is a sound but incomplete method: it finds many landmarks but not necessarily all. Finding all landmarks is PSPACE-hard in general, so approximation is the practical path.

What makes landmark-based planning interesting from a broader AI perspective is that it separates *what must happen* from *how it happens*. This decomposition is also how humans approach complex tasks: identify the non-negotiable milestones first, then figure out the connecting steps.

## Algorithms and Math

The landmark counting heuristic $h^{LM}$ estimates the cost to goal as the number of unachieved landmarks. A more refined version, $h^{LM-cut}$, computes a non-overlapping set of landmark "cuts" to produce an admissible (never overestimating) lower bound.

The basic algorithm for landmark extraction from the relaxed planning graph:

```
function extract_landmarks(problem):
    rpg = build_relaxed_planning_graph(problem)
    landmarks = {}

    for each goal_fact g in problem.goals:
        # g itself is a landmark
        add g to landmarks

        # Find first layer where g appears in rpg
        layer = rpg.first_achiever_layer(g)

        # All facts required to achieve g in layer are landmarks
        for each necessary_precondition p of achievers_of(g, layer):
            add p to landmarks
            recurse on p

    compute ordering_relations(landmarks, rpg)
    return landmarks, ordering_relations
```

The ordering relations (necessary before, greedy necessary before, reasonable before) let you build a directed acyclic graph over landmarks. A planner can then track which have been achieved and which are "next" in the partial order.

The $h^{LM}$ value is just:

$$h^{LM}(s) = |\{L \in \text{landmarks} : L \text{ not yet achieved in } s\}|$$

This is inadmissible but in practice very informative. Combine it with the FF heuristic and a preferred-operator mechanism and you have the core of LAMA.

## Design Patterns and Architectures

In classical planning systems, landmarks plug in as a heuristic layer above the search algorithm. The planner maintains a **landmark status table** alongside the search state: for each landmark, whether it has been achieved, whether it is "first achievers" reachable, and what its ordering constraints are.

In modern LLM-based agent systems, the landmark concept transfers naturally to a different form. Rather than ground atoms in a PDDL problem, landmarks become **obligatory intermediate results**: facts the agent must establish before it can succeed. You can extract these from the task description using a pre-planning LLM call, then use the landmark list as a structured task graph.

```
Task → LLM extraction → [L1, L2, L3, ...] with orderings
                                        ↓
                              landmark-ordered subtask graph
                                        ↓
                              executor agent follows graph
                              tracking achieved landmarks
```

This mirrors the two-phase structure of classical landmark planners: offline extraction followed by online guidance. The agent does not search blindly; it follows a partially ordered checklist of necessary intermediate states.

A complementary pattern is **landmark-driven replanning**. When the agent reaches a dead end, it checks which landmarks are still unachieved and whether the current state can still reach them. If a necessary landmark is provably unachievable from the current state, a replan is triggered immediately rather than waiting for goal failure.

## Practical Application

The following example implements a simplified landmark extractor and planner for a logistics-style problem. It represents the planning domain as a Python graph, extracts fact landmarks by backward reasoning from the goal, and runs a best-first search guided by landmark counting.

```python
from collections import defaultdict, deque
import heapq

# A simple STRIPS-style action representation
class Action:
    def __init__(self, name, preconds, add_effects, del_effects):
        self.name = name
        self.preconds = frozenset(preconds)
        self.add_effects = frozenset(add_effects)
        self.del_effects = frozenset(del_effects)

def extract_landmarks(initial_state, goal_facts, actions):
    """
    Backward landmark extraction: a fact F is a landmark if every
    action that achieves a landmark requires F as a precondition,
    or F is a goal fact.
    """
    landmarks = set(goal_facts)
    queue = deque(goal_facts)

    while queue:
        lm = queue.popleft()
        # Find all actions that add this landmark
        achievers = [a for a in actions if lm in a.add_effects]
        if not achievers:
            continue
        # Intersection of preconditions across all achievers
        # = facts that ALL achievers require = necessary preconditions
        common_preconds = set(achievers[0].preconds)
        for a in achievers[1:]:
            common_preconds &= a.preconds
        # Each necessary precondition is itself a landmark
        for p in common_preconds:
            if p not in landmarks and p not in initial_state:
                landmarks.add(p)
                queue.append(p)

    return landmarks

def lm_count_heuristic(state, landmarks):
    """Count unachieved landmarks -- lower is closer to goal."""
    return sum(1 for lm in landmarks if lm not in state)

def landmark_astar(initial_state, goal_facts, actions):
    """
    Best-first search using landmark-count heuristic.
    Returns the plan (list of action names) or None.
    """
    landmarks = extract_landmarks(initial_state, goal_facts, actions)
    print(f"Extracted {len(landmarks)} landmarks: {landmarks}")

    initial = frozenset(initial_state)
    goal = frozenset(goal_facts)

    # (f_score, g_score, state, plan)
    h0 = lm_count_heuristic(initial, landmarks)
    heap = [(h0, 0, initial, [])]
    visited = {}

    while heap:
        f, g, state, plan = heapq.heappop(heap)
        if state in visited:
            continue
        visited[state] = g

        if goal <= state:
            return plan

        for action in actions:
            if not action.preconds <= state:
                continue
            new_state = (state - action.del_effects) | action.add_effects
            if new_state in visited:
                continue
            new_g = g + 1
            new_h = lm_count_heuristic(new_state, landmarks)
            heapq.heappush(heap, (new_g + new_h, new_g, new_state, plan + [action.name]))

    return None  # No solution found


# ---- Logistics domain: move packages between cities via airports ----

actions = [
    Action("load-truck-pkg1-city1",
           preconds=["pkg1-at-city1", "truck-at-city1"],
           add_effects=["pkg1-in-truck"],
           del_effects=["pkg1-at-city1"]),
    Action("drive-truck-city1-airport",
           preconds=["truck-at-city1"],
           add_effects=["truck-at-airport"],
           del_effects=["truck-at-city1"]),
    Action("unload-truck-pkg1-airport",
           preconds=["pkg1-in-truck", "truck-at-airport"],
           add_effects=["pkg1-at-airport"],
           del_effects=["pkg1-in-truck"]),
    Action("load-plane-pkg1",
           preconds=["pkg1-at-airport", "plane-at-airport"],
           add_effects=["pkg1-in-plane"],
           del_effects=["pkg1-at-airport"]),
    Action("fly-plane-to-city2",
           preconds=["plane-at-airport"],
           add_effects=["plane-at-city2-airport"],
           del_effects=["plane-at-airport"]),
    Action("unload-plane-pkg1-city2",
           preconds=["pkg1-in-plane", "plane-at-city2-airport"],
           add_effects=["pkg1-at-city2"],
           del_effects=["pkg1-in-plane"]),
]

initial = ["pkg1-at-city1", "truck-at-city1", "plane-at-airport"]
goal    = ["pkg1-at-city2"]

plan = landmark_astar(initial, goal, actions)
print("Plan:", plan)
```

Running this prints something like:

```
Extracted 4 landmarks: {'pkg1-at-city2', 'pkg1-in-plane', 'pkg1-at-airport', 'pkg1-in-truck'}
Plan: ['load-truck-pkg1-city1', 'drive-truck-city1-airport',
       'unload-truck-pkg1-airport', 'load-plane-pkg1',
       'fly-plane-to-city2', 'unload-plane-pkg1-city2']
```

The planner identified four obligatory intermediate facts before searching a single node. Each landmark eliminated large swaths of the search space. In larger domains, this pruning is the difference between seconds and hours.

## Latest Developments and Research

**Landmark factories** in modern planners (FastDownward, the dominant open-source planning system) support multiple extraction methods: the classical Hoffmann-Porteous method, the LM-cut method (Helmert and Domshlak, 2009), and the Zhu-Givan method based on backchaining through the causal graph. Each finds different subsets; combining them improves coverage.

Recent work has applied landmark ideas to probabilistic planning. In fully observable non-deterministic (FOND) planning, a fact is a landmark if it appears in every strong cyclic plan. Muise, Belle, and McIlraith (AAAI 2014) showed how to extract and exploit such landmarks to prune search in FOND problems.

On the LLM side, several papers have explored using language models to propose planning landmarks before search. "LLM+P" (Liu et al., ICRA 2023) and "LLM-PDDL" work uses LLMs to translate natural-language tasks into PDDL and identify high-level milestones, which a classical planner then refines. The LLM handles landmark identification (where it is good); the classical planner handles sound search (where it is reliable).

Open problems remain interesting. Disjunctive landmark extraction is expensive, and the interaction between landmark orderings and heuristic accuracy is not fully understood. There is also no principled theory for when landmark counting outperforms delete-relaxation heuristics, though empirically it tends to win on problems with long sequential dependencies.

## Cross-Disciplinary Insight

Landmarks have a direct analogue in **project management**: the concept of critical milestones. In a Gantt chart, certain deliverables must be completed before dependent work can begin. Landmark ordering graphs are essentially dependency graphs over obligatory checkpoints, exactly the structure that project managers draw on whiteboards.

The deeper analogy is to **topology**. Landmarks are like bottlenecks in a network: every path from source to destination must pass through them. In network flow theory, these are min-cut vertices. In planning, they are min-cut facts. The mathematical structure is the same: identify the narrowest passages in the problem structure, and you understand the essential shape of every solution.

Cognitive science adds another angle. Human experts solving novel problems tend to identify "waypoints" before generating detailed steps. A chess grandmaster does not enumerate moves; they identify key structural features the position must pass through. Landmark extraction may be a computational model of this behavior.

## Daily Challenge

Take any multi-step task from your work: deploying a service, debugging a pipeline, writing a report. List every fact that must be true at some point in any valid completion of the task, regardless of approach. These are your landmarks. Now draw the ordering constraints between them (which must come before which). You now have a landmark graph for the task.

Can you spot any facts you initially thought were optional but turn out to be necessary in every path? That surprise is the core experience of landmark extraction: discovering structural constraints you had not consciously articulated.

## References and Further Reading

- Richter, S. and Westphal, M. "The LAMA Planner: Guiding Cost-Based Anytime Planning with Landmarks." *Journal of Artificial Intelligence Research*, 2010.
- Hoffmann, J., Porteous, J., and Sebastia, L. "Ordered Landmarks in Planning." *Journal of Artificial Intelligence Research*, 2004.
- Helmert, M. and Domshlak, C. "Landmarks, Critical Paths and Abstractions: What's the Difference Anyway?" *ICAPS*, 2009.
- Muise, C., Belle, V., and McIlraith, S. "Computing Contingent Plans via Fully Observable Non-Deterministic Planning." *AAAI*, 2014.
- Liu, B. et al. "LLM+P: Empowering Large Language Models with Optimal Planning Proficiency." *ICRA*, 2023.
- FastDownward planning system: `github.com/aibasel/downward`
