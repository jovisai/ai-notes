---
title: "Anytime Heuristic Search How Agents Find Good Plans Fast and Improve Them Over Time"
date: 2026-03-17
draft: false
tags: ["ai-agents", "search", "planning", "heuristics", "algorithms"]
description: "Anytime heuristic search, including Weighted A* and ARA*, lets agents commit to a suboptimal plan immediately and refine it as time allows, with provable bounds on solution quality at every step."
---

Planning agents face a brutal tradeoff. A* search, run to completion, finds the provably optimal path. But the search can take a long time, and in many real scenarios the agent must act before it finishes. A robot navigating a warehouse cannot wait two seconds for an optimal route when a customer is watching. A game-playing agent has a fixed clock. A scheduling system needs a plan now, not a perfect plan five minutes from now.

**Anytime search** resolves this by separating two concerns: getting a valid solution quickly and improving its quality over time. The agent always has an answer it can act on, and every additional compute cycle makes that answer better. The quality guarantee is explicit: at any moment, you know exactly how suboptimal the current solution might be.

## Concept Introduction

A standard A* search finds the optimal solution but commits to a strict ordering that avoids suboptimal nodes. The key modification in anytime search is to inflate the heuristic by a factor $\varepsilon \geq 1$, which biases the search toward the goal aggressively. Instead of minimizing $f(n) = g(n) + h(n)$, weighted A* minimizes:

$$f(n) = g(n) + \varepsilon \cdot h(n)$$

With a higher $\varepsilon$, the heuristic dominates and the frontier expands far fewer nodes before reaching the goal. The solution found is not optimal, but it satisfies a useful guarantee: if $h$ is admissible (never overestimates), then the cost of the weighted A* solution is at most $\varepsilon$ times the optimal cost. A solution at $\varepsilon = 2.5$ costs no more than 2.5x the best possible path.

**ARA*** (Anytime Repairing A*), introduced by Likhachev, Gordon, and Thrun in 2003, iterates on this. It starts with a high $\varepsilon$ to find a solution fast, then decrements $\varepsilon$ and re-runs the search, reusing work from the previous iteration. Each pass tightens the quality bound until $\varepsilon = 1$ (optimal) or time runs out. The agent publishes its best solution after each pass, always ready to act.

## Historical and Theoretical Context

The term "anytime algorithm" comes from Dean and Boddy (1988), who formalized the idea of algorithms that can be interrupted at any time and return a valid answer whose quality improves with more computation. Their motivation was real-time planning systems under resource constraints, a problem that predates modern AI agents but maps perfectly onto them.

Weighted A* itself appeared in the 1970s in work by Pohl. The key insight, that inflating an admissible heuristic trades optimality for speed with a bounded loss, was understood early but treated as a curiosity. ARA* made it systematic by embedding the inflated search inside an anytime loop with guaranteed convergence to optimality.

The formal analysis connects to bounded suboptimality results in classical search theory. If the heuristic $h$ is admissible and consistent, then the solution cost $C$ returned by weighted A* with factor $\varepsilon$ satisfies $C \leq \varepsilon \cdot C^*$, where $C^*$ is the optimal cost. As $\varepsilon$ decreases toward 1, the bound tightens. At $\varepsilon = 1$ you recover standard A* behavior.

## Algorithms and Design Patterns

The ARA* loop is simple to state:

```
epsilon = epsilon_max   # e.g. 3.0
while epsilon >= 1.0 and time_budget_remains:
    run weighted_astar with current epsilon
    if solution found:
        publish solution  # guaranteed <= epsilon * optimal
    epsilon -= delta      # e.g. 0.5 per iteration
```

This pattern shows up in modern agent architectures in several ways. A planning agent with a hard deadline can run ARA* and return whatever it has when the clock expires. A hierarchical agent can allocate a time budget per sub-task and let anytime search fill each slot. A multi-agent system can run ARA* in a background thread while the foreground executes the last committed plan, swapping in the improved plan when it arrives.

The separation between "search in the background" and "act on current best solution in the foreground" is a recurring architectural pattern in robotics and game AI. ARA* makes it theoretically clean because you always know the quality bound of the plan you're executing.

One important design choice is the decrement schedule for $\varepsilon$. A large step (e.g., 1.0) means fewer but coarser iterations. A small step (e.g., 0.1) means more fine-grained improvement but more passes. In practice, a geometric schedule (multiply by a factor < 1 each time) often works better than linear decrement when the search space has varying density.

## Practical Application

Here is a working Python implementation of ARA* on a grid world. The agent searches from start to goal, immediately commits to a fast 3x-suboptimal path, then progressively improves its solution:

```python
import heapq
import math
import time
from collections import defaultdict

def weighted_astar(grid, start, goal, epsilon):
    """Weighted A* with inflation factor epsilon.
    Returns (path, cost) or (None, inf) if unreachable."""
    rows, cols = len(grid), len(grid[0])

    def h(a, b):
        # Euclidean distance: admissible for 8-connected grid
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    g = defaultdict(lambda: float("inf"))
    g[start] = 0.0
    parent = {start: None}
    open_set = [(epsilon * h(start, goal), start)]
    closed = set()

    while open_set:
        _, current = heapq.heappop(open_set)
        if current in closed:
            continue
        closed.add(current)

        if current == goal:
            path, node = [], goal
            while node is not None:
                path.append(node)
                node = parent[node]
            return list(reversed(path)), g[goal]

        r, c = current
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if grid[nr][nc] == 1:  # obstacle
                continue
            step_cost = math.sqrt(dr * dr + dc * dc)
            new_g = g[current] + step_cost
            neighbor = (nr, nc)
            if new_g < g[neighbor]:
                g[neighbor] = new_g
                parent[neighbor] = current
                f = new_g + epsilon * h(neighbor, goal)
                heapq.heappush(open_set, (f, neighbor))

    return None, float("inf")


def ara_star(grid, start, goal, epsilon_start=3.0, epsilon_min=1.0, delta=0.5,
             time_limit=None):
    """ARA*: anytime repairing A*.
    Starts fast and suboptimal, improves quality each iteration."""
    epsilon = epsilon_start
    best = None
    start_time = time.time()

    while epsilon >= epsilon_min - 1e-9:
        elapsed = time.time() - start_time
        if time_limit and elapsed >= time_limit:
            print(f"  Time limit reached after {elapsed:.3f}s")
            break

        path, cost = weighted_astar(grid, start, goal, max(epsilon, 1.0))
        if path is not None:
            best = (path, cost)
            # The suboptimality guarantee: solution cost <= epsilon * optimal
            print(f"  epsilon={epsilon:.1f}  cost={cost:.3f}  "
                  f"(within {epsilon:.1f}x of optimal)")

        epsilon = max(epsilon - delta, epsilon_min - delta)
        if epsilon < epsilon_min:
            break

    return best


if __name__ == "__main__":
    # 0 = free, 1 = wall
    grid = [
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,1,1,0,0,0,1,0,0],
        [0,0,0,1,0,1,0,1,0,0],
        [0,1,0,1,0,1,0,0,0,0],
        [0,1,0,0,0,1,1,1,0,0],
        [0,0,0,1,0,0,0,0,0,0],
        [0,0,1,1,1,0,1,0,0,0],
        [0,0,0,0,1,0,1,0,1,0],
        [0,0,0,0,0,0,1,0,1,0],
        [0,0,0,0,0,0,0,0,0,0],
    ]

    print("ARA* search from (0,0) to (9,9)\n")
    result = ara_star(grid, (0, 0), (9, 9),
                      epsilon_start=3.0, epsilon_min=1.0, delta=0.5,
                      time_limit=1.0)

    if result:
        path, cost = result
        print(f"\nFinal plan: {len(path)} steps, cost={cost:.3f} (epsilon=1.0: optimal)")
```

Running this prints the solution found at each epsilon, showing cost improving with each pass. At $\varepsilon = 3.0$ the agent has a valid (if rough) path in microseconds. By $\varepsilon = 1.0$ it has the optimal path, using the same code path.

## Latest Developments and Research

Several variants have extended ARA* for different agent contexts.

ANA* (Anytime Nonparametric A*, van den Berg et al., 2011) removes the need to specify $\varepsilon$ manually. It maintains a single open list and continuously tightens the bound as nodes are expanded, without requiring restarts. This is more memory efficient for large state spaces.

MPGAA* (Moving Point Graph A*) extends the idea to dynamic environments where the graph changes during search, relevant for agents operating in the real world where new obstacles appear mid-plan.

In robotics planning, ARA* and its successors are standard components in navigation stacks. The PR2 and MoveIt motion planners used ARA*-based approaches extensively because manipulation planning has wildly variable search difficulty and robot controllers need path updates at 10-50Hz.

For LLM-based agents, the anytime philosophy shows up differently. Recent work on "fast-slow" planning (inspired by Kahneman's System 1/System 2 framing) uses a cheap heuristic model to generate a candidate plan immediately, then an expensive model to critique and refine it. The structure mirrors anytime search: commit fast, refine over time, bounded quality.

Open problems include learning good $\varepsilon$ schedules from experience (the optimal decrement depends on the problem structure), combining anytime search with learned heuristics that may be inadmissible, and applying the framework to planning in latent state spaces where the graph is not explicit.

## Cross-Disciplinary Insight

Anytime search formalizes something economists call **satisficing**, a term coined by Herbert Simon in 1956. A fully rational agent maximizes utility, but Simon argued that real agents (human or artificial) search for solutions that are "good enough" given bounded time and cognitive resources. They satisfice rather than optimize.

What anytime search adds to Simon's concept is a mechanism: the agent doesn't just give up and pick the first acceptable solution. It sets an explicit quality floor, finds a solution meeting that floor fast, and then systematically improves it. The quality bound at any moment is known. This is closer to how human experts actually work under time pressure: make a credible first move, then improve if the clock allows.

## Daily Challenge

Take the grid in the code example and replace the fixed $\varepsilon$ schedule with an adaptive one. After each pass, measure how much the cost improved compared to the time spent. If the improvement per second is falling rapidly, stop early rather than continuing to a preset $\varepsilon$. This is sometimes called "expected improvement" stopping.

For bonus difficulty: instrument the search to count how many nodes were expanded at each $\varepsilon$ level. Plot node count vs. cost improvement. What does this curve look like? Is most of the cost reduction happening in the first few passes or spread evenly?

## References and Further Reading

- **"ARA*: Anytime A* with Provable Bounds on Sub-Optimality"**, Likhachev, Gordon, Thrun. Advances in Neural Information Processing Systems (NeurIPS), 2003.
- **"Anytime Algorithms in AI: A Survey"**, Zilberstein. AI Magazine, 1996. The definitive survey of the anytime paradigm.
- **"ANA*: Anytime Nonparametric A*"**, van den Berg, Shah, Huang, Goldberg. Proceedings of AAAI, 2011.
- **"Bounded Suboptimal Search: A Direct Approach Using Inadmissible Estimates"**, Thayer and Ruml. Proceedings of IJCAI, 2011.
- **"Rational Metareasoning and the Feeling of Difficulty"**, Shenhav, Musslick, et al. Trends in Cognitive Sciences, 2017. Connects anytime computation to metacognitive resource allocation in humans.
- **"Fast Downward"**, Helmert. Journal of Artificial Intelligence Research, 2006. The leading classical planner, which uses weighted heuristic search internally; widely used as a baseline for planning research.
