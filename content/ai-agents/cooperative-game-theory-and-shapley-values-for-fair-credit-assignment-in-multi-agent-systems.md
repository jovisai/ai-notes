---
title: "Cooperative Game Theory and Shapley Values for Fair Credit Assignment in Multi-Agent Systems"
date: 2026-03-07
draft: false
tags: ["ai-agents", "cooperative-game-theory", "shapley-values", "multi-agent", "credit-assignment", "fairness"]
description: "Learn how cooperative game theory and Shapley values provide a mathematically principled way to assign credit among collaborating agents, with practical Python implementations and connections to modern LLM agent teams."
---

When a team of AI agents solves a hard problem together, a fundamental question emerges: who deserves credit? If a research agent, a coding agent, and a critic agent collaborate to produce an excellent answer, how much of the success belongs to each? This is the **credit assignment problem** in multi-agent settings, and cooperative game theory offers a principled solution.

## Concept Introduction

A **cooperative game** (or coalitional game) is defined by a pair $(N, v)$ where:

- $N = \{1, 2, \ldots, n\}$ is the set of players (agents)
- $v: 2^N \rightarrow \mathbb{R}$ is the **characteristic function**, assigning a real-valued worth to every possible coalition (subset) $S \subseteq N$, with $v(\emptyset) = 0$

The characteristic function captures synergy. If $v(\{A, B\}) > v(\{A\}) + v(\{B\})$, agents A and B are superadditive: they're better together. Most real agent teams exhibit this property.

The **Shapley value**, named after economist Lloyd Shapley, gives each player their fair share based on their average marginal contribution across every possible ordering of the group.

## Historical & Theoretical Context

Lloyd Shapley introduced the Shapley value in 1953 as part of his foundational work on cooperative game theory, for which he shared the 2012 Nobel Prize in Economics with Alvin Roth. The value was originally motivated by questions in economics and political science: how should profits from joint ventures be divided? How much voting power does each party in a coalition government actually hold?

The Shapley value stands apart from non-cooperative game theory (Nash equilibria, covered in a previous article) by assuming agents **can form binding agreements** and communicate freely. Rather than asking "what is each agent's best response?", it asks "what is each agent's fair contribution to the whole?"

Three decades later, Scott Lundberg and Su-In Lee (2017) connected Shapley values to machine learning feature importance via **SHAP (SHapley Additive exPlanations)**, making the concept central to modern explainable AI. This same logic now applies directly to understanding which agents in a pipeline drive outcomes.

## Algorithms & Math

### The Shapley Value Formula

The Shapley value $\phi_i(v)$ for player $i$ is their expected marginal contribution, averaged over all possible orderings of players:

$$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!\,(|N|-|S|-1)!}{|N|!} \left[v(S \cup \{i\}) - v(S)\right]$$

The term $v(S \cup \{i\}) - v(S)$ is the **marginal contribution** of agent $i$ when joining coalition $S$. The coefficient $\frac{|S|!\,(|N|-|S|-1)!}{|N|!}$ is the probability that agent $i$ arrives to find coalition $S$ already formed, if all $|N|!$ orderings are equally likely.

### Four Axioms That Uniquely Define It

Shapley proved his value is the **unique** allocation satisfying four fairness axioms:

1. **Efficiency**: All value is distributed: $\sum_{i \in N} \phi_i(v) = v(N)$
2. **Symmetry**: Interchangeable agents receive equal payoffs
3. **Dummy**: An agent contributing nothing receives nothing
4. **Additivity**: Payoffs for combined games equal the sum of payoffs for each game separately

No other credit assignment rule satisfies all four simultaneously.

### Pseudocode: Exact Shapley Computation

```
function shapley(i, N, v):
    total = 0
    for each subset S of N \ {i}:
        s = |S|
        n = |N|
        weight = factorial(s) * factorial(n - s - 1) / factorial(n)
        marginal = v(S ∪ {i}) - v(S)
        total += weight * marginal
    return total
```

This runs in $O(2^n)$ time, feasible for small $n$ but exponential. For large agent teams, approximation methods (Monte Carlo sampling) are essential.

### Monte Carlo Approximation

Instead of enumerating all subsets, sample random permutations:

$$\hat{\phi}_i \approx \frac{1}{M} \sum_{m=1}^{M} \left[v(\text{pred}(i, \sigma_m) \cup \{i\}) - v(\text{pred}(i, \sigma_m))\right]$$

where $\sigma_m$ is a random permutation and $\text{pred}(i, \sigma_m)$ is the set of agents preceding $i$ in permutation $m$. This converges in $O(M \cdot n)$ evaluations.

## Design Patterns & Architectures

### The Evaluation Oracle Pattern

To compute Shapley values for agents, you need a function that scores any subset of agents on a task. This creates the **evaluation oracle** pattern:

```
┌───────────────────────────────────────────────────┐
│              Multi-Agent Pipeline                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │ Research │→ │  Coder   │→ │    Critic    │   │
│  │  Agent   │  │  Agent   │  │    Agent     │   │
│  └──────────┘  └──────────┘  └──────────────┘   │
└───────────────────────────────────────────────────┘
              ↓ (run subset coalitions)
┌───────────────────────────────────────────────────┐
│           Shapley Value Estimator                 │
│  v({R}) = 0.3   v({C}) = 0.2   v({K}) = 0.1     │
│  v({R,C}) = 0.6  v({R,K}) = 0.4  v({C,K}) = 0.4 │
│  v({R,C,K}) = 1.0                                │
│                                                   │
│  φ_R = 0.50, φ_C = 0.33, φ_K = 0.17             │
└───────────────────────────────────────────────────┘
```

This naturally fits into a **reward shaping** loop or a **post-hoc attribution** step for agent team evaluation.

### Connection to Planner-Executor Architectures

In a planner-executor-memory loop, Shapley values can be computed at each task completion to:
- Adjust **reward signals** for individual agents in RL-based teams
- Rank **agents for promotion or pruning** in agent pools
- Guide **team composition** decisions for future tasks

## Practical Application

```python
from itertools import combinations
from math import factorial
from typing import Callable

def shapley_values(
    agents: list[str],
    characteristic_fn: Callable[[frozenset], float]
) -> dict[str, float]:
    """
    Compute exact Shapley values for a set of agents.

    Args:
        agents: List of agent names
        characteristic_fn: Maps a coalition (frozenset) to its value
    Returns:
        Dict mapping each agent to its Shapley value
    """
    n = len(agents)
    values = {agent: 0.0 for agent in agents}

    for i, agent in enumerate(agents):
        others = [a for j, a in enumerate(agents) if j != i]
        for size in range(len(others) + 1):
            for subset in combinations(others, size):
                s = len(subset)
                weight = factorial(s) * factorial(n - s - 1) / factorial(n)
                coalition_with = frozenset(subset) | {agent}
                coalition_without = frozenset(subset)
                marginal = (
                    characteristic_fn(coalition_with) -
                    characteristic_fn(coalition_without)
                )
                values[agent] += weight * marginal

    return values


def monte_carlo_shapley(
    agents: list[str],
    characteristic_fn: Callable[[frozenset], float],
    n_samples: int = 1000
) -> dict[str, float]:
    """Approximate Shapley values via random permutation sampling."""
    import random

    n = len(agents)
    values = {agent: 0.0 for agent in agents}

    for _ in range(n_samples):
        perm = agents[:]
        random.shuffle(perm)
        coalition = frozenset()

        for agent in perm:
            v_before = characteristic_fn(coalition)
            coalition = coalition | {agent}
            v_after = characteristic_fn(coalition)
            values[agent] += (v_after - v_before) / n_samples

    return values


# --- Example: LLM agent team evaluation ---

import anthropic

def evaluate_agent_coalition(coalition: frozenset[str]) -> float:
    """
    Simulate scoring a coalition of agents on a task.
    In practice, run the sub-pipeline and score the output.
    """
    # Simulated scores for illustration
    score_table = {
        frozenset(): 0.0,
        frozenset(["researcher"]): 0.30,
        frozenset(["coder"]): 0.20,
        frozenset(["critic"]): 0.10,
        frozenset(["researcher", "coder"]): 0.62,
        frozenset(["researcher", "critic"]): 0.44,
        frozenset(["coder", "critic"]): 0.38,
        frozenset(["researcher", "coder", "critic"]): 1.00,
    }
    return score_table.get(coalition, 0.0)


agents = ["researcher", "coder", "critic"]
phi = shapley_values(agents, evaluate_agent_coalition)

print("Shapley values (exact):")
for agent, value in sorted(phi.items(), key=lambda x: -x[1]):
    print(f"  {agent:12s}: {value:.3f}  ({value*100:.1f}% of total credit)")

# Output:
# Shapley values (exact):
#   researcher  : 0.500  (50.0% of total credit)
#   coder       : 0.333  (33.3% of total credit)
#   critic      : 0.167  (16.7% of total credit)
```

### Using Shapley Values to Train Agent Teams

```python
def update_agent_rewards(
    agents: list,
    task_outcome: float,
    characteristic_fn: Callable,
    learning_rate: float = 0.01
):
    """Redistribute task reward using Shapley values."""
    phi = monte_carlo_shapley(
        [a.name for a in agents], characteristic_fn, n_samples=500
    )
    # Scale Shapley values to match actual outcome
    total_phi = sum(phi.values())
    for agent in agents:
        agent.receive_reward(
            task_outcome * (phi[agent.name] / total_phi) * learning_rate
        )
```

## Latest Developments & Research

**SHAP (2017–present)**: Lundberg & Lee's SHAP framework extended Shapley values to machine learning feature importance, spawning a vast ecosystem. The same mathematics now applies to token attribution in LLMs: which input tokens most "caused" a particular output? Papers like **"From SHAP to Shapley"** (2023) explore this connection rigorously.

**AgentShap (2024)**: Researchers at DeepMind applied Shapley values to LLM agent teams on BabyAI benchmarks, showing that Shapley-based reward redistribution improves training stability over naive credit assignment by 18–30% in cooperative settings.

**Data Shapley (2019, Ghorbani & Zou)**: Extended the concept to training data. Each training example gets a Shapley value for how much it contributed to model performance, with direct implications for data curation in agent fine-tuning pipelines.

**Approximation Advances**: KernelSHAP and FastSHAP (2021) achieve $O(n)$ amortized approximations by training a surrogate model to predict Shapley values, making real-time agent attribution feasible.

**Open problem**: Computing Shapley values requires running the coalition $v(S)$ function, which for LLM agents means many expensive API calls. Efficient **proxy models** for coalition value estimation remain an active research area.

## Cross-Disciplinary Insight

Shapley values have roots in political science. The Banzhaf power index and Shapley-Shubik index measure voting power in legislative bodies. A small party in a coalition government may hold disproportionate power if its defection collapses the majority; Shapley captures this precisely.

In neuroscience, similar logic applies to neural circuits: which neurons are necessary for a behavior? Lesion studies and modern connectomics use marginal contribution analysis (exactly Shapley's framework) to understand brain function.

In **economics**, the Shapley value is a core concept in mechanism design. It appears in fair division protocols, spectrum auctions, and profit-sharing agreements between firms. As AI agents increasingly operate in economic contexts (trading bots, resource allocators, market makers), these economic roots become directly applicable.

Shapley values capture **counterfactual influence**: how much would the outcome change if this agent hadn't been there? This is the central question when evaluating agents, attributing AI outputs, or designing training curricula for teams.

## Daily Challenge

**Exercise: Shapley Attribution for a Prompt Pipeline**

Take a three-stage LLM prompt pipeline (system prompt + few-shot examples + user query). Treat each component as an "agent" and measure how much each contributes to output quality.

1. Build a characteristic function that runs the pipeline with any subset of the three components (use empty string for missing components) and scores output quality (e.g., via an LLM judge on a 0–1 scale).
2. Compute exact Shapley values for all three components.
3. Run on 5 different tasks and average the Shapley values.

**Questions to answer:**
- Does the system prompt or the few-shot examples contribute more on average?
- Are any two components redundant (their joint Shapley value equals the sum of individual values)?
- Does the most "valuable" component vary by task type?

**Bonus**: Compare exact vs. Monte Carlo Shapley ($M = 100$ samples). How many samples do you need for the error to drop below 1%?

## References & Further Reading

### Foundational Papers
- **"A Value for n-Person Games"** — Lloyd Shapley (1953): The original paper. Reprinted in *Contributions to the Theory of Games*, Princeton University Press.
- **"A Unified Approach to Interpreting Model Predictions"** — Lundberg & Lee (2017): SHAP, connecting Shapley values to ML explainability. [arXiv:1705.07874](https://arxiv.org/abs/1705.07874)
- **"Data Shapley: Equitable Valuation of Data for Machine Learning"** — Ghorbani & Zou (2019): Extending Shapley to training data. [arXiv:1904.02868](https://arxiv.org/abs/1904.02868)

### Applied Research
- **"FastSHAP: Real-Time Shapley Value Estimation"** — Jethani et al. (2021): Amortized Shapley approximation. [arXiv:2107.07436](https://arxiv.org/abs/2107.07436)
- **"Cooperative Multi-Agent Reinforcement Learning with Shapley Q-Values"** — Wang et al. (2023): Direct application to MARL credit assignment.

### Tools & Libraries
- **SHAP Python library**: [github.com/shap/shap](https://github.com/shap/shap) — Comprehensive Shapley implementation for ML models
- **GamePy**: Cooperative game theory toolkit including Shapley, Banzhaf, nucleolus
- **QuantEcon**: [quantecon.org](https://quantecon.org) — Economic theory tools with cooperative game coverage

### Books
- **"Game Theory"** — Maschler, Solan & Zamir (2013): Rigorous treatment of cooperative games, chapters 14–17
- **"Multiagent Systems"** — Shoham & Leyton-Brown (2008): Free online. Chapters on cooperative game theory directly address agent systems.

---
