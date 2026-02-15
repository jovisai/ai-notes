---
title: "Building Reusable Agent Capabilities with Skill Libraries"
date: 2025-12-06
draft: false
tags: ["ai-agents", "skill-learning", "hierarchical-rl", "program-synthesis", "compositionality"]
categories: ["AI Agents"]
description: "How AI agents acquire, store, and compose reusable skills—from hierarchical reinforcement learning to LLM-based skill synthesis and the emerging paradigm of lifelong learning agents."
---

## What If Agents Could Learn Once and Reuse Forever?

Imagine teaching a robot to open a door. Without skill libraries, the robot starts from scratch every time it encounters a door—even if it's opened thousands before. With a skill library, the agent says: "I already know how to open doors. Let me use that skill and focus on what's new."

**Skill libraries** are structured collections of reusable behaviors that agents can invoke, compose, and refine. Instead of solving every problem from first principles, agents with skill libraries build up a repertoire of capabilities over time—much like how humans develop muscle memory and procedural knowledge.

This paradigm shift matters because it enables:
- **Transfer learning**: Skills learned in one context apply to others
- **Compositional reasoning**: Complex behaviors emerge from combining simple skills
- **Lifelong learning**: Agents improve without catastrophic forgetting

## Historical & Theoretical Context

### Origins in Robotics and Hierarchical Control

The idea of decomposing complex behaviors into reusable primitives has deep roots:

1. **Motor Primitives (1990s)**: Neuroscientists proposed that biological movement arises from combining "motor primitives"—basic building blocks of motion (Mussa-Ivaldi & Bizzi, 2000)

2. **Options Framework (1999)**: Sutton, Precup, and Singh formalized "options" in reinforcement learning—temporally extended actions with initiation sets, policies, and termination conditions

3. **Hierarchical Task Networks (HTN)**: Classical AI planning used method libraries to decompose high-level tasks into primitive actions

4. **Skill Chaining (2009)**: Konidaris and Barto showed agents could automatically discover and chain skills in continuous domains

### The LLM Revolution

The emergence of large language models created new possibilities:

- **Code as Skills**: Instead of neural network policies, skills can be programs—interpretable, composable, and editable
- **Language-Guided Discovery**: Natural language descriptions help agents index and retrieve relevant skills
- **Zero-Shot Composition**: LLMs can combine skills they've never seen together by reasoning about descriptions

Key milestones include VOYAGER (2023), which demonstrated autonomous skill discovery in Minecraft, and Eureka (2023), which used LLMs to generate reward functions for skill learning.

## The Anatomy of a Skill Library

### What Is a Skill?

A **skill** is a reusable, parameterized behavior with:

```python
@dataclass
class Skill:
    name: str                    # Human-readable identifier
    description: str             # What the skill does (for retrieval)
    preconditions: Callable      # When can this skill be invoked?
    parameters: Dict[str, Type]  # Input arguments
    implementation: Callable     # The actual behavior (code or policy)
    postconditions: Callable     # What's true after execution?
    examples: List[str]          # Usage examples for LLM prompting
```

### Example: A Navigation Skill

```python
navigate_to_skill = Skill(
    name="navigate_to",
    description="Move the agent to a target location while avoiding obstacles",
    preconditions=lambda state: state.agent_can_move,
    parameters={"target": "Position", "speed": "float"},
    implementation=navigation_policy,
    postconditions=lambda state, target: distance(state.position, target) < 0.1,
    examples=[
        "navigate_to(target=kitchen, speed=1.0)",
        "navigate_to(target=Position(5, 3), speed=0.5)"
    ]
)
```

### Skill Library Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    SKILL LIBRARY                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Skill A    │  │   Skill B    │  │   Skill C    │  │
│  │  (primitive) │  │  (primitive) │  │  (composite) │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                 │                  │          │
│  ┌──────▼─────────────────▼──────────────────▼──────┐  │
│  │              Skill Index / Embeddings             │  │
│  │         (for semantic retrieval)                  │  │
│  └──────────────────────┬───────────────────────────┘  │
│                         │                               │
│  ┌──────────────────────▼───────────────────────────┐  │
│  │              Skill Composer / Planner             │  │
│  │    (chains skills to achieve complex goals)       │  │
│  └──────────────────────────────────────────────────┘  │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Algorithms for Skill Acquisition

### 1. Option Discovery via Subgoal Detection

The classic approach identifies **bottleneck states**—states frequently visited on paths to goals—and creates skills to reach them.

**Betweenness-Based Discovery:**

```python
def discover_skills_betweenness(trajectories, graph):
    """Find bottleneck states and create skills to reach them"""
    # Compute betweenness centrality
    betweenness = {}
    for node in graph.nodes:
        paths_through_node = count_shortest_paths_through(node, graph)
        total_paths = count_all_shortest_paths(graph)
        betweenness[node] = paths_through_node / total_paths

    # Top-k nodes become subgoals
    subgoals = sorted(betweenness, key=betweenness.get, reverse=True)[:k]

    # Create skills to reach each subgoal
    skills = []
    for subgoal in subgoals:
        skill = learn_goal_reaching_policy(
            goal=subgoal,
            trajectories=trajectories
        )
        skills.append(skill)

    return skills
```

### 2. LLM-Based Skill Synthesis (VOYAGER-Style)

Modern approaches use LLMs to write skills as code:

```python
def synthesize_skill_with_llm(task_description, environment_api, existing_skills):
    """Generate a new skill using an LLM"""

    prompt = f"""
    You are a skill synthesis agent. Write a Python function to accomplish:

    Task: {task_description}

    Available API:
    {environment_api}

    Existing skills you can call:
    {[s.name + ': ' + s.description for s in existing_skills]}

    Write a function that:
    1. Has a clear name describing what it does
    2. Uses existing skills when possible
    3. Handles edge cases gracefully
    4. Returns True on success, False on failure

    ```python
    def new_skill(...):
    ```
    """

    code = llm.generate(prompt)

    # Verify the skill works
    success = test_skill_in_sandbox(code, environment_api)

    if success:
        skill = parse_code_to_skill(code)
        return skill
    else:
        # Retry with error feedback
        return synthesize_skill_with_llm(
            task_description + f"\nPrevious attempt failed: {error}",
            environment_api,
            existing_skills
        )
```

### 3. Skill Verification Loop

Critical insight: generated skills must be **verified** before adding to the library.

```
┌──────────────────────────────────────────────────────┐
│              SKILL VERIFICATION LOOP                  │
├──────────────────────────────────────────────────────┤
│                                                       │
│   ┌─────────┐    ┌──────────┐    ┌─────────────┐    │
│   │ Generate│───>│ Test in  │───>│  Success?   │    │
│   │  Skill  │    │ Sandbox  │    └──────┬──────┘    │
│   └─────────┘    └──────────┘           │           │
│        ▲                                 │           │
│        │              ┌──────────────────┴───┐      │
│        │              │                      │      │
│        │         ┌────▼────┐          ┌─────▼────┐ │
│        │         │  NO     │          │   YES    │ │
│        │         │ Refine  │          │ Add to   │ │
│        └─────────│ + Retry │          │ Library  │ │
│                  └─────────┘          └──────────┘ │
│                                                      │
└──────────────────────────────────────────────────────┘
```

## Skill Retrieval and Composition

### Semantic Skill Retrieval

When facing a new task, agents retrieve relevant skills using semantic similarity:

```python
class SkillLibrary:
    def __init__(self, embedding_model):
        self.skills = []
        self.embeddings = []
        self.embedding_model = embedding_model

    def add_skill(self, skill: Skill):
        # Embed the skill description
        embedding = self.embedding_model.encode(
            f"{skill.name}: {skill.description}"
        )
        self.skills.append(skill)
        self.embeddings.append(embedding)

    def retrieve(self, task_description: str, top_k: int = 5) -> List[Skill]:
        """Find most relevant skills for a task"""
        query_embedding = self.embedding_model.encode(task_description)

        # Cosine similarity
        similarities = [
            np.dot(query_embedding, emb) /
            (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
            for emb in self.embeddings
        ]

        # Return top-k skills
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.skills[i] for i in top_indices]
```

### Skill Composition Strategies

**1. Sequential Composition (Chaining)**

```python
def chain_skills(skills: List[Skill], goal):
    """Execute skills in sequence"""
    for skill in skills:
        if not skill.preconditions(current_state):
            return False, "Precondition failed"

        success = skill.implementation(current_state)

        if not success:
            return False, f"Skill {skill.name} failed"

    return True, "Goal achieved"
```

**2. Hierarchical Composition**

Higher-level skills call lower-level ones:

```python
make_coffee_skill = Skill(
    name="make_coffee",
    description="Prepare a cup of coffee",
    implementation=lambda: (
        skill_library.get("navigate_to")(target="kitchen") and
        skill_library.get("pick_up")(object="coffee_cup") and
        skill_library.get("use_machine")(machine="coffee_maker") and
        skill_library.get("pour")(source="coffee_maker", target="cup")
    )
)
```

**3. LLM-Planned Composition**

Let an LLM decide which skills to combine:

```python
def plan_with_skills(task: str, skill_library: SkillLibrary, llm):
    relevant_skills = skill_library.retrieve(task, top_k=10)

    prompt = f"""
    Task: {task}

    Available skills:
    {[f"- {s.name}: {s.description}" for s in relevant_skills]}

    Plan a sequence of skill calls to accomplish the task.
    Output as JSON: [{{"skill": "name", "params": {{...}}}}, ...]
    """

    plan = llm.generate(prompt)
    return json.loads(plan)
```

## Practical Implementation: A Complete Skill Library System

Here's a working implementation combining the concepts:

```python
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer

@dataclass
class Skill:
    name: str
    description: str
    implementation: Callable
    parameters: Dict[str, type] = field(default_factory=dict)
    preconditions: Optional[Callable] = None
    postconditions: Optional[Callable] = None
    success_count: int = 0
    failure_count: int = 0

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0

    def execute(self, **kwargs) -> bool:
        try:
            result = self.implementation(**kwargs)
            if result:
                self.success_count += 1
            else:
                self.failure_count += 1
            return result
        except Exception as e:
            self.failure_count += 1
            return False


class SkillLibrary:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.skills: Dict[str, Skill] = {}
        self.encoder = SentenceTransformer(embedding_model_name)
        self.embeddings: Dict[str, np.ndarray] = {}

    def add(self, skill: Skill) -> None:
        """Add a skill to the library"""
        self.skills[skill.name] = skill
        # Create embedding for retrieval
        text = f"{skill.name}: {skill.description}"
        self.embeddings[skill.name] = self.encoder.encode(text)
        print(f"Added skill: {skill.name}")

    def get(self, name: str) -> Optional[Skill]:
        """Get a skill by exact name"""
        return self.skills.get(name)

    def search(self, query: str, top_k: int = 5) -> List[Skill]:
        """Semantic search for relevant skills"""
        query_emb = self.encoder.encode(query)

        scores = []
        for name, emb in self.embeddings.items():
            similarity = np.dot(query_emb, emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(emb)
            )
            scores.append((name, similarity))

        scores.sort(key=lambda x: x[1], reverse=True)
        return [self.skills[name] for name, _ in scores[:top_k]]

    def compose(self, skill_sequence: List[str], **shared_kwargs) -> bool:
        """Execute a sequence of skills"""
        for skill_name in skill_sequence:
            skill = self.get(skill_name)
            if skill is None:
                print(f"Skill not found: {skill_name}")
                return False

            if skill.preconditions and not skill.preconditions():
                print(f"Precondition failed for: {skill_name}")
                return False

            success = skill.execute(**shared_kwargs)
            if not success:
                print(f"Skill failed: {skill_name}")
                return False

        return True

    def statistics(self) -> Dict[str, Any]:
        """Get library statistics"""
        return {
            "total_skills": len(self.skills),
            "skills_by_success": sorted(
                [(s.name, s.success_rate) for s in self.skills.values()],
                key=lambda x: x[1],
                reverse=True
            )
        }


# Example usage
def create_example_library():
    library = SkillLibrary()

    # Define primitive skills
    library.add(Skill(
        name="move_forward",
        description="Move the agent forward by a specified distance",
        parameters={"distance": float},
        implementation=lambda distance: print(f"Moving forward {distance}") or True
    ))

    library.add(Skill(
        name="turn",
        description="Rotate the agent by specified degrees",
        parameters={"degrees": float},
        implementation=lambda degrees: print(f"Turning {degrees} degrees") or True
    ))

    library.add(Skill(
        name="pick_up",
        description="Pick up an object within reach",
        parameters={"object_name": str},
        implementation=lambda object_name: print(f"Picking up {object_name}") or True
    ))

    library.add(Skill(
        name="place",
        description="Place held object at current location",
        parameters={"surface": str},
        implementation=lambda surface: print(f"Placing on {surface}") or True
    ))

    # Composite skill using primitives
    def navigate_to_object(target: str):
        # In practice, this would use path planning
        library.get("turn").execute(degrees=45)
        library.get("move_forward").execute(distance=2.0)
        return True

    library.add(Skill(
        name="navigate_to_object",
        description="Navigate to a named object in the environment",
        parameters={"target": str},
        implementation=navigate_to_object
    ))

    return library


if __name__ == "__main__":
    library = create_example_library()

    # Semantic search
    print("\nSearching for 'go to something':")
    results = library.search("go to something")
    for skill in results:
        print(f"  - {skill.name}: {skill.description}")

    # Execute a skill
    print("\nExecuting pick_up:")
    library.get("pick_up").execute(object_name="red_cube")

    # Compose skills
    print("\nComposing skill sequence:")
    library.compose(["turn", "move_forward", "pick_up"],
                   degrees=90, distance=1.0, object_name="blue_ball")

    # Statistics
    print("\nLibrary statistics:")
    print(library.statistics())
```

## Comparisons & Tradeoffs

| Approach | Strengths | Weaknesses |
|----------|-----------|------------|
| **Hand-coded Skills** | Reliable, interpretable, precise | Time-consuming, doesn't scale, no adaptation |
| **RL-Learned Options** | Discovers emergent behaviors | Sample inefficient, hard to interpret |
| **LLM-Synthesized Code** | Fast iteration, human-readable | May have bugs, needs verification |
| **Neural Skill Policies** | Handles continuous control | Black box, hard to compose |
| **Hybrid (Code + RL)** | Best of both worlds | Complex to implement |

### When to Use Skill Libraries

**Good fit:**
- Long-horizon tasks with recurring subtasks
- Transfer learning across environments
- Multi-task agents
- Interactive, adaptive systems

**Poor fit:**
- Single, unique tasks
- Highly dynamic environments where skills become stale
- Real-time systems where retrieval adds latency

## Latest Developments & Research

### Recent Breakthroughs (2023-2025)

**1. VOYAGER (Wang et al., 2023)**
- Autonomous Minecraft agent that discovers and stores skills
- Uses GPT-4 to write executable JavaScript code
- Achieved 3x more items than baselines through skill accumulation

**2. Eureka (Ma et al., 2023)**
- LLM-generated reward functions for skill learning
- Outperformed human-designed rewards on dexterous manipulation
- Key insight: iterate on reward code, not policy architecture

**3. Skill-It (Dai et al., 2024)**
- Multi-task skill discovery without predefined task boundaries
- Automatically segments demonstrations into reusable chunks

**4. BOSS (Zhang et al., 2024)**
- Bootstrap Your Own Skills from demonstrations
- No reward engineering—learns from play data

### Open Problems

1. **Skill interference**: When do stored skills become counterproductive?
2. **Abstraction level**: How primitive vs. complex should skills be?
3. **Forgetting**: How to prune outdated skills?
4. **Grounding**: Ensuring code-based skills connect to physical reality

## Cross-Disciplinary Insight: Procedural Memory in Neuroscience

Human skill acquisition follows a similar trajectory:

1. **Cognitive Stage**: Explicit, verbal instructions (declarative)
2. **Associative Stage**: Practiced sequences become fluid
3. **Autonomous Stage**: Skills become automatic (procedural memory)

Neuroimaging shows skill learning involves transfer from prefrontal cortex (executive control) to basal ganglia (automatic execution)—analogous to moving from LLM planning to cached skill execution.

**Implication for AI**: Perhaps agents should maintain two systems:
- **Slow/deliberate**: LLM-based reasoning for novel situations
- **Fast/automatic**: Cached skills for familiar patterns

This mirrors the System 1/System 2 distinction from cognitive psychology.

## Daily Challenge: Build a Skill Discovery Agent

**Task**: Create an agent that automatically discovers and stores skills from demonstration trajectories.

**Setup** (30 minutes):

1. Given: A list of demonstration trajectories (action sequences)
2. Goal: Identify recurring action subsequences as skills

**Starter code**:

```python
def discover_skills_from_demos(demonstrations: List[List[str]], min_length=2, min_frequency=3):
    """
    Find recurring action sequences across demonstrations.

    Args:
        demonstrations: List of action sequences, e.g., [["move", "pick", "place"], ...]
        min_length: Minimum actions in a skill
        min_frequency: Minimum occurrences to be considered a skill

    Returns:
        List of discovered skill patterns
    """
    # TODO: Implement n-gram frequency analysis
    # Hint: Use sliding windows and count subsequence occurrences

    from collections import Counter

    subsequence_counts = Counter()

    # Your implementation here
    for demo in demonstrations:
        for length in range(min_length, len(demo) + 1):
            for start in range(len(demo) - length + 1):
                subseq = tuple(demo[start:start + length])
                subsequence_counts[subseq] += 1

    # Filter by minimum frequency
    skills = [
        seq for seq, count in subsequence_counts.items()
        if count >= min_frequency
    ]

    # Remove subsequences of longer skills
    # ...

    return skills

# Test data
demos = [
    ["navigate", "pick", "navigate", "place"],
    ["navigate", "pick", "navigate", "place", "navigate"],
    ["pick", "navigate", "place"],
    ["navigate", "pick", "navigate", "place", "rest"]
]

skills = discover_skills_from_demos(demos)
print("Discovered skills:", skills)
```

**Extension**: Convert discovered patterns into Skill objects and add to a SkillLibrary.

## References & Further Reading

### Foundational Papers

- **Sutton, R. S., Precup, D., & Singh, S.** (1999). "Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning." *Artificial Intelligence*, 112(1-2), 181-211.

- **Konidaris, G., & Barto, A. G.** (2009). "Skill Discovery in Continuous Reinforcement Learning Domains using Skill Chaining." *NeurIPS*.

### Modern Skill Learning

- **Wang, G., et al.** (2023). "VOYAGER: An Open-Ended Embodied Agent with Large Language Models." *arXiv:2305.16291* [Paper](https://arxiv.org/abs/2305.16291) | [Code](https://github.com/MineDojo/Voyager)

- **Ma, Y., et al.** (2023). "Eureka: Human-Level Reward Design via Coding Large Language Models." *arXiv:2310.12931* [Paper](https://arxiv.org/abs/2310.12931)

### Practical Resources

- **MineDojo**: Open-ended agent benchmark with skill evaluation [GitHub](https://github.com/MineDojo/MineDojo)

- **Hierarchical RL Survey**: Pateria et al. (2021) "Hierarchical Reinforcement Learning: A Comprehensive Survey" [Paper](https://arxiv.org/abs/2011.01835)

### Related Articles in This Series

- "Hierarchical Task Networks: Planning with Decomposition" (formal task decomposition)
- "Tool Composition and Chaining" (composing external capabilities)
- "The Agent's Mind: Architecting Short-Term and Long-Term Memory" (skill storage)

---

**Key Takeaway**: Skill libraries transform agents from one-task wonders into lifelong learners. By structuring knowledge as reusable, composable behaviors, agents can tackle increasingly complex problems while building on past experience. The future of AI agents isn't about bigger models—it's about smarter organization of what they've learned.
