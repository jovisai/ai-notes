---
title: "Systems That Search for New Algorithms Automatically"
date: 2025-09-17
description: "An introduction to AlphaEvolve, DeepMind's new AI agent that uses evolutionary search and LLM-driven code synthesis to autonomously discover and optimize algorithms, outperforming human-designed solutions in several domains."
tags: [AI, LLMs, Research, Deep Research, Autonomous Agents]
---

Let’s start with a **very simple version** so you can see the idea in action.

We’ll use a genetic algorithm to “discover” a formula that approximates a target function.

This isn’t as powerful as DeepMind’s AlphaTensor, but it shows how machines can *search for algorithms or formulas automatically*.

### Example: Discovering a formula for `f(x) = x^2 + x + 1`

```python
import random
import math

# Our target function
def target(x):
    return x**2 + x + 1

# Generate a random "algorithm" = coefficients for ax^2 + bx + c
def random_algorithm():
    return [random.randint(-5, 5) for _ in range(3)]

# Evaluate how good an algorithm is
def fitness(alg):
    a, b, c = alg
    error = 0
    for x in range(-5, 6):  # test on values -5 to 5
        guess = a*x**2 + b*x + c
        error += abs(guess - target(x))
    return -error  # higher is better (less error)

# Mutate the algorithm slightly
def mutate(alg):
    new = alg[:]
    idx = random.randint(0, 2)
    new[idx] += random.choice([-1, 1])
    return new

# Evolutionary search
population = [random_algorithm() for _ in range(20)]

for gen in range(100):
    # Evaluate fitness
    population.sort(key=fitness, reverse=True)
    best = population[0]
    if fitness(best) == 0:  # perfect match
        break
    # Keep top half, mutate to create new
    new_pop = population[:10]
    while len(new_pop) < 20:
        new_pop.append(mutate(random.choice(population[:10])))
    population = new_pop

print("Discovered formula coefficients (a, b, c):", best)
print("Which means: f(x) = {}*x^2 + {}*x + {}".format(*best))
```

---

### What happens here

1. We define a **target function**: `x^2 + x + 1`.
2. The system randomly generates candidate formulas like `-3x^2 + 4x + 2`.
3. Each candidate is **scored** on how close it comes to the real target.
4. The best candidates survive, mutate, and evolve.
5. After several generations, the system “discovers” the correct algorithm `[1, 1, 1]`.

---

This is a toy version of **automatic algorithm discovery**.
Instead of just coefficients, real systems search for **entire algorithmic steps** (loops, branches, optimizations).

👉 Do you want me to extend this so the system can actually **discover sorting algorithms** (like bubble sort or insertion sort) using the same approach? That would be a closer analogy to real-world cases.
