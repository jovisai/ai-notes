---
title: "The Architecture is the Prompt - Guiding AI with Hexagonal Design"
date: 2025-11-12
tags: ["ai", "software-development", "agentic-ai", "systems-architecture", "coding"]
---

As developers, we've all felt the frustration. You have a well-structured repository with clear patterns, but the new AI coding assistant you're working with seems determined to ignore them. Despite providing documentation, examples, and explicit instructions in your prompts, the AI generates code that tangles concerns, bypasses your service layers, and writes directly to the database from a controller.

The immediate reaction is to blame the tool and focus on "better prompt engineering." We try longer, more detailed prompts, hoping that with enough context, the AI will finally understand. But this is often a losing battle. The problem isn't just the prompt; it's that we're asking the AI to understand the invisible, implicit rules of a complex system.

What if we reframed the problem? Instead of trying to make the AI follow the rules, what if we built systems where the rules are explicit and unavoidable? This is where **Hexagonal Architecture**, also known as the **Ports and Adapters** pattern, becomes a game-changer for AI collaboration. The core insight is this: **the architecture is the prompt.**

## Why Prompt Engineering Alone Fails

When a codebase has tangled dependencies and mixed concerns, its architecture is implicit. The "rules" live in documentation, tribal knowledge, and the minds of senior developers. An AI, no matter how advanced, struggles to perceive these unwritten laws.

Trying to control an AI in such an environment through prompts is like giving someone a 20-page document on how to navigate a maze. A much better solution is to redesign the maze to have clear, simple paths.

A well-structured architecture doesn't need a lengthy explanation. It naturally guides both human and AI developers toward the correct implementation. The code won't compile or run if you break the rules.

## Introducing Hexagonal Architecture

Hexagonal Architecture is a design pattern that decouples the core business logic of your application from the external services it interacts with, such as databases, APIs, and user interfaces.

It achieves this by creating a clear separation between the "inside" and the "outside" of your application:

1.  **The Core (The Hexagon)**: This is the heart of your application. It contains the pure, unadulterated business logic and domain models (e.g., an `Order` object with its rules). The core has no dependencies on any external technology. It doesn't know about PostgreSQL, REST, or gRPC.
2.  **Ports (The Gates)**: These are interfaces that define how information flows in and out of the core. They are the formal contracts for interaction.
    *   **Inbound/Driving Ports**: Define how the outside world can interact with your application (e.g., a `CreateOrderUseCase` interface).
    *   **Outbound/Driven Ports**: Define what your application needs from the outside world (e.g., an `OrderRepository` interface to save an order).
3.  **Adapters (The Bridges)**: These are the concrete implementations of the ports. They translate requests from the outside world into calls on the inbound ports and implement the outbound ports using specific technologies.
    *   **Driving Adapters**: A REST controller that calls the `CreateOrderUseCase`.
    *   **Driven Adapters**: A PostgreSQL implementation of the `OrderRepository` interface.

This structure creates a powerful "dependency inversion." The infrastructure code (like database access) depends on the core business logic, not the other way around.

### A High-Level View

Here is a Graphviz diagram illustrating the overall architecture. The core is isolated, and all interactions happen through well-defined ports.

![Hexagonal architecture](/hexagon_hld.png)

## The Power of Targeted Prompts

When your architecture is this clean, your prompts to an AI agent become radically simpler and more effective. You are no longer describing the entire system; you are asking for a small, isolated change within a clearly defined boundary.

Let's consider an example: adding a user deactivation feature.

### Prompt 1: Modify the Domain Logic

First, we ask the AI to change the core business model. The prompt is focused purely on business rules.

> "In the `User` domain model, add a `deactivate()` method. This method should set the user's status to `INACTIVE` and record the current timestamp in a `deactivatedAt` field. Ensure the object is always in a valid state."

The AI can't make a mistake and call a database here. The `User` object has no access to database libraries. The compiler or interpreter will enforce this boundary.

### Prompt 2: Implement a New Use Case

Next, we create the application service that orchestrates the deactivation.

> "Create a `DeactivateUser` use case. It should accept a `userId`, use the `UserRepository` port to load the `User` aggregate, call the user's `deactivate()` method, and then use the repository to save the updated user."

The prompt is a simple set of instructions. The AI is guided by the interfaces (`UserRepository`) and the domain model (`User`).

### Prompt 3: Create a Driving Adapter

Finally, we expose this functionality via a REST API.

> "Create a new endpoint `/users/{id}/deactivate` in the `User` REST controller. It should be a `POST` request. This endpoint should call the `DeactivateUser` use case with the provided user ID."

Each prompt is small, targeted, and builds upon the solid foundation of the architecture. The AI's task is constrained and clear at every step.

### A Detailed Diagram of the Flow

This Graphviz diagram shows the specific components and dependencies for our example. Notice how all arrows point inward, demonstrating the dependency rule.

![diagram shows the specific components and dependencies](detailed_flow.png)

## The Benefits for AI Collaboration

Adopting a hexagonal architecture yields immense benefits when working with AI:

1.  **Clarity and Focus**: The AI works on one small, isolated component at a time. Cognitive load is drastically reduced.
2.  **Structural Enforcement**: The compiler becomes your best friend. The AI physically cannot violate architectural boundaries because the necessary dependencies are not available.
3.  **Enhanced Testability**: It's trivial to ask an AI to write unit tests for your pure domain logic or use cases because they can be tested in isolation without mocking complex infrastructure.
4.  **Reduced Hallucination**: With clear interfaces (ports) acting as contracts, the AI has a precise target for its implementation, reducing the chances it will invent incorrect solutions.

## Take away

Stop spending hours crafting the "perfect prompt" to tame an AI in a complex codebase. Instead, invest in a clean architecture that does the hard work for you. By embracing Hexagonal Architecture, you create a system with clear boundaries, single responsibilities, and explicit contracts.

In this world, the architecture itself becomes the ultimate prompt—a silent guide that naturally constrains and directs the AI to produce code that is clean, maintainable, and correct by design. This not only improves your collaboration with AI but also leads to more robust and scalable software for years to come.
