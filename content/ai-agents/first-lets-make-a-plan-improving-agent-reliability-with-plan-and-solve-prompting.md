---
title: "Improving Agent Reliability with Plan-and-Solve Prompting"
date: 2025-10-05
tags: ["AI Agents", "Prompting", "Planning", "Reasoning", "Reliability"]
---

## 1. Concept Introduction

Before you cook a complex meal, you read the recipe. Before you assemble furniture, you consult the instructions. In any complex task, separating the *planning* from the *doing* is a fundamental strategy for success. You don't just start mixing ingredients or screwing parts together randomly; you first form a plan.

**Plan-and-Solve (PS) Prompting** applies this exact logic to AI agents. Instead of asking an LLM to solve a multi-step problem in one go, you instruct it to first break down the problem and create a clear, step-by-step plan. Only after the plan is explicitly written out do you then instruct the model to execute that plan.

This simple separation of concerns dramatically improves the reliability and accuracy of agents on any task that requires more than one step, from solving math problems to executing complex user requests.

## 2. Historical & Theoretical Context

Plan-and-Solve Prompting is a refinement of Chain of Thought (CoT) reasoning. While CoT encourages the model to "think step by step," the planning and execution are often interleaved and implicit. This can lead to the model getting lost in its own reasoning or making early calculation errors that derail the entire process.

Researchers, such as those behind the 2023 paper *"Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning..."*, identified this weakness. They found that by explicitly telling the model to first devise a plan and then execute it, they could significantly improve performance on complex reasoning tasks. The core insight is that this decouples the strategic part of the problem (the plan) from the tactical part (the execution), reducing the model's cognitive load at each stage.

## 3. The Mechanics: A Two-Part Structure

The elegance of Plan-and-Solve lies in its simplicity. It's not a complex algorithm but a specific structure for your prompt.

A standard prompt might ask:
> "Solve the following problem: [Problem Description]"

A Plan-and-Solve prompt looks like this:
> "[Problem Description]
>
> First, understand the problem and devise a plan to solve it.
> Then, carry out the plan step by step to arrive at the solution."

This structure guides the LLM to produce a much more organized and reliable output:

**Model Output:**
```
**The Plan:**
1.  First, I will calculate the cost of the apples.
2.  Next, I will calculate the cost of the oranges.
3.  Then, I will sum the costs to find the total.
4.  Finally, I will subtract the total from the initial amount to find the change.

**Execution:**
1.  The cost of 3 apples at $0.50 each is 3 * 0.50 = $1.50.
2.  The cost of 5 oranges at $0.75 each is 5 * 0.75 = $3.75.
3.  The total cost is $1.50 + $3.75 = $5.25.
4.  The change from $10.00 is $10.00 - $5.25 = $4.75.

**Final Answer:** The final answer is $4.75.
```

By forcing the model to commit to a plan upfront, we anchor its reasoning process and make it less likely to wander off-track during the execution phase.

## 4. Design Patterns & Architectures

-   **The Concrete Planner:** PS Prompting is the most direct way to implement the **Planner** module in a **Planner-Executor** agent architecture. The output of the "Plan" section of the prompt becomes the formal plan that the Executor component will follow.
-   **Plan Validation & Human-in-the-Loop:** Because the plan is explicit and generated first, it creates a critical checkpoint. Before any tools are called or irreversible actions are taken, the plan can be reviewed. This review can be done by:
    -   A human user ("Does this plan look correct to you?").
    -   A separate, "Critique" AI agent that is specifically prompted to find flaws in plans.
    -   A set of programmatic checks or rules.
    This allows for correction *before* execution, which is far more efficient than cleaning up after a flawed execution.
-   **Tool Selection:** The planning phase can be used to determine which tools are needed to solve a problem. The agent can be prompted to include the specific tools it will use in its plan (e.g., "Step 1: Use the `web_search` tool to find the current price of gold.").

## 5. Practical Application

Let's see the difference with a math word problem.

**Problem:** "A juggler is using 5 clubs. He starts with 1 club in his hand and 4 in the air. He throws a club every 0.5 seconds. How many clubs will he have thrown in one minute?"

**Standard CoT Prompt (might fail):**
> "A juggler has 5 clubs... How many clubs thrown in one minute? Let's think step by step. There are 60 seconds in a minute. He throws a club every 0.5 seconds, so that's 60 / 0.5 = 120 throws. The number of clubs doesn't matter. So, 120 throws."
> *(This is a common error, as it doesn't consider the cyclical nature of juggling.)*

**Plan-and-Solve Prompt (more reliable):**
> "A juggler is using 5 clubs... How many clubs thrown in one minute?
>
> First, devise a plan to solve this. Then, execute the plan."

**Model's PS Output:**
> **The Plan:**
> 1.  Understand the core action: a throw happens every 0.5 seconds.
> 2.  Calculate the total number of throws possible in 60 seconds.
> 3.  Consider the number of clubs. The problem is about the number of *throws*, not which specific club is thrown. The number of clubs is extra information not needed for the final calculation.
> 4.  Execute the calculation.
>
> **Execution:**
> 1.  The time period is one minute, which is 60 seconds.
> 2.  A throw occurs every 0.5 seconds.
> 3.  The total number of throws is the total time divided by the time per throw.
> 4.  Calculation: 60 seconds / 0.5 seconds/throw = 120 throws.
>
> **Final Answer:** The juggler will have thrown 120 clubs.
> *(Even if the answer is the same, the reasoning is more robust and the plan explicitly states that the number of clubs is irrelevant, reducing the chance of error.)*

## 6. Comparisons & Tradeoffs

-   **vs. Chain of Thought (CoT):** PS is a more structured and explicit form of CoT. It improves reliability by separating planning from doing. CoT is simpler and can be sufficient for less complex tasks, but PS is superior for multi-step reasoning.
-   **vs. Tree of Thoughts (ToT):** ToT is about exploring *many different plans* in parallel. PS is about creating and executing *one good plan*. PS is vastly cheaper and faster, making it a great default choice. ToT is a more powerful but resource-intensive technique reserved for very complex problems that require exploration.

-   **Strengths:**
    -   Significantly improves accuracy and reliability on multi-step tasks.
    -   Makes the agent's reasoning process transparent and easy to debug.
    -   Creates a natural checkpoint for plan validation.
-   **Limitations:**
    -   Can add slight verbosity and latency for very simple, one-step problems.
    -   The model can still make mistakes during the execution phase, even with a perfect plan.

## 7. Latest Developments & Research

-   **Adaptive Planning:** The most advanced agents use the PS pattern in a dynamic loop. They generate an initial plan, execute only the first step, and then observe the result. Based on this new information, they *update or refine the rest of the plan* before proceeding. This makes the agent far more robust to unexpected outcomes.
-   **Hierarchical Planning:** For very large tasks, agents can use PS in a nested way. The top-level plan might have a step like "Step 1: Research competitor strategies." This step is then passed to a sub-agent, which uses PS again to create a detailed plan for how it will conduct that research.

## 8. Cross-Disciplinary Insight

The Plan-and-Solve pattern is a direct reflection of established methodologies in **Project Management** and **Software Engineering**.
-   The process is analogous to the **Waterfall model**, where a full project plan and specification are created before any coding begins. This front-loading of planning is known to reduce errors in large, complex projects.
-   It also mirrors the planning phase of an **Agile sprint**. The team first defines the stories and tasks for the sprint (the plan) before beginning the implementation work (the execution). This structured approach is a proven method for managing complexity in any domain, human or artificial.

## 9. Daily Challenge / Thought Exercise

Pick a non-trivial task you need to complete, like "filing your taxes" or "booking a vacation." Before you do anything, open a document and use the Plan-and-Solve structure:
1.  **The Plan:** Write a detailed, numbered list of every step you need to take, from gathering documents to making the final submission or booking.
2.  **Execution:** Go through the plan and execute each step.

Notice how having the complete plan laid out beforehand reduces mental overhead and prevents you from forgetting a critical step. This is the same benefit an AI agent gets from this prompting strategy.

## 10. References & Further Reading

1.  **Wang, L., et al. (2023).** *Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought...* [https://arxiv.org/abs/2305.04091](https://arxiv.org/abs/2305.04091) (The key research paper).
2.  **Microsoft - Prompting Engineering Guide:** [https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/prompt-engineering](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/prompt-engineering) (Discusses various strategies, including plan-and-solve).
3.  **LangChain Expression Language (LCEL):** [https://python.langchain.com/docs/expression_language/](https://python.langchain.com/docs/expression_language/) (The tools in frameworks like LangChain are used to chain together the planning and execution steps of a PS agent).
---
