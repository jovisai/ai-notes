---
title: "Improving Agent Reliability with Plan-and-Solve Prompting"
date: 2025-10-05
tags: ["AI Agents", "Prompting", "Planning", "Reasoning", "Reliability"]
---

## Concept Introduction

**Plan-and-Solve (PS) Prompting** is a structured approach to AI agent reliability. Instead of asking an LLM to solve a multi-step problem in one go, you instruct it to first break down the problem and create a clear, step-by-step plan. Only after the plan is explicitly written out do you then instruct the model to execute that plan.

This separation of concerns dramatically improves the reliability and accuracy of agents on any task that requires more than one step, from solving math problems to executing complex user requests.

## Historical & Theoretical Context

Plan-and-Solve Prompting is a refinement of Chain of Thought (CoT) reasoning. While CoT encourages the model to "think step by step," the planning and execution are often interleaved and implicit. This can lead to the model getting lost in its own reasoning or making early calculation errors that derail the entire process.

Researchers, such as those behind the 2023 paper *"Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning..."*, identified this weakness. They found that by explicitly telling the model to first devise a plan and then execute it, they could significantly improve performance on complex reasoning tasks. The core insight is that this decouples the strategic part of the problem (the plan) from the tactical part (the execution), reducing the model's cognitive load at each stage.

## The Mechanics: A Two-Part Structure

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

## Design Patterns & Architectures

-   **The Concrete Planner:** PS Prompting is the most direct way to implement the **Planner** module in a Planner-Executor agent architecture. The output of the "Plan" section of the prompt becomes the formal plan that the Executor component will follow.
-   **Plan Validation & Human-in-the-Loop:** Because the plan is explicit and generated first, it creates a critical checkpoint. Before any tools are called or irreversible actions are taken, the plan can be reviewed. This review can be done by a human user, a separate "Critique" AI agent prompted to find flaws, or a set of programmatic checks. Correction before execution is far more efficient than cleaning up after a flawed run.
-   **Tool Selection:** The planning phase can be used to determine which tools are needed to solve a problem. The agent can be prompted to include the specific tools it will use in its plan (e.g., "Step 1: Use the `web_search` tool to find the current price of gold.").

## Practical Application

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

## Latest Developments & Research

-   **Adaptive Planning:** The most advanced agents use the PS pattern in a dynamic loop. They generate an initial plan, execute only the first step, and then observe the result. Based on this new information, they *update or refine the rest of the plan* before proceeding. This makes the agent far more robust to unexpected outcomes.
-   **Hierarchical Planning:** For very large tasks, agents can use PS in a nested way. The top-level plan might have a step like "Step 1: Research competitor strategies." This step is then passed to a sub-agent, which uses PS again to create a detailed plan for how it will conduct that research.

## Cross-Disciplinary Insight

The Plan-and-Solve pattern is a direct reflection of established methodologies in **Project Management** and **Software Engineering**.
-   The process is analogous to the **Waterfall model**, where a full project plan and specification are created before any coding begins. This front-loading of planning is known to reduce errors in large, complex projects.
-   It also mirrors the planning phase of an **Agile sprint**: the team first defines the stories and tasks for the sprint before beginning implementation. Decoupling strategy from execution is a recurring theme across engineering disciplines.

## Daily Challenge / Thought Exercise

Pick a non-trivial task you need to complete, like "filing your taxes" or "booking a vacation." Before you do anything, open a document and use the Plan-and-Solve structure:
1.  **The Plan:** Write a detailed, numbered list of every step you need to take, from gathering documents to making the final submission or booking.
2.  **Execution:** Go through the plan and execute each step.

Notice how having the complete plan laid out beforehand reduces mental overhead and prevents you from forgetting a critical step. This is the same benefit an AI agent gets from this prompting strategy.

## References & Further Reading

1.  **Wang, L., et al. (2023).** *Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought...* [https://arxiv.org/abs/2305.04091](https://arxiv.org/abs/2305.04091) (The key research paper).
2.  **Microsoft - Prompting Engineering Guide:** [https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/prompt-engineering](https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/prompt-engineering) (Discusses various strategies, including plan-and-solve).
3.  **LangChain Expression Language (LCEL):** [https://python.langchain.com/docs/expression_language/](https://python.langchain.com/docs/expression_language/) (The tools in frameworks like LangChain are used to chain together the planning and execution steps of a PS agent).
---
