---
title: "AI Is Making Coding Faster. What Happens to the Joy of Solving Problems?"
date: 2026-10-28
typeset: true
tags: ["AI", "Software Engineering", "Productivity", "AI Agents"]
---

{{< figure src="/images/ai-coding-flow/divided-attention.webp" alt="A developer surrounded by software task windows connected by tangled threads, illustrating attention divided across several AI agents." caption="More work can happen in parallel while our attention remains limited." >}}

You delegate a bug, start another agent on a feature, then alternate between prompts, diffs, and test results. Work moves forward, yet the session ends without the satisfaction of having understood something deeply.

AI is shifting more of programming from solving problems to directing and evaluating solutions. That can improve output while changing what makes the work rewarding.

## What gets lost between tasks

Solving a difficult bug builds a continuous mental model: form a hypothesis, test it, revise your understanding. The eventual fix rewards both persistence and discovery.

Supervising several agents can break that continuity. Each return requires reconstructing the task, checking assumptions, and deciding whether the output is correct. Generating solutions scales more easily than understanding them.

Review and architecture can be deeply engaging. The problem is a stream of unrelated decisions that leaves little room to follow one line of thought. Delegating the investigation may also remove opportunities to develop the intuition needed to maintain the result.

## Faster does not always mean better to work with

A May 2026 preprint followed 95 professional engineers across two surveys six months apart. At both points, 84% reported improved productivity with AI assistants. Meanwhile, reports of worsening developer experience in at least one measured dimension rose from 14% to 27%, with deterioration around flow and cognitive load. These were self-reported perceptions, not proof that AI caused the changes. [Study](https://arxiv.org/abs/2605.23135)

The effect is not universal. A study involving JetBrains researchers found that developers perceived better focus when AI answered routine technical questions, helping them continue without interruptions. [Study](https://kairos.jetbrains.com/research/publications/genai-human-interactions)

The distinction is useful: assistance can preserve attention when it supports the current problem, and fragment it when it multiplies competing demands.

## Try one problem, one agent

{{< figure src="/images/ai-coding-flow/focused-collaboration.webp" alt="A developer focused on one monitor with a small AI assistant window, beside a tangled teal thread becoming an orderly loop." caption="One problem, sustained attention, and AI assistance that stays within the same line of thought." >}}

For one 60–90 minute session, choose a single problem and a primary agent. Use AI for exploration and implementation; spend its execution time examining assumptions, failure cases, and validation criteria.

Compare the session with your usual workflow: completed work, context switches, fatigue, and how clearly you can explain the solution afterward. Treat this as an experiment; independent tasks may still benefit from parallel agents.

Are we designing coding tools to help engineers think and build well, or to keep as many agents busy as possible?
