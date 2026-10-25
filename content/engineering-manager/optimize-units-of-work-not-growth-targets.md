---
title: Optimize Units of Work, Not Growth Targets
date: 2026-06-23
draft: false
featured: true
tags: ['engineering-management', 'AI', 'operations']
description: "Growth becomes more actionable when leaders stop treating it as a target and start removing friction from the individual units of work that create it."
---

A revenue target gives a company direction. It does not tell a team what to do on Monday.

Nobody performs a “20% growth” task. People qualify leads, review contracts, resolve tickets, and ship features. To improve the outcome, inspect the repeatable work that produces it.

## Zoom In Until You Can Change the Work

“Automate sales” is too broad to test. Researching one inbound lead has a clearer input, output, and quality standard.

{{< figure src="/images/units-of-work/zoom.svg" link="/images/units-of-work/zoom.svg" alt="Zoom from the sales function to the lead-qualification process to one unit of work: researching an inbound lead. The smallest level is specific enough to change and measure." >}}

AI makes it practical to improve individual tasks. Some need automation; others need a better template, clearer ownership, or removal.

## Find Where Effort Disappears

Dashboards show what the system produced. Follow the work to see where it waited, changed hands, or came back for another pass.

{{< figure src="/images/units-of-work/friction.svg" link="/images/units-of-work/friction.svg" alt="Illustrative support workflow: a ticket arrives, gets triaged and routed, reaches specialist review, and is resolved. Waiting occurs at the specialist handoff, while rework sends the ticket back to triage. Measure these delays as well as active work." >}}

Look for a concrete constraint in each function:

| Function | Unit of work | Friction to investigate |
|---|---|---|
| Sales | Lead qualification | Repeated research |
| Support | Ticket triage | Manual categorization and routing |
| Engineering | Code review | Waiting on a senior approver |
| Finance | Invoice processing | Data entry and reconciliation |
| HR | Candidate screening | Repetitive resume review |

## Choose Where AI Fits

Do not assume every task needs automation. Start by asking whether the task should exist at all.

{{< figure src="/images/units-of-work/ai-fit.svg" link="/images/units-of-work/ai-fit.svg" alt="Task-level decision filter: if a task can be removed, fix the UX or remove the step. If it is necessary, bounded and repetitive, consider automation with checks. Otherwise assist a human, keeping context, judgment, empathy and accountability with people." >}}

For support, this means distinguishing routine tickets from account-specific questions, policy decisions, and engineering investigations. Better product UX may prevent some tickets entirely.

## Define a Measurable Experiment

Name the task, establish a baseline, set a target, and specify what must stay good. Faster work that creates more errors is not an improvement.

{{< figure src="/images/units-of-work/experiment.svg" link="/images/units-of-work/experiment.svg" alt="Example experiment, not a reported result: reduce inbound-lead research from a baseline of twenty minutes to a target of five minutes per lead while preserving qualification quality. Both times share a zero-based minutes scale." >}}

## Audit One Team First

Choose a team with frequent work, visible delays, and meaningful outcomes. Map its recurring tasks before buying tools or launching a transformation program.

{{< figure src="/images/units-of-work/audit.svg" link="/images/units-of-work/audit.svg" alt="Five-step unit-of-work audit: list twenty to fifty recurring weekly tasks; measure frequency, time, waiting, errors and handoffs; tag judgment, context, repetition and risk; pick the top five by effort or delay; run small experiments. An experiment can be a template, decision rule, AI draft, routing change, checklist, or one less step." >}}

Compare minutes per unit, wait time, handoffs, and rework before and after each change. Check quality alongside throughput. Scale what improves useful work, then inspect the next constraint.

Growth targets still set direction. Daily progress comes from removing the bottlenecks that keep people and customers waiting.
