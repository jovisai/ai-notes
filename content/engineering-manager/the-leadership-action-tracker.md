---
title: "Build a Leadership Action Tracker with Five Fields"
date: "2026-09-10"
tags: ["engineering-management"]
---

Leadership teams rarely need another project-management system. They need a simple way to make commitments visible, keep ownership clear, and focus attention where help is required.

That is the purpose of a leadership action tracker.

The tracker should have five fields only:

> **Action | Owner | Due | Status | Blocker / Next move**

Anything beyond these fields starts pushing the tool toward project management rather than commitment management.

Research supports this degree of simplicity. Implementation-intention research shows that specifying when, where, and how an action will happen improves goal attainment; a meta-analysis of 94 tests reported an effect size of *d* = .65. [1] Progress-monitoring research likewise found improved goal attainment across 138 studies and 19,951 participants, with stronger effects when progress was explicitly recorded or visible to others. [2] Goal-setting research also consistently finds that specific goals outperform vague instructions such as “do your best.” [3]

## The Minimum Viable Tracker

| Field | Purpose | Principle |
|---|---|---|
| **Action** | Defines the observable thing that must happen | Goal specificity and next-action clarity |
| **Owner** | Creates one point of accountability | Accountability |
| **Due** | Converts intention into a time-bound commitment | Implementation intentions |
| **Status** | Makes progress visible without requiring discussion | Progress monitoring |
| **Blocker / Next move** | Explains what requires attention or what happens next | Planning and externalizing unfinished work |

The final field is especially important. There should not be separate columns for blockers, notes, updates, dependencies, next steps, risks, and comments.

One short field answers the practical leadership question:

> **What is stopping this, or what happens next?**

That is enough.

Research on unfulfilled goals also suggests that unfinished commitments can continue consuming cognitive resources. Making a concrete plan can reduce that interference by externalizing the work and clarifying the next step. [4]

## A Practical Leadership Action Tracker

Use four active statuses:

**On Track · At Risk · Blocked · Overdue**

Completion does not need to remain as a permanent status. Once an action is complete, remove it from the active tracker and archive it if historical records are useful.

| Action | Owner | Due | Status | Blocker / Next move |
|---|---|---:|---|---|
| Get Security approval for production use of Claude Code | Ravi | Sep 15 | **On Track** | Security review scheduled Sep 12 |
| Reduce API P95 latency below 500 ms by deploying the caching change | Priya | Sep 13 | **At Risk** | Load test fails above 5K RPS; decide whether to reduce initial scope |
| Complete Salesforce migration after Finance confirms field mappings | Arun | Sep 12 | **Blocked** | Waiting for Finance mapping approval |
| Send revised Q4 hiring plan to leadership | Meera | Sep 8 | **Overdue** | Revised plan to be sent today |

That is effectively the whole system.

## Write Actions as Observable Outcomes

The most important field is **Action**, so it should be edited aggressively for specificity.

Bad:

> Improve platform reliability

Better:

> Define Q4 reliability targets

Better still:

> Publish Q4 reliability targets for the five Tier-1 services

The test is simple:

> **Could another person objectively determine whether this happened?**

If the answer is no, rewrite the action.

Specific actions provide direction, focus attention, and make performance easier to evaluate than broad intentions. [5]

## What to Leave Out

Do not add Priority, Created Date, Project, Category, Description, % Complete, Comments, Last Updated, Dependencies, Requester, Team, Effort, Start Date, ID, Labels, or Severity.

These may be useful fields in a project-management system. They are unnecessary in a leadership commitment tracker.

In particular, avoid **% Complete**. “70% complete” often tells leadership less than:

> **At Risk: load test failing above 5K RPS**

The tracker’s job is to direct attention, not describe every detail of the underlying project.

## The Operating Rhythm

The tracker works only if the meeting behavior around it remains minimal.

1. **Add commitments immediately.** When a leadership discussion produces a commitment, record it at once. Require one observable action, exactly one owner, and a due date.

2. **Review for five minutes weekly.** Review the tracker at the beginning or end of the leadership meeting in this order: **Overdue → Blocked → At Risk → On Track**.

3. **Discuss exceptions only.** On Track actions receive no airtime unless someone needs a decision. Blocked and At Risk actions should answer: **What decision or help is needed?**

4. **Remove completed actions.** Once the observable action has happened, remove it from the active view. Archive it automatically if history matters.

5. **Recommit explicitly.** If an action remains overdue, do not silently move the date. The team must explicitly **recommit, reassign, redefine, or drop** the action.

That final rule prevents the tracker from becoming a graveyard of commitments.

## Make Status Rules Objective

| Status | Definition |
|---|---|
| **On Track** | The owner expects completion by the committed date |
| **At Risk** | The owner believes the date may be missed without intervention |
| **Blocked** | Progress requires an external decision, dependency, or resource |
| **Overdue** | The due date has passed and the action is incomplete |

**Overdue** should be calculated automatically from the due date rather than selected manually.

Use visual emphasis only for exceptions:

- **Overdue:** red
- **Blocked:** red
- **At Risk:** amber
- **On Track:** no highlight

A leader opening the tracker should immediately see where attention is required.

## The Five-Minute Leadership Review

For a tracker containing 10 to 25 active commitments, the review might look like this:

> **Overdue:** 2  
> **Blocked:** 1  
> **At Risk:** 3  
> **On Track:** 12

The meeting should focus on the six exceptions, not narrate all eighteen actions.

This is where the progress-monitoring evidence becomes operationally useful: progress is more likely to improve when it is recorded, visible, and connected to timely intervention. [2]

The resulting operating model is:

> **Commit → Record → Review exceptions → Intervene → Remove**

That is very different from:

> **Create → Update → Comment → Categorize → Report → Maintain**

The first model is a leadership execution mechanism. The second is another project-management system.

## Start with the Sheet, Not the System

Create a shared sheet with exactly these five columns:

> **Action | Owner | Due | Status | Blocker / Next move**

Then move the current leadership commitments into it. Do not add any other fields until the team has demonstrated a real, recurring need for them.

Simplicity is not a lack of rigor. In this case, it is what keeps attention on the commitments that matter.

## Sources

[1] [Implementation Intentions and Goal Achievement: A Meta-Analysis of Effects and Processes](https://doi.org/10.1016/S0065-2601%2806%2938002-1)

[2] [Does Monitoring Goal Progress Promote Goal Attainment?](https://pubmed.ncbi.nlm.nih.gov/26479070/)

[3] [New Directions in Goal-Setting Theory](https://doi.org/10.1111/j.1467-8721.2006.00449.x)

[4] [Consider It Done! Plan Making Can Eliminate the Cognitive Effects of Unfulfilled Goals](https://pubmed.ncbi.nlm.nih.gov/21688924/)

[5] [Motivate Employee Performance Through Goal Setting](https://onlinelibrary.wiley.com/doi/10.1002/9781119206422.ch9)
