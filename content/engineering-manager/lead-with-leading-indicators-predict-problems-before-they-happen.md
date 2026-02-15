---
title: "Lead with Leading Indicators to Predict Problems Before They Happen"
date: 2025-10-11
tags: ['engineering-management']
---

Most engineering managers are firefighters. They react to outages, missed deadlines, and team burnout after the damage is done. Elite EMs are meteorologists—they see the storm forming days before it hits and adjust course early.

The difference? They manage with **leading indicators** instead of **lagging indicators**.

## Lagging vs. Leading: The Critical Distinction

**Lagging indicators** tell you what already happened:
- Sprint velocity dropped by 30%
- Production incident occurred
- Engineer quit unexpectedly
- Release slipped two weeks

By the time you see these signals, you're already in crisis mode. You're managing the aftermath, not preventing the problem.

**Leading indicators** tell you what's about to happen:
- PR review time increased from 4 hours to 2 days
- Code churn rate spiked on the authentication service
- Senior engineer stopped contributing to design docs
- On-call handoff notes became terse and incomplete

These signals appear days or weeks before the crisis. They give you time to intervene, adjust, and prevent the bigger problem.

## Why Most Managers Miss Leading Indicators

1. **They're not urgent**: A missed deadline screams for attention. A gradual increase in meeting load whispers.
2. **They require systems**: You can't "notice" leading indicators by feel. You need to deliberately track them.
3. **They demand action on weak signals**: It feels awkward to intervene when nothing is "broken" yet.

But this is precisely what separates good managers from great ones. **Great managers act on trends, not just events.**

## The Leading Indicator Framework

### 1. Team Health Indicators

These predict burnout, attrition, and morale issues:

- **Slack response times**: If your usually-responsive senior engineer is now taking hours to respond, they're likely overwhelmed or disengaged.
- **Calendar density**: When your engineers' calendars go from 30% meetings to 60%+ meetings, their productivity will crater—just not immediately.
- **1:1 postponements**: If someone reschedules your 1:1 twice in a row, it's a yellow flag. Three times? Red flag.
- **After-hours commits**: A sudden increase in weekend/late-night work signals unsustainable pressure.

**Action**: Set up a simple spreadsheet. Weekly, track these metrics for each team member. Look for trends over 3-4 weeks, not single data points.

### 2. Delivery Health Indicators

These predict missed deadlines and quality issues:

- **PR size creep**: When average PR size grows from 200 lines to 800 lines, you're building up integration risk and review fatigue.
- **Work-in-progress (WIP) inflation**: If your team's Jira board shows 3 tickets per person in progress (instead of the usual 1-2), you're context-switching into chaos.
- **Design doc silence**: When engineers stop writing design docs or getting feedback on them, you're about to ship something that needs to be rewritten.
- **Test coverage delta**: If coverage is dropping 1% per sprint, you're 6 months from a quality crisis.

**Action**: Add these to your weekly team dashboard. Set thresholds (e.g., "average PR > 500 lines triggers review") and create lightweight review rituals.

### 3. System Health Indicators

These predict outages and tech debt explosions:

- **Deploy frequency drop**: If you used to deploy 5x/week and now deploy 2x/week, your system is getting harder to change safely.
- **Time-to-recovery trend**: When the average incident resolution time goes from 30 minutes to 2 hours, your system complexity is outpacing your tooling.
- **Hotspot files**: If the same 3 files appear in 80% of PRs, they're becoming a bottleneck and future bug magnet.
- **Dependency update lag**: When you're 10+ versions behind on a critical library, you're accruing security and upgrade risk.

**Action**: Automate these with your CI/CD and observability tools. Create a monthly "system health review" where you examine trends, not just current state.

## How to Start Leading with Leading Indicators

### Week 1: Identify Your Top 3
Pick the three leading indicators most relevant to your current context. If you're scaling rapidly, focus on team health. If you're shipping a critical feature, focus on delivery health.

Don't try to track everything. Start small and build the habit.

### Week 2: Build a Simple Dashboard
Create a Google Sheet or Notion page. Set up a recurring 15-minute Friday ritual to update it. Keep it simple: just the metric, the trend (↑ ↓ →), and a one-sentence note.

Example:
```
Week of 10/7:
- Avg PR review time: 18h ↑ (was 8h last week) - Need to investigate
- WIP per person: 1.4 → (stable)
- After-hours commits: 12 ↑ (was 5) - Check sprint load
```

### Week 3: Set Trigger Points
Define thresholds that warrant action. For example:
- If PR review time > 24h for 2 consecutive weeks → discuss in retro
- If any individual's after-hours commits > 10/week → 1:1 to check load
- If test coverage drops > 2% in a sprint → make testing a sprint goal

### Week 4: Create Feedback Loops
Share the relevant leading indicators with your team in your weekly sync. Make them part of the conversation:

"I noticed our WIP has been creeping up. Let's talk about what's causing that and how we can reduce context-switching."

This does two things: (1) shows your team you're paying attention to systemic issues, not just outputs, and (2) invites them to help solve problems before they become crises.

## The Compounding Advantage

Here's what changes when you lead with leading indicators:

- **You stop firefighting** and start gardening—tending to problems when they're small and manageable.
- **Your team trusts you more** because you intervene early and thoughtfully, not reactively and frantically.
- **You build organizational resilience** because you're constantly adjusting based on real-time feedback, not post-mortems.
- **You sleep better** because you see problems coming and have time to plan your response.

## The Ultimate Leading Indicator

There's one meta-indicator that predicts almost everything else: **the quality of questions your team asks**.

When engineers ask thoughtful questions about trade-offs, edge cases, and long-term consequences—you're in good shape. When questions become shallow, rushed, or non-existent—trouble is brewing.

Pay attention to the questions in design reviews, code reviews, and team meetings. They're the earliest signal of all.

## Start Today

Open a blank document right now. Title it "Leading Indicators - [Your Team]".

Write down:
1. One team health indicator you'll start tracking this week
2. One delivery health indicator you'll start tracking this week
3. One system health indicator you'll start tracking this week

Set a 15-minute recurring Friday meeting with yourself to update it.

In 6 weeks, you'll catch a problem before it becomes a crisis. In 6 months, your team will wonder why other teams are always in firefighting mode while yours runs smoothly.

**Don't manage the present. Manage the future that's already forming.**
