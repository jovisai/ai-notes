---
title: "The Doorman Fallacy and Why AI Keeps Failing at Replacing People"
date: 2026-06-24
tags: ["AI", "Automation", "Software Engineering", "Engineering Management"]
---

In 2023, AI was going to replace programmers. A year later, customer support was next. By 2025, some executives had moved on to talking about whole departments.

By 2026, the story looks messier. Companies are still automating wherever they can, but a lot of them are learning the same lesson the hard way: replacing a visible task is not the same as replacing the job.

That sounds like a small distinction until the automation goes live.

Then it becomes the difference between useful leverage and organizational self-harm.

Rory Sutherland has a name for this mistake: the Doorman Fallacy.

## The assumption behind AI replacement

For the last few years, the pitch has been simple. AI is getting smarter, cheaper, and more capable.

The benchmarks seemed to support it. Higher scores. Longer context windows. More code generated. More tickets closed. More tasks completed.

So the conclusion felt obvious:

> If AI can do the visible part of a job, eventually it can do the whole job.

That works as a slide deck argument. It gets weaker once it touches a real organization.

## The Doorman Fallacy

Sutherland's [Doorman Fallacy](https://www.mfpwealthmanagement.co.uk/blog/doorman-fallacy-rory-sutherland) starts with a luxury hotel.

Management looks at the entrance and sees someone being paid to open doors. On a spreadsheet, the job is easy to summarize:

> The doorman opens the door.

Once you describe the role that way, the fix looks obvious. Install automatic doors. Remove the salary. Save the money.

Except the doorman was never only opening doors.

He was welcoming guests, recognizing regulars, helping with luggage, watching the entrance, setting the tone for the hotel, and quietly telling people what kind of place they had walked into.

Opening the door was the visible task. The value was the cluster of things around it.

When management automates only the visible task, it may save money while damaging the service, trust, and social context that made the role useful in the first place.

That is the fallacy: mistaking the visible action for the whole job.

## How companies make the same mistake with AI

Organizations often evaluate knowledge work the same way. They collapse a role into the easiest output to name.

Customer service becomes answering questions. Programming becomes writing code. Management becomes sending emails. Design becomes making graphics. Recruiting becomes screening resumes. Sales becomes writing follow-up messages.

Then someone asks:

> If AI can do that part, why do we still need people?

The question is understandable. It is also incomplete.

The visible output is usually the easiest part to observe, measure, and automate. That does not make it the most valuable part.

The value often lives in judgment, trust, escalation, taste, coordination, accountability, and memory.

AI can help with some of that work. Help is not the same as replacement.

## Software development is the easiest place to see it

Software development makes the mistake obvious because the output is so easy to misunderstand.

A lot of non-technical leaders think developers are paid to write code. Experienced engineers know that is only partly true.

Writing code is visible. Sometimes it is also the easy part.

The harder work is figuring out what the system should do, what the user actually needs, which trade-offs are acceptable, which old assumptions are hiding in the codebase, and which shortcut will become an incident six months later.

Engineers debug failures that do not match the documentation. They read old code carefully because the dangerous behavior is often implied, not written down. They decide when not to build something. They own the consequences when a change breaks production.

The code is the artifact. The work is engineering judgment.

That is why raw code generation can be impressive and still fall short. A model can produce a function, test, migration, or endpoint quickly. Useful. But producing code is not the same as owning correctness, maintainability, security, observability, and the long-term cost of the change.

Senior engineers tend to see this faster. Not because they are dismissive of AI. Many use it heavily. They are just less likely to confuse output speed with engineering quality.

## Customer support has the same problem

Customer support looks easy to automate when you define it as answering questions.

But support teams do more than produce answers. They detect frustration. They notice product gaps. They spot broken processes. They know when a customer needs an exception instead of a policy quote. They know when a technically correct answer will make the customer angrier.

If a company replaces support with a chatbot and measures only ticket deflection, the early numbers can look great.

Fewer tickets reach humans. Cost per interaction falls. The dashboard improves.

The cost shows up later. Customers trust the company less. Edge cases take longer to resolve. Escalations arrive angrier. Product teams lose feedback from the front line. The remaining support agents inherit only the worst cases.

The company thinks it automated support. It may have automated the first response and damaged the support system.

That is why some customer-service automation is already being [reconsidered](https://www.cmswire.com/customer-experience/the-great-customer-service-ai-rehiring-is-coming/). The problem is not that AI cannot answer questions. Often it can. The problem is that support is not only answering questions.

## Real work has edges

The same pattern shows up outside office work.

McDonald's [ended its IBM-powered drive-thru voice-ordering test in 2024](https://apnews.com/article/mcdonalds-ai-drive-thru-ibm-bebc898363f2d550e1a0cd3c682fa234) after trying it in more than 100 restaurants. The promise was simple: automate order taking.

The real environment was not simple. Accents, background noise, menu changes, interruptions, corrections, impatient customers, and local operating quirks all showed up at once.

The lesson is not that restaurant automation is doomed. It probably is not.

The lesson is that jobs contain hidden context. A task that looks narrow in a demo often expands in production.

The demo says:

> Take the order.

The real job says:

> Understand the order through noise, timing pressure, ambiguity, changing inventory, customer emotion, payment constraints, and exceptions.

That gap is where automation plans break.

## The better question

When people say AI writes code now, answers customer questions now, or generates designs now, they are usually pointing at the visible task.

In many cases, they are right. AI can do that part.

The better question is whether it can replace the value around the task.

That value includes context, judgment, accountability, trust, taste, relationships, exception handling, institutional memory, and long-term ownership.

These things are harder to measure, which makes them easy to ignore.

Spreadsheets reward visible costs. Dashboards reward visible outputs. Automation business cases reward visible savings.

Organizations still run on the work that does not fit neatly into those boxes.

## Define the job before automating it

The answer is not to reject AI. That is just the mirror image of assuming it can replace everyone.

The useful move is to define the work honestly before automating it.

Before replacing a role, leaders should ask:

- What does this person do that is not captured in the job title?
- What exceptions do they handle?
- What decisions do they make that are not written down?
- What trust do they create?
- What problems do they prevent?
- What information flows through them?
- What breaks if only the visible output is automated?

Those questions turn automation from a cost-cutting reflex into a systems problem.

Sometimes the answer will still be automation. Sometimes it will be augmentation. Sometimes the role should be redesigned. And sometimes the supposedly simple job turns out to be holding more of the system together than anyone wanted to admit.

## AI should remove drudgery, not judgment

The best use of AI is not to pretend invisible work does not exist. It is to remove the drudgery around valuable human judgment.

For software teams, that means using AI to draft code, explain old code, create tests, summarize logs, search documentation, and speed up repetitive implementation work. Humans still own architecture, trade-offs, review, and consequences.

For support teams, it means using AI to retrieve knowledge, summarize history, suggest replies, classify issues, and help agents move faster. Humans still handle trust-sensitive, emotional, or high-impact cases.

For managers, it means using AI to summarize information and reduce administrative drag. Humans still own context, prioritization, coaching, and accountability.

The pattern is simple:

Automate the door. Do not forget why the doorman was there.

## The real lesson

AI is not failing in these organizations because it is useless.

It is failing because the organization misunderstood the job.

They saw the output and missed the system around it. They saw the door and missed the welcome. They saw the code and missed the engineering. They saw the answer and missed the trust.

The biggest mistake is assuming a job is the output you can see.

The output is usually the easy part.

The value is everything around it.
