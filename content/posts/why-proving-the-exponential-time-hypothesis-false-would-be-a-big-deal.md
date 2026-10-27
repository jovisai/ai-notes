---
title: "Why Proving the Exponential Time Hypothesis False Would Be Such a Big Deal"
date: 2026-10-27
description: "OpenAI says its system found a subexponential algorithm for SAT, challenging a widely used assumption behind computational lower bounds. Here is what that would change, and what it would not."
tags: ["computer-science", "algorithms", "ai-research", "complexity-theory"]
math: true
---

OpenAI says an AI system has found a proof that the Exponential Time Hypothesis (ETH) is false. If the proof holds up, it would overturn one of theoretical computer science’s most widely used assumptions about how hard certain problems are.

That is a major claim, but it is not yet a settled result. OpenAI published a natural-language proof and a Lean formalization on October 26, 2026. A group led by Russell Impagliazzo, Ramamohan Paturi, and Ryan Williams is organizing an independent examination. The researchers’ participation does not mean they have endorsed the proof. ([OpenAI announcement](https://openai.com/index/building-scientific-understanding-with-academic-communities/))

## The idea in one picture

ETH is about how quickly the hardest instances of a problem can be solved as the input grows. The picture below compares two example runtimes. It is a teaching illustration: the curve $2^{\sqrt{n}}$ is not the specific runtime claimed in the new work.

{{< diagram src="/images/eth/eth-growth-rates.html" title="Exponential and subexponential growth" height="840" caption="The exponent of a subexponential runtime grows more slowly than a linear exponent. The curve shown is illustrative, not the algorithm’s bound." >}}

For a formula with $n$ Boolean variables, brute force tries up to $2^n$ assignments. ETH says there is no algorithm that solves every 3-SAT instance in subexponential time, meaning time $2^{o(n)}$. It does not say a computer must try every assignment, or that the best algorithm must take exactly $2^n$ steps. It says no clever method can make the worst-case runtime grow slower than every fixed exponential rate.

OpenAI’s claim is that a deterministic algorithm can solve, and even count the satisfying assignments of, fixed-width Boolean formulas in subexponential time. Its Lean scope document states a formally checked bound of roughly $2^{O_k(n(\log\log n)^2/\sqrt{\log n})}$ for a canonical encoding. The accompanying paper gives a sharper bound. Both are subexponential because their exponent grows slower than $n$; neither gives a polynomial-time guarantee. ([Lean formalization scope](https://github.com/openai/math/blob/main/lean/docs/378.md))

## Why one assumption reaches so far

Researchers use ETH in conditional lower bounds. The proof pattern is: assume SAT cannot be solved subexponentially; show that a much faster algorithm for another problem would also solve SAT too quickly; conclude that the other problem is unlikely to have that faster algorithm, unless ETH is false.

The same starting assumption feeds results across several areas:

- **Graph algorithms:** ETH-based results bound the time needed to compute graph measures such as treewidth and treedepth. These measures help researchers identify graph structure that makes otherwise hard tasks tractable. Recent work, for example, derives near-exponential lower bounds for exact and approximate treedepth algorithms from ETH. ([Bonnet, Neuen, and Sokołowski](https://drops.dagstuhl.de/storage/00lipics/lipics-vol358-ipec2025/html/LIPIcs.IPEC.2025.17/LIPIcs.IPEC.2025.17.html))
- **Computational geometry:** The Earth Mover’s Distance under translation compares two point sets by the cost of matching them after shifting one set. A published result uses ETH to show that its running time cannot improve past a particular dependence on the dimension, unless ETH fails. ([Bringmann et al.](https://doi.org/10.20382/jocg.v16i2a10))
- **Databases and computational biology:** OpenAI notes that hundreds of papers use ETH-derived hardness results in these and other fields. If ETH is refuted, researchers will need to identify which individual results rely on it and whether another assumption still supports them. ([OpenAI announcement](https://openai.com/index/building-scientific-understanding-with-academic-communities/))

{{< diagram src="/images/eth/eth-where-it-matters.html" title="Where ETH-based lower bounds appear" height="840" caption="ETH-based lower bounds connect one assumption about SAT to results in several research areas." >}}

## What would change if the proof is correct?

An ETH-dependent lower bound would lose its stated justification. Researchers would revisit those papers and ask whether the same conclusion can be proved from a weaker assumption, a different reduction, or an unconditional argument. Some results would survive; others might need to be weakened or withdrawn.

That would change what we can confidently say about the limits of algorithms. It could also redirect work toward faster exact algorithms, approximation methods, or new ways to exploit structure in real inputs.

It would **not** mean that every problem covered by an ETH-based lower bound now has a fast algorithm. Refuting a reason that an algorithm cannot be faster is not the same as producing a faster algorithm. Practical performance would still depend on constants, input sizes, implementation, and whether real data resembles the hardest theoretical cases.

It would also not prove $P=NP$. A subexponential algorithm can still take superpolynomial time. ETH concerns a particular family of SAT instances and their asymptotic runtime; $P$ vs. $NP$ asks whether every problem whose solutions can be checked quickly can also be solved quickly. Nor would it automatically break encryption: the ETH claim concerns worst-case SAT, while cryptographic systems usually rely on different average-case hardness assumptions. Nothing in this announcement alone shows that HTTPS, banking systems, or digital signatures are insecure.

## What could this mean for people?

If ETH is false, it would show that at least one widely used picture of computational difficulty is wrong. That could reshape algorithm research. It would not, by itself, deliver new medicines or optimize power grids: a theoretical improvement must be turned into algorithms that work well on useful problem sizes and real data.

| Area | Possible long-term connection |
| --- | --- |
| Medicine | Better algorithms for searching molecular candidates or analyzing biological data |
| Logistics | Faster approaches to routing, scheduling, and supply-chain planning |
| Energy | New ways to optimize power-grid operations and energy use |
| Software | More efficient verification, testing, and optimization of complex systems |
| Research | Better computational tools for exploring scientific models and hypotheses |

These are possible downstream applications of advances in algorithms, not outcomes demonstrated by the ETH result. OpenAI itself says finding better algorithms and turning theory into practical improvements will take substantial further research. ([OpenAI announcement](https://openai.com/index/building-scientific-understanding-with-academic-communities/))

### The larger possibility: AI accelerating discovery

The result also raises a broader question: could AI help create new scientific knowledge, rather than only help researchers work with existing knowledge? In principle, a system could help identify promising questions, explore mathematical approaches, produce a proof, and formalize parts of it in a proof assistant such as Lean. Researchers would still need to interpret the result, check its assumptions, and connect it to applications.

OpenAI’s October 26 announcement describes three claimed results in mathematics, computer science, and physics. They are being handled through different review processes; the ETH work has been released for examination by an external group, and that group’s involvement is not an endorsement. One result, even if confirmed, would be evidence of a capability—not proof that AI can repeatedly make reliable discoveries. Repeated results that withstand independent review would make the case much stronger.

If such a process becomes dependable, advances could compound: better mathematics may lead to better algorithms, which may enable better simulations, which may help researchers investigate chemistry, biology, and physics. That is a possibility, not a consequence established by this one result. It also raises practical questions about who can access these tools, how results are verified, and how the benefits are shared.

| Claim | What we can say now |
| --- | --- |
| ETH has been refuted | OpenAI has released a proof and Lean formalization; independent examination is underway |
| Hard practical problems will become easier | Possible in the long term, but no immediate real-world speedup follows from the claim |
| AI can accelerate fundamental discovery | Potentially the broadest implication if the results are repeatable and independently verified |

## What Lean verification tells us

Lean checks whether a formal proof follows from a precisely stated theorem and the system’s rules. That is powerful evidence about the formalized argument. It does not by itself answer whether the formal theorem captures the standard version of ETH or whether every bridge from the paper’s statement to that theorem is sound. Those are among the questions for independent reviewers.

So the immediate story is not “hard problems are solved.” It is that a foundational assumption may have been overturned, and the evidence is now public for experts to examine. If the proof survives that examination, the biggest near-term consequence will be a rewrite of parts of theoretical computer science. Practical gains may follow later, but they will require new algorithms and engineering.

The result may eventually be remembered as an important development in complexity theory. The larger milestone would be a repeatable process in which AI helps produce discoveries that independent experts can understand, verify, and build on.

## Sources

- [OpenAI: Working with academic communities to build scientific understanding](https://openai.com/index/building-scientific-understanding-with-academic-communities/)
- [OpenAI’s Lean formalization scope for the ETH result](https://github.com/openai/math/blob/main/lean/docs/378.md)
- [Treedepth lower bounds under ETH](https://drops.dagstuhl.de/storage/00lipics/lipics-vol358-ipec2025/html/LIPIcs.IPEC.2025.17/LIPIcs.IPEC.2025.17.html)
- [Earth Mover’s Distance under Translation](https://doi.org/10.20382/jocg.v16i2a10)
