#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <path-to-article.md>" >&2
  exit 1
fi

ARTICLE="$1"

if [[ ! -f "$ARTICLE" ]]; then
  echo "ERROR: File not found: $ARTICLE" >&2
  exit 1
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Humanizing: $ARTICLE"

/home/leopard/.local/bin/claude --dangerously-skip-permissions --verbose -p "You are an editor improving a technical blog article to sound less AI-generated. Edit the file $ARTICLE in place, making only the changes listed below. Do not rewrite, restructure, or change the technical content.

**Changes to make:**

1. **Em-dashes**: Replace every em-dash (—) with a comma, period, or parentheses — whichever reads most naturally in context. Do not leave any em-dashes.

2. **Opening paragraph**: If the article opens with a metaphor, anecdote, or 'hook' paragraph designed to engage the reader, rewrite just that paragraph to open directly with the technical substance instead.

3. **Key Takeaways / Summary section**: If there is a 'Key Takeaways', 'Summary', or similar section at the end that restates the article's content, delete it entirely.

4. **Sub-section labels**: If sections have sub-headers like '### Simple Explanation' or '### Technical Detail', remove those sub-headers and let the text flow as continuous prose within the parent section.

5. **Pattern labels**: If design patterns are labeled as 'Pattern: X', remove the label and integrate the name naturally into the prose sentence.

6. **Over-bolding**: Remove bold formatting from terms that are not truly critical. Keep bold only on the one or two most important terms per section. Un-bold anything that is bolded merely because it is a technical term appearing for the first time.

7. **Triple parallel lists**: Find sentences that use the 'X, Y, and Z' triple construction repeatedly. Rewrite some of them with varied structure — a dependent clause, a short follow-up sentence, or a simple two-item contrast.

8. **Tidy bow phrases**: Remove or rewrite closing sentences that wrap up a paragraph with phrases like 'exactly the architecture proposed in...', 'this is X in action', 'a design principle worth borrowing', or similar AI-typical wrap-ups. Cut to the point or end the paragraph one sentence earlier.

9. **Sentence rhythm**: If multiple consecutive sentences have similar length and structure, vary them. Add a short punchy sentence or break a long one in two.

Only edit the file if changes are needed. Preserve all markdown structure, frontmatter, code blocks, math, and mermaid diagrams exactly."

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done: $ARTICLE"
