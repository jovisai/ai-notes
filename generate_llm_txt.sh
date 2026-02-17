#!/bin/bash
# Generates static/llm.txt from content front matter
set -e

OUTPUT="static/llm.txt"
BASE_URL="https://notes.muthu.co"

cat > "$OUTPUT" << 'HEADER'
# Engineering Notes

> Thoughts and Ideas on AI by Muthukrishnan

This blog by Muthu Krishnan covers AI agents, LLM technologies, multi-agent systems, and engineering management. Muthu is an AI Agents builder at Sanas.ai with 16+ years of experience building scalable SaaS applications. He is the author of "Essential Search Algorithms: Navigating the Digital Maze."

## Sections

- [All Posts](https://notes.muthu.co/posts): General articles on AI, coding agents, and software engineering
- [AI Agents](https://notes.muthu.co/ai-agents): Deep-dive series on AI agent concepts, algorithms, and architectures
- [Engineering Manager](https://notes.muthu.co/engineering-manager): Leadership, team management, and organizational design for engineering managers
- [About](https://notes.muthu.co/about): About the author
- [Resume](https://notes.muthu.co/resume): Professional background
HEADER

generate_section() {
    local dir="$1"
    local header="$2"
    local section_name=$(basename "$dir")

    echo "" >> "$OUTPUT"
    echo "## $header" >> "$OUTPUT"
    echo "" >> "$OUTPUT"

    for f in "$dir"/*.md; do
        local basename_f=$(basename "$f" .md)
        [[ "$basename_f" == "_index" ]] && continue

        local title=$(grep -m1 '^title:' "$f" | sed "s/^title: *//;s/^[\"']//;s/[\"']$//")
        local desc=$(grep -m1 '^description:' "$f" | sed "s/^description: *//;s/^[\"']//;s/[\"']$//")
        local date=$(grep -m1 '^date:' "$f" | sed 's/^date: *//' | cut -c1-7)

        # Extract year and month for URL
        local year=$(echo "$date" | cut -d- -f1)
        local month=$(echo "$date" | cut -d- -f2)

        if [[ -n "$title" && -n "$year" && -n "$month" ]]; then
            local url="${BASE_URL}/${year}/${month}/${basename_f}"
            if [[ -n "$desc" ]]; then
                echo "- [${title}](${url}): ${desc}" >> "$OUTPUT"
            else
                echo "- [${title}](${url})" >> "$OUTPUT"
            fi
        fi
    done
}

generate_section "content/posts" "Posts"
generate_section "content/ai-agents" "AI Agents Series"
generate_section "content/engineering-manager" "Engineering Manager Series"

echo "Generated $OUTPUT"
