#!/usr/bin/env bash
set -euo pipefail

base_url="https://notes.muthu.co"
llm_file="${1:-static/llm.txt}"
public_dir="${2:-public}"

url_count=0
missing_count=0

while IFS= read -r url; do
    ((url_count += 1))

    path="${url#"$base_url"/}"
    path="${path%/}"
    index_file="$public_dir/$path/index.html"

    if [[ ! -f "$index_file" ]]; then
        printf 'Missing generated page for %s (expected %s)\n' "$url" "$index_file" >&2
        ((missing_count += 1))
    fi
done < <(grep -oE 'https://notes\.muthu\.co/[^)[:space:]]+' "$llm_file" | sort -u)

if ((url_count == 0)); then
    printf 'No internal URLs found in %s\n' "$llm_file" >&2
    exit 1
fi

if ((missing_count > 0)); then
    printf '%d of %d internal URLs in %s do not have generated pages\n' "$missing_count" "$url_count" "$llm_file" >&2
    exit 1
fi

printf 'All %d internal URLs in %s have generated pages\n' "$url_count" "$llm_file"
