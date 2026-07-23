# SEO Re-audit: notes.muthu.co

Audit date: 2026-07-23  
Scope: live production verification after commit `8c834f7`, full 222-URL sitemap check, 50-page metadata/image crawl, representative article review, and cached baseline comparison.

## Executive summary

**SEO Health Score: 83/100, up from 66/100.**

The published technical SEO fixes are live. Crawl access is strong, all sitemap URLs are reachable, canonicals and standard descriptions are consistently present, structured data is deployed, author attribution is visible, and `/llms.txt` is available. Remaining work is concentrated in host-controlled security headers, the `/index.html` redirect, independently measured field performance, and editorial sourcing across the existing archive.

| Category | Prior | Current |
|---|---:|---:|
| Technical SEO | 82 | 86 |
| Content quality | 68 | 72 |
| On-page SEO | 62 | 88 |
| Structured data | 18 | 92 |
| Performance | 74 | 80* |
| AI-search readiness | 49 | 82 |
| Images | 82 | 100 |

*Performance is a deterministic heuristic because PageSpeed/CrUX access was unavailable, not field CWV.

## Verified improvements

- The XML sitemap has **222 unique HTTPS URLs**. All 222 returned HTTP 200 with no redirect, have `lastmod`, and 97 distinct `lastmod` values.
- A 50-page representative crawl found 50/50 title tags, standard meta descriptions, and self-referencing canonicals. Every canonical exactly matched its sitemap URL.
- Every sampled page returned HTTP 200. No missing image alt attributes were found in the sample (20 images).
- Homepage has `Person` and `WebSite` JSON-LD. Articles have `Person` plus `BlogPosting` JSON-LD and a linked author byline.
- `/llms.txt` now returns HTTP 200 (22,971 bytes). `robots.txt` explicitly allows major AI search/user agents and references the sitemap.
- Homepage now exposes “Start here” and topic routes; sampled articles include a “Continue reading” path.
- KaTeX is now loaded only on math pages, reducing unnecessary site-wide JavaScript.

## Remaining actions

| Priority | Issue | Owner / action |
|---|---|---|
| High | Security headers are absent: HSTS, CSP/frame-ancestors, X-Content-Type-Options, Referrer-Policy, and Permissions-Policy. | Configure a CDN or proxy in front of GitHub Pages. |
| Medium | `/index.html` returns 200 rather than redirecting to `/`; tracking parameters also resolve to the same document. Canonicals mitigate this. | Add a host/CDN 301 redirect for `/index.html`; manage parameter handling in Search Console if needed. |
| Medium | Current archive citations remain uneven. The content-generation prompt now requires verified primary sources, but existing evergreen posts need editorial review. | Add verified primary citations to the highest-traffic / evergreen articles first. |
| Low | IndexNow implementation is not discoverable. | Add it if Bing/Yandex discovery speed is a goal. |
| Low | No authenticated PSI, CrUX, Search Console, GA4, or backlink data was available. | Connect measurement sources to validate field CWV, index coverage, organic traffic, and authority. |

## Performance and limitations

The fallback heuristic estimated LCP 1.90s, INP 188ms, CLS 0.030, TBT 131.6ms, and response time 66ms. Treat those as directional lab-style estimates only. PageSpeed API was quota-blocked, and no CrUX field data was available.

## Conclusion

The deployment resolved the original site-controlled SEO issues. The remaining high-priority work requires infrastructure authority, while the remaining content task is deliberate, source-verified editorial improvement rather than bulk automated linking.
