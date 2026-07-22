# SEO Audit: notes.muthu.co

Audit date: 2026-07-23  
Scope: live homepage, robots.txt, XML sitemap, a 50-URL representative crawl, source configuration review, mobile lab audit, and representative content/UX review.

## Executive summary

**SEO Health Score: 66/100 — needs focused improvement.**

The site is a technically clean, server-rendered Hugo publication with 222 crawlable sitemap URLs, useful long-form content, a clear topical niche, HTTPS, and good baseline visual readability. Its constraints are not crawl access; they are weak canonicalization, absent structured data, incomplete conventional metadata, slow mobile lab LCP, and a weak orientation path for new visitors.

Business type: editorial / expert personal blog. No local-business or e-commerce audit was applicable.

| Category | Score | Weight |
|---|---:|---:|
| Technical SEO | 82 | 22% |
| Content quality | 68 | 23% |
| On-page SEO | 62 | 20% |
| Structured data | 18 | 10% |
| Performance | 74 | 10% |
| AI-search readiness | 49 | 10% |
| Images | 82 | 5% |

### Top issues

1. **High — Canonical tags are absent.** `/`, `/index.html`, and `/?utm_source=a` returned byte-identical, indexable HTML. Add absolute self-referencing canonicals and redirect `/index.html` to `/`.
2. **High — Mobile lab LCP is 4.9 seconds.** This is above the 2.5-second good threshold. Lab testing attributes roughly 102 KiB of potentially unused JavaScript, notably analytics and site-wide KaTeX.
3. **High — No JSON-LD entity or article schema.** Homepage, About page, and sampled articles contained no JSON-LD, Microdata, or RDFa entity graph.
4. **High — E-E-A-T and article attribution are understated.** Articles lack a visible byline/author link despite strong author credentials being available on About.
5. **Medium — Indexable pages lack standard meta descriptions.** Open Graph and itemprop descriptions exist, but sampled pages do not emit `meta name="description"`.

### Quick wins

- Publish the already-present `static/llm.txt` source asset; live `/llms.txt` returned 404.
- Add a `BlogPosting` JSON-LD template and global `Person` + `WebSite` schema.
- Add a shared byline that links to the About page and includes a concise credential.
- Load KaTeX only on pages that render mathematics; defer or consent-gate analytics as appropriate.
- Add a homepage “Start here” path and article-level related/next reading modules.

## Technical SEO

### Strengths

- `robots.txt` permits crawling and declares the canonical sitemap.
- The sitemap is valid XML with **222 unique HTTPS URLs**, all with `lastmod`; the full set resolved as HTTP 200 with no redirect hop.
- The site is server-rendered Hugo HTML, so primary content does not depend on JavaScript.
- HTTPS and HTTP/2 are in place. Sitemap URLs are clean and human-readable.

### Findings

| Priority | Finding | Recommended action |
|---|---|---|
| High | No canonical links; duplicate homepage variants are served. | Emit one absolute self-canonical per indexable page; 301 `/index.html` to `/`; keep tracking parameters out of canonicals. |
| Medium | No standard meta description on the sampled 50 pages. | Add a unique 140–160-character `meta name="description"`; derive a fallback from front matter. |
| Medium | HSTS, CSP/frame-ancestors, X-Content-Type-Options, Referrer-Policy, and Permissions-Policy were absent. | Configure these at a CDN/proxy layer, since GitHub Pages does not provide custom response headers. |
| Low | No explicit AI crawler policy or IndexNow implementation was discoverable. | Decide the intended training/search policy and state it in robots.txt; consider IndexNow for Bing/Yandex discovery. |

## Performance

Mobile Lighthouse lab results: **Performance 74**, SEO 92, FCP **3.4 s**, LCP **4.9 s**, CLS **0**, TBT **90 ms**. Server response time was strong in this sample (about 20–53 ms at the edge), so rendering/third-party assets are the larger opportunity.

Prioritize conditional KaTeX loading for posts that use math. Then review the analytics payload, defer non-essential scripts, and re-test. Field CrUX and Search Console metrics were unavailable because Google API credentials/quota were not available; this is a lab result, not a user-population measurement.

## Content and on-page SEO

The content corpus has useful depth: 114 posts plus topical navigation, and the 50-page sample had 100% HTTP 200 status, title presence, and no thin pages (<300 extracted words). Articles have readable headings, dates, code/tables/diagrams, and are accessible in raw HTML.

The limiting factors are trust and discovery signals:

- Add a visible author byline and link it to an enriched author profile on every post.
- Cite primary papers, official documentation, benchmarks, and vendor sources beside factual or technical claims. Two of three detailed article samples had no outbound body citations.
- Make post descriptions distinct in front matter and use them in conventional metadata.
- Create curated topical hubs rather than relying primarily on tag archives and reverse chronology.

## Schema and AI-search readiness

No structured-data implementation was detected. Add:

- `BlogPosting` to each article: headline, description, canonical URL, dates, image, author, and publisher.
- Global `WebSite` plus `Person` (appropriate for this expert-led site); use `ProfilePage` + `Person` on About.
- `CollectionPage`/`Blog` schema for the homepage and key hub pages.

AI search readiness is constrained by the absent schema/entity graph, thin citations, no visible per-article attribution, and no deployed `/llms.txt`. The repository includes `static/llm.txt`, but the production endpoint was 404 during this audit; ensure the deploy artifact includes it. Keep content accessible to crawlers, as it is now.

## UX and search experience

The visual presentation is clean and responsive, with no mobile horizontal overflow observed. However, the homepage is a chronological archive, not a strong response to broad informational searches such as “AI agent engineering.” Add a “Start here” collection of 3–5 cornerstone guides and topic cards for AI agents and engineering management. On articles, add related posts, next-in-series links, and a low-friction subscribe/contact action.

On small screens, navigation wraps to three rows with 14px links and limited target padding. Increase touch-target area and switch the globally justified body copy to left alignment at mobile widths to avoid irregular spacing.

## Images

Across the 50-page representative crawl, 70 images were found and 8 had empty or missing `alt` attributes. Audit and repair those eight attributes; preserve empty alt only for genuinely decorative images. The site’s small local avatar and CSS are not a material transfer-size concern.

## Limitations

- This used a complete sitemap availability check and a representative 50-page metadata/image crawl, rather than an exhaustive semantic review of all 114 posts.
- No authenticated Search Console, GA4, CrUX, backlink, or rank data was available. Authority and AI visibility assessments are evidence-based estimates.
- Lighthouse data is one controlled mobile lab run and may vary by location, cache state, and device.
