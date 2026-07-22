# SEO Action Plan: notes.muthu.co

## Critical / first week

1. **Canonicalize every indexable page** — add an absolute `<link rel="canonical">` in the Hugo base template, generated from `.Permalink`; redirect `/index.html` to `/` at the hosting/CDN layer.
2. **Fix mobile LCP** — conditionally include KaTeX only where math is present, then audit/defer Google Analytics. Re-run Lighthouse and aim for LCP below 2.5 seconds.
3. **Add a reusable schema layer** — publish `BlogPosting` on posts, `Person` + `WebSite` globally, and `ProfilePage` on About. Validate with Google’s Rich Results Test and Schema.org Validator.
4. **Add per-post authorship** — a visible “By Muthu Krishnan” linked to About, with a compact credential such as AI-agent builder / 16+ years in SaaS.

## High / next two weeks

1. Emit unique standard meta descriptions for the homepage, lists, and posts from Hugo front matter.
2. Publish `/llms.txt`; verify it returns HTTP 200 after deployment. Include author bio, sitemap, RSS, topic hubs, and a selection of evergreen guides.
3. Add citations to the top 20 evergreen technical posts, prioritizing factual claims, benchmark values, framework behavior, and security/cost advice.
4. Add a homepage “Start here” section with 3–5 ordered cornerstone guides and two audience/topic routes: AI-agent practitioners and engineering leaders.
5. Add related-content or next-in-series modules to every post.

## Medium / this month

1. Turn important tag clusters into curated hub pages: AI agents, agent architectures, RAG/context, evaluation/reliability, and engineering management.
2. Review the 8 images with missing/empty alt text from the sampled crawl; write descriptive alt text when the image conveys content.
3. Improve small-screen navigation: larger padded targets and left-aligned article text below 900px.
4. Set hardening headers through a suitable CDN/proxy: HSTS, CSP or frame-ancestors, X-Content-Type-Options, Referrer-Policy, and Permissions-Policy.
5. Decide and document AI crawler policy in `robots.txt`; the current permissive policy is valid if broad discovery is intended.

## Verification

- Use `curl -I` to confirm canonicals, `/llms.txt`, and redirect behavior after deployment.
- Run Lighthouse mobile before/after each performance change.
- Validate schema on the homepage, About page, and three different post types.
- Add Search Console and CrUX access to measure indexing, query performance, and field CWV rather than relying only on lab tests.
