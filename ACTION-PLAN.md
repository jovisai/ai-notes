# SEO Action Plan: notes.muthu.co

## High priority

1. Configure HSTS, CSP/frame-ancestors, X-Content-Type-Options, Referrer-Policy, and Permissions-Policy through a CDN or reverse proxy. GitHub Pages cannot provide these response headers directly.

## Medium priority

1. Add a 301 redirect from `/index.html` to `/` at the host/CDN layer. Self-canonicals already protect the preferred URL.
2. Add verified primary citations to priority evergreen content. Start with high-traffic technical guides and claims about benchmarks, framework behavior, security, and costs.
3. Connect Search Console, CrUX/PageSpeed, and GA4 to replace the current lab-style performance estimate with production data.

## Low priority

1. Implement IndexNow if faster Bing/Yandex discovery is valuable.
2. Re-run the audit after infrastructure headers and redirect rules are deployed.
