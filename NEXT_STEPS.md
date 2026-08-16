# Sensei AI — Next Session TODO

Status as of 2026-08-16: FastAPI backend (`api/`) and Next.js frontend
(`frontend/`) are built, tested locally, and pushed to `main`. The
Streamlit app (`app.py`) still works too and can stay as a fallback demo.
Nothing is deployed yet — everything below is what's left to actually
ship and grow this.

## 1. Deploy (the immediate next step)

- [ ] Pick a backend host — **Render** or **Hugging Face Spaces** (both
      have a free tier and build the `Dockerfile` server-side). Render is
      simpler to set up; HF Spaces is purpose-built for ML workloads like
      this one. Neither has been tried yet — the `Dockerfile` itself was
      never build-tested locally (no Docker installed on the dev machine).
- [ ] Deploy the backend, get its public URL.
- [ ] Deploy `frontend/` to Vercel (connect the GitHub repo).
- [ ] Set env vars on Vercel: `NEXT_PUBLIC_API_URL` (the backend's real
      URL) and `NEXT_PUBLIC_SITE_URL` (the Vercel URL, or a custom domain).
      See `frontend/.env.example` for what's needed.
- [ ] Set `ALLOWED_ORIGINS` on the backend to the real Vercel URL once
      known (currently defaults to `*`, open to any origin — fine for
      testing, should be locked down once the frontend domain is fixed).
- [ ] Smoke-test the deployed site end to end: home page loads, search
      works, a stock detail page loads (first load will be slow — cold
      ML pipeline — confirm the pre-warm loop kicks in afterward).

## 2. IPO section (discussed, not built)

Reference: ipogyani.com's structure (GMP tracking, AI listing-gain
predictions, subscription data, mainboard/SME calendar).

- [ ] IPO calendar: Upcoming / Open / Listed tabs, mainboard + SME split
- [ ] Per-IPO detail page: price band, lot size, issue size, key dates,
      subscription multiple by category (QIB/NII/Retail)
- [ ] Original AI listing-gain model — train on subscription data +
      fundamentals + sector comps (reuse the `decision_engine` ensemble
      pattern already built for stocks). This is the higher-value move
      over scraping unofficial GMP data, and keeps the "Sensei AI"
      branding honest (an original model, not a GMP aggregator clone).
- [ ] Educational content: how to read subscription multiples, anchor
      investor quality, why GMP alone is a weak signal.

## 3. Broader stock coverage

Currently: full NIFTY 50 list is searchable, but only 8 stocks are
pre-warmed/featured as cards on the home page. Consider expanding
featured cards, or making the "Popular Stocks" set dynamic (e.g. most
searched, or biggest movers today) instead of a hardcoded first-8 list.

## 4. Known technical debt / risks (not blockers, but revisit)

- **Data sources are unofficial**: `yfinance` and Google News RSS are
  unauthenticated scraping, not official APIs. They've already been
  flaky more than once this project. At real traffic they can get
  rate-limited or blocked — no fix planned yet, just something to watch.
- **News sentiment is a keyword heuristic**, not a real ML model —
  `transformers`/FinBERT was never installed (the code has a fallback
  path that always triggers). Fine for now; don't oversell it as
  sophisticated AI in any copy/marketing.
- **In-memory cache + rate limiter are per-process** (`api/cache.py`,
  `api/rate_limit.py`). Fine for a single backend instance. If this
  ever scales to multiple instances/workers, both need to move to a
  shared store (Redis) or they'll behave inconsistently per-instance.
- **No automated tests** anywhere (backend or frontend). Worth adding
  before making bigger changes, especially to the ML pipeline glue code.
- **No analytics** — no way to see real usage yet (Vercel Analytics,
  Plausible, or PostHog are all reasonable lightweight options).

## 5. Model quality (the Colab idea)

Retraining the LSTM/TCN/PPO models with GPU access (e.g. via Colab) is
a legitimate way to improve prediction quality, but it's a separate,
longer-effort track from shipping — the current models already work
end-to-end. Don't block deployment on this; revisit once the app is
live and there's a sense of what's actually worth improving.

## 6. Design gaps

Only the Stock Detail → Technicals tab was actually designed in Stitch;
Overview, News, Trade Setup tabs and the AI Signal Panel were built to
match that visual system by extrapolation, not from their own mockups.
If pixel-perfect consistency matters, get those specific screens
designed and reconcile.

## 7. Mobile verification

Responsive behavior was fixed based on code review (Tailwind breakpoints,
flex/grid ordering) since the local browser-automation tooling couldn't
reliably produce a true narrow-viewport screenshot this session. Worth
an actual phone/DevTools-device-mode check once deployed.
