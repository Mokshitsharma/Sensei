import Link from "next/link";
import { SignInButton } from "@clerk/nextjs";
import { auth } from "@clerk/nextjs/server";
import { api } from "@/lib/api";
import { TickerStrip } from "@/components/TickerStrip";
import { BackgroundChart } from "@/components/BackgroundChart";
import { PhoneStory } from "@/components/PhoneStory";
import { FeaturedStockTicker } from "@/components/FeaturedStockTicker";

const FEATURED_TICKER = "SBIN.NS";
const FEATURED_NAME = "SBI";

export default async function LandingPage() {
  const [indices, { userId }, featuredQuote] = await Promise.all([
    api.indices().catch(() => ({})),
    auth(),
    api.quote(FEATURED_TICKER).catch(() => null),
  ]);
  const isSignedIn = Boolean(userId);

  return (
    <div className="min-h-screen flex flex-col bg-background">
      <TickerStrip indices={indices} />

      <header className="flex items-center justify-between px-6 sm:px-10 py-4">
        <span className="text-xl font-bold text-accent">Sensei AI</span>
        <div className="flex items-center gap-3">
          {isSignedIn ? (
            <Link
              href="/explore"
              className="rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-black hover:bg-accent-hover"
            >
              Go to Dashboard
            </Link>
          ) : (
            <SignInButton mode="modal" forceRedirectUrl="/explore">
              <button className="text-sm font-medium text-muted hover:text-foreground px-3 py-2">
                Sign in
              </button>
            </SignInButton>
          )}
        </div>
      </header>

      <main className="relative flex-1 overflow-hidden px-6 py-16 lg:py-20">
        <BackgroundChart />

        <div className="relative z-10 mx-auto flex max-w-6xl flex-col items-center gap-16 lg:flex-row lg:items-center lg:justify-between lg:gap-10">
          <div className="flex flex-col items-center text-center lg:items-start lg:text-left">
            <span className="inline-flex items-center gap-2 rounded-full border border-border bg-surface px-3 py-1 text-xs font-mono text-muted mb-8">
              <span className="relative flex h-1.5 w-1.5">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-green opacity-75" />
                <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-green" />
              </span>
              System Operational
            </span>

            <h1 className="text-4xl sm:text-6xl font-bold tracking-tight leading-tight">
              <span className="text-foreground">See the Market Like an</span>
              <br />
              <span className="text-accent">Analyst. Free. No Login.</span>
            </h1>

            <p className="mt-6 max-w-xl text-muted text-base sm:text-lg leading-relaxed">
              AI-powered price predictions, technical signals, and trade setups
              across 2,400+ NSE stocks. Browse everything free — create an
              account only if you want to save a watchlist.
            </p>

            <FeaturedStockTicker
              initial={{ name: FEATURED_NAME, ticker: FEATURED_TICKER, quote: featuredQuote }}
            />

            <div className="mt-10 flex flex-col sm:flex-row items-center gap-4">
              {/* Browsing is fully free — no sign-in required anywhere. Signing
                  in is entirely optional (e.g. to save a watchlist later). */}
              <Link
                href="/explore"
                className="rounded-lg bg-accent px-6 py-3 text-sm font-semibold text-black hover:bg-accent-hover"
              >
                Browse Stocks →
              </Link>
              {!isSignedIn && (
                <SignInButton mode="modal" forceRedirectUrl="/explore">
                  <button className="text-sm font-medium text-muted hover:text-foreground px-3 py-3">
                    or sign in to save your watchlist
                  </button>
                </SignInButton>
              )}
            </div>

            <p className="mt-16 text-xs text-muted max-w-md">
              Informational only — Sensei AI does not place trades or connect
              to a broker. Nothing here is investment advice.
            </p>
          </div>

          <PhoneStory />
        </div>
      </main>
    </div>
  );
}
