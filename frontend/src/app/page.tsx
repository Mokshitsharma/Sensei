import Link from "next/link";
import { SignInButton } from "@clerk/nextjs";
import { auth } from "@clerk/nextjs/server";
import { api } from "@/lib/api";
import { TickerStrip } from "@/components/TickerStrip";

export default async function LandingPage() {
  const [indices, { userId }] = await Promise.all([
    api.indices().catch(() => ({})),
    auth(),
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
            <>
              <SignInButton mode="modal" forceRedirectUrl="/explore">
                <button className="text-sm font-medium text-muted hover:text-foreground px-3 py-2">
                  Login
                </button>
              </SignInButton>
              <SignInButton mode="modal" forceRedirectUrl="/explore">
                <button className="rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-black hover:bg-accent-hover">
                  Join Pro
                </button>
              </SignInButton>
            </>
          )}
        </div>
      </header>

      <main className="flex-1 flex flex-col items-center justify-center px-6 py-20 text-center">
        <span className="inline-flex items-center gap-2 rounded-full border border-border bg-surface px-3 py-1 text-xs font-mono text-muted mb-8">
          <span className="relative flex h-1.5 w-1.5">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-green opacity-75" />
            <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-green" />
          </span>
          System Operational
        </span>

        <h1 className="text-4xl sm:text-6xl font-bold tracking-tight leading-tight">
          <span className="text-foreground">AI Intelligence for the</span>
          <br />
          <span className="text-accent">Modern Trader.</span>
        </h1>

        <p className="mt-6 max-w-xl text-muted text-base sm:text-lg leading-relaxed">
          Precision forecasting and algorithmic clarity across India&apos;s
          most-traded large-cap stocks. Stop guessing. Start executing.
        </p>

        <div className="mt-10 flex flex-col sm:flex-row items-center gap-4">
          {isSignedIn ? (
            <Link
              href="/explore"
              className="rounded-lg bg-accent px-6 py-3 text-sm font-semibold text-black hover:bg-accent-hover"
            >
              Open Dashboard
            </Link>
          ) : (
            <>
              <SignInButton mode="modal" forceRedirectUrl="/explore">
                <button className="rounded-lg bg-accent px-6 py-3 text-sm font-semibold text-black hover:bg-accent-hover">
                  Start Free Trial
                </button>
              </SignInButton>
              <SignInButton mode="modal" forceRedirectUrl="/explore">
                <button className="flex items-center gap-2 rounded-lg border border-border px-6 py-3 text-sm font-semibold text-foreground hover:bg-surface-2">
                  <span>▷</span> View Live Demo
                </button>
              </SignInButton>
            </>
          )}
        </div>

        <p className="mt-16 text-xs text-muted max-w-md">
          Informational only — Sensei AI does not place trades or connect to
          a broker. Nothing here is investment advice.
        </p>
      </main>
    </div>
  );
}
