import { api } from "@/lib/api";
import { AppShell } from "@/components/AppShell";
import { StockCard } from "@/components/StockCard";

export default async function HomePage() {
  const start = performance.now();
  const [indices, stocks, popular] = await Promise.all([
    api.indices(),
    api.stocks(),
    api.popularStocks(),
  ]);
  const latencyMs = Math.round(performance.now() - start);

  return (
    <AppShell indices={indices} stocks={stocks}>
      <div className="mx-auto max-w-6xl px-6 py-8">
        <div className="mb-6 flex justify-center">
          <span className="inline-flex items-center gap-2 rounded-full border border-green/30 bg-green/10 px-3 py-1 text-xs font-mono text-green">
            <span className="relative flex h-1.5 w-1.5">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-green opacity-75" />
              <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-green" />
            </span>
            System Operational &middot; {latencyMs}ms latency
          </span>
        </div>

        <div className="flex items-center justify-between mb-1">
          <h1 className="text-lg font-semibold">Algorithmic Radar</h1>
          <span className="rounded-md border border-border bg-surface-2 px-2 py-1 text-xs text-muted">
            Large-cap focus
          </span>
        </div>
        <p className="text-sm text-muted mb-6">
          AI-scored signals across India&apos;s most-traded large-cap stocks.
        </p>

        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
          {popular.map((stock, i) => (
            <StockCard key={stock.ticker} stock={stock} index={i} />
          ))}
        </div>

        <p className="mt-8 text-center text-xs text-muted">
          More stocks, IPOs, and mutual funds are coming soon to the
          intelligence engine.
        </p>
      </div>
    </AppShell>
  );
}
