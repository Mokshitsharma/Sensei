import { api } from "@/lib/api";
import { AppShell } from "@/components/AppShell";
import { RecentlyVisitedRow } from "@/components/explore/RecentlyVisitedRow";
import { MoversRow } from "@/components/explore/MoversRow";
import { PredictionAccuracyRow } from "@/components/explore/PredictionAccuracyRow";

export default async function ExplorePage() {
  const start = performance.now();
  const [indices, stocks, byCapLarge] = await Promise.all([
    api.indices(),
    api.stocks(),
    api.byCap("large"),
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

        <h1 className="text-lg font-semibold mb-1">Explore</h1>
        <p className="text-sm text-muted mb-6">
          AI-scored signals across India&apos;s most-traded stocks.
        </p>

        <RecentlyVisitedRow />

        <MoversRow
          title="Top Gainers Today"
          type="gainers"
          initialSegment="large"
          initialData={byCapLarge}
        />

        <MoversRow
          title="Top Losers Today"
          type="losers"
          initialSegment="large"
          initialData={byCapLarge}
        />

        <PredictionAccuracyRow />
      </div>
    </AppShell>
  );
}
