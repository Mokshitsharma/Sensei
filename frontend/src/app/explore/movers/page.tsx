import Link from "next/link";
import { api } from "@/lib/api";
import { AppShell } from "@/components/AppShell";
import { Change } from "@/components/ui/Badge";
import type { CapSegment } from "@/lib/types";

const SEGMENT_LABELS: Record<CapSegment, string> = {
  large: "Large Cap",
  mid: "Mid Cap",
  small: "Small Cap",
};

export default async function MoversListPage(
  props: PageProps<"/explore/movers">
) {
  const params = await props.searchParams;
  const type = params.type === "losers" ? "losers" : "gainers";
  const segment: CapSegment =
    params.segment === "mid" || params.segment === "small" ? params.segment : "large";

  const [indices, stocks, byCap] = await Promise.all([
    api.indices(),
    api.stocks(),
    api.byCap(segment),
  ]);

  const ranked = [...byCap]
    .filter((r) => (type === "gainers" ? r.expected_move_pct > 0 : r.expected_move_pct < 0))
    .sort((a, b) =>
      type === "gainers"
        ? b.expected_move_pct - a.expected_move_pct
        : a.expected_move_pct - b.expected_move_pct
    );

  return (
    <AppShell indices={indices} stocks={stocks}>
      <div className="mx-auto max-w-6xl px-6 py-8">
        <Link href="/explore" className="text-sm text-muted hover:text-foreground">
          ← Back to Explore
        </Link>

        <h1 className="mt-3 mb-1 text-lg font-semibold">
          Top {type === "gainers" ? "Gainers" : "Losers"} — {SEGMENT_LABELS[segment]}
        </h1>
        <p className="text-sm text-muted mb-6">
          AI 7-day forecast, ranked by expected move.
        </p>

        <div className="rounded-xl border border-border bg-surface divide-y divide-border">
          {ranked.length === 0 && (
            <div className="p-6 text-sm text-muted text-center">
              No {type} in this segment right now.
            </div>
          )}
          {ranked.map((stock) => (
            <Link
              key={stock.ticker}
              href={`/stock/${encodeURIComponent(stock.ticker)}`}
              className="flex items-center justify-between px-5 py-3.5 hover:bg-surface-2 transition-colors"
            >
              <div className="min-w-0">
                <div className="text-sm font-semibold text-foreground truncate">
                  {stock.company}
                </div>
                <div className="text-xs text-muted font-mono">{stock.ticker}</div>
              </div>
              <div className="flex items-center gap-4 shrink-0">
                <span className="font-mono text-sm font-semibold">
                  ₹{stock.current_price.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                </span>
                <Change
                  value={stock.current_price * (stock.expected_move_pct / 100)}
                  pct={stock.expected_move_pct}
                  showValue={false}
                />
              </div>
            </Link>
          ))}
        </div>
      </div>
    </AppShell>
  );
}
