import Link from "next/link";
import { SignInButton } from "@clerk/nextjs";
import { auth } from "@clerk/nextjs/server";
import { api } from "@/lib/api";
import { AppShell } from "@/components/AppShell";
import { Change } from "@/components/ui/Badge";
import { StatTile, StatGrid } from "@/components/ui/StatTile";

export default async function PortfolioPage() {
  const { userId, getToken } = await auth();
  const [indices, stocks] = await Promise.all([api.indices(), api.stocks()]);

  if (!userId) {
    return (
      <AppShell indices={indices} stocks={stocks}>
        <div className="mx-auto max-w-md px-6 py-24 text-center">
          <h1 className="text-lg font-semibold mb-2">Sign in for Paper Trading</h1>
          <p className="text-sm text-muted mb-6">
            Track a simulated ₹1,00,000 portfolio and test the AI&apos;s calls
            with fake money — no broker, no real trades.
          </p>
          <SignInButton mode="modal" forceRedirectUrl="/portfolio">
            <button className="rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-black hover:bg-accent-hover">
              Sign in free
            </button>
          </SignInButton>
        </div>
      </AppShell>
    );
  }

  const token = await getToken();
  const [portfolio, history] = await Promise.all([
    api.portfolio(token!),
    api.portfolioHistory(token!),
  ]);

  const positionsValue = portfolio.positions.reduce(
    (sum, p) => sum + p.quantity * p.current_price,
    0
  );
  const totalUnrealized = portfolio.positions.reduce((sum, p) => sum + p.unrealized_pnl, 0);
  const netWorth = portfolio.cash_balance + positionsValue;

  return (
    <AppShell indices={indices} stocks={stocks}>
      <div className="mx-auto max-w-6xl px-6 py-8">
        <h1 className="text-lg font-semibold mb-1">Paper Portfolio</h1>
        <p className="text-sm text-muted mb-6">
          Simulated trading — informational only, no real money involved.
        </p>

        <div className="mb-8">
          <StatGrid>
            <StatTile
              label="Net Worth"
              value={`₹${netWorth.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
            />
            <StatTile
              label="Cash"
              value={`₹${portfolio.cash_balance.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
            />
            <StatTile
              label="Unrealized P&L"
              value={`₹${totalUnrealized.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
              color={totalUnrealized >= 0 ? "green" : "red"}
            />
            <StatTile
              label="Realized P&L"
              value={`₹${portfolio.realized_pnl.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
              color={portfolio.realized_pnl >= 0 ? "green" : "red"}
            />
          </StatGrid>
        </div>

        <h2 className="text-sm font-semibold text-foreground mb-3">Positions</h2>
        <div className="rounded-xl border border-border bg-surface divide-y divide-border mb-8">
          {portfolio.positions.length === 0 && (
            <div className="p-6 text-sm text-muted text-center">
              No open positions — trade from any stock page.
            </div>
          )}
          {portfolio.positions.map((pos) => (
            <Link
              key={pos.ticker}
              href={`/stock/${encodeURIComponent(pos.ticker)}`}
              className="flex items-center justify-between px-5 py-3.5 hover:bg-surface-2 transition-colors"
            >
              <div className="min-w-0">
                <div className="text-sm font-semibold text-foreground truncate">
                  {pos.company}
                </div>
                <div className="text-xs text-muted font-mono">
                  {pos.ticker} · {pos.quantity} @ ₹{pos.avg_entry_price.toFixed(2)}
                </div>
              </div>
              <div className="flex items-center gap-4 shrink-0">
                <span className="font-mono text-sm font-semibold">
                  ₹{pos.current_price.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                </span>
                <Change value={pos.unrealized_pnl} pct={pos.unrealized_pnl_pct} />
              </div>
            </Link>
          ))}
        </div>

        <h2 className="text-sm font-semibold text-foreground mb-3">Trade History</h2>
        <div className="rounded-xl border border-border bg-surface divide-y divide-border">
          {history.length === 0 && (
            <div className="p-6 text-sm text-muted text-center">No trades yet.</div>
          )}
          {history.map((trade) => (
            <div key={trade.id} className="flex items-center justify-between px-5 py-3 text-sm">
              <div className="flex items-center gap-3">
                <span
                  className={`font-mono text-xs font-semibold ${
                    trade.side === "BUY" ? "text-green" : "text-red"
                  }`}
                >
                  {trade.side}
                </span>
                <span className="font-medium">{trade.ticker}</span>
              </div>
              <div className="font-mono text-xs text-muted">
                {trade.quantity} @ ₹{trade.price.toFixed(2)} ·{" "}
                {new Date(trade.executed_at).toLocaleString()}
              </div>
            </div>
          ))}
        </div>
      </div>
    </AppShell>
  );
}
