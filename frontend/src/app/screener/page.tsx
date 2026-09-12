import { api } from "@/lib/api";
import { AppShell } from "@/components/AppShell";
import { ScreenerList } from "@/components/ScreenerList";

export default async function ScreenerPage() {
  const [indices, stocks, screens, initialData] = await Promise.all([
    api.indices(),
    api.stocks(),
    api.screens(),
    api.screener("rsi_oversold"),
  ]);

  return (
    <AppShell indices={indices} stocks={stocks}>
      <div className="mx-auto max-w-6xl px-6 py-8">
        <h1 className="text-lg font-semibold mb-1">Screener</h1>
        <p className="text-sm text-muted mb-6">
          Stocks currently sitting at technical extremes, across momentum,
          trend, and 52-week range screens.
        </p>
        <ScreenerList screens={screens} initialFilter="rsi_oversold" initialData={initialData} />
      </div>
    </AppShell>
  );
}
