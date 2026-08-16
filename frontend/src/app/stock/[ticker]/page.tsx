import Link from "next/link";
import { notFound } from "next/navigation";
import type { Metadata } from "next";
import { auth } from "@clerk/nextjs/server";
import { api } from "@/lib/api";
import { AppShell } from "@/components/AppShell";
import { PriceChart } from "@/components/PriceChart";
import { Tabs } from "@/components/ui/Tabs";
import { AiSignalPanel } from "@/components/AiSignalPanel";
import { OverviewTab } from "@/components/OverviewTab";
import { TechnicalsTab } from "@/components/TechnicalsTab";
import { NewsPanel } from "@/components/NewsPanel";
import { TradeSetupTab } from "@/components/TradeSetupTab";

export async function generateMetadata(
  props: PageProps<"/stock/[ticker]">
): Promise<Metadata> {
  const { ticker: rawTicker } = await props.params;
  const ticker = decodeURIComponent(rawTicker);
  try {
    const stocks = await api.stocks();
    const match = stocks.find((s) => s.ticker === ticker);
    if (!match) return { title: "Stock not found | Sensei AI" };
    return {
      title: `${match.name} (${ticker}) | Sensei AI`,
      description: `AI-powered price analysis, technicals, news sentiment, and trade setup for ${match.name}.`,
    };
  } catch {
    return { title: "Sensei AI" };
  }
}

export default async function StockDetailPage(
  props: PageProps<"/stock/[ticker]">
) {
  const { ticker: rawTicker } = await props.params;
  const ticker = decodeURIComponent(rawTicker);

  const [indices, stocks, { userId }] = await Promise.all([
    api.indices(),
    api.stocks(),
    auth(),
  ]);
  const isSignedIn = Boolean(userId);

  if (!stocks.some((s) => s.ticker === ticker)) {
    notFound();
  }

  const [price, fundamentals, analysis, news, backtest] = await Promise.all([
    api.price(ticker),
    api.fundamentals(ticker),
    api.analysis(ticker),
    api.news(ticker),
    api.backtest(ticker),
  ]);

  const priceUp = analysis.signals.regime === "BULL";

  return (
    <AppShell indices={indices} stocks={stocks}>
      <div className="mx-auto max-w-6xl px-6 py-8">
        <Link href="/explore" className="text-sm text-muted hover:text-foreground">
          ← Back to Explore
        </Link>

        <div className="mt-3 mb-6 flex items-baseline gap-3">
          <h1 className="text-2xl font-bold">{analysis.company}</h1>
          <span className="text-xs text-muted font-mono">{ticker}</span>
          <span
            className={`text-xl font-mono font-semibold ${
              priceUp ? "text-green" : "text-red"
            }`}
          >
            ₹{fundamentals.current_price.toLocaleString(undefined, {
              maximumFractionDigits: 2,
            })}
          </span>
        </div>

        <div className="rounded-xl border border-border bg-surface p-2 mb-6">
          <PriceChart records={price.records} />
        </div>

        <div className="grid lg:grid-cols-3 gap-6 items-start">
          <div className="lg:col-span-2 min-w-0">
            <Tabs
              tabs={[
                {
                  label: "Overview",
                  content: (
                    <OverviewTab
                      signals={analysis.signals}
                      fundamentals={fundamentals}
                      backtest={backtest}
                    />
                  ),
                },
                {
                  label: "Technicals",
                  content: (
                    <TechnicalsTab
                      signals={analysis.signals}
                      decision={analysis.decision}
                      supportResistance={analysis.support_resistance}
                      isSignedIn={isSignedIn}
                    />
                  ),
                },
                {
                  label: "News",
                  content: (
                    <NewsPanel news={news} forecast={analysis.news_price_forecast} />
                  ),
                },
                {
                  label: "Trade Setup",
                  content: (
                    <TradeSetupTab
                      intraday={analysis.setup.intraday}
                      swing={analysis.setup.swing}
                      isSignedIn={isSignedIn}
                    />
                  ),
                },
              ]}
            />
          </div>

          <div className="order-first lg:order-last">
            <AiSignalPanel
              decision={analysis.decision}
              swingSetup={analysis.setup.swing}
              isSignedIn={isSignedIn}
            />
          </div>
        </div>
      </div>
    </AppShell>
  );
}
