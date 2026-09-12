"use client";

import { useMemo, useRef, useState } from "react";
import { api } from "@/lib/api";
import { Badge } from "@/components/ui/Badge";
import { StatTile, StatGrid } from "@/components/ui/StatTile";
import { ShapChart } from "@/components/ShapChart";
import type { Outlook, OutlookHorizon, StockListItem } from "@/lib/types";

const HORIZONS: { label: string; value: OutlookHorizon }[] = [
  { label: "Tomorrow", value: "1d" },
  { label: "1 Week", value: "7d" },
  { label: "1 Month", value: "30d" },
  { label: "3 Months", value: "90d" },
  { label: "6 Months", value: "180d" },
  { label: "1 Year", value: "365d" },
  { label: "2 Years", value: "730d" },
];

const ACTION_TONE = { BUY: "green", SELL: "red", HOLD: "amber" } as const;

function StockPicker({
  stocks,
  onSelect,
}: {
  stocks: StockListItem[];
  onSelect: (s: StockListItem) => void;
}) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);

  const results = useMemo(() => {
    if (!query.trim()) return [];
    const q = query.toLowerCase();
    return stocks
      .filter((s) => s.name.toLowerCase().includes(q) || s.ticker.toLowerCase().includes(q))
      .slice(0, 8);
  }, [query, stocks]);

  return (
    <div className="relative w-full max-w-md">
      <input
        value={query}
        onChange={(e) => {
          setQuery(e.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
        placeholder="Search a stock to predict…"
        className="w-full rounded-lg border border-border bg-surface-2 px-4 py-2.5 text-sm outline-none placeholder:text-muted"
      />
      {open && results.length > 0 && (
        <div className="absolute z-20 mt-1 w-full rounded-lg border border-border bg-surface shadow-xl overflow-hidden">
          {results.map((s) => (
            <button
              key={s.ticker}
              onMouseDown={() => {
                onSelect(s);
                setQuery(s.name);
                setOpen(false);
              }}
              className="flex w-full items-center justify-between px-3 py-2 text-left text-sm hover:bg-surface-2"
            >
              <span>{s.name}</span>
              <span className="text-muted font-mono text-xs">{s.ticker}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export function PredictStockExplorer({ stocks }: { stocks: StockListItem[] }) {
  const [selected, setSelected] = useState<StockListItem | null>(null);
  const [horizon, setHorizon] = useState<OutlookHorizon>("30d");
  const [outlook, setOutlook] = useState<Outlook | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const requestId = useRef(0);

  function fetchOutlook(ticker: string, h: OutlookHorizon) {
    const id = ++requestId.current;
    setLoading(true);
    setError(null);
    api
      .outlook(ticker, h)
      .then((res) => {
        if (requestId.current === id) setOutlook(res);
      })
      .catch(() => {
        if (requestId.current === id) setError("Couldn't load a prediction for this stock.");
      })
      .finally(() => {
        if (requestId.current === id) setLoading(false);
      });
  }

  function handleSelect(s: StockListItem) {
    setSelected(s);
    fetchOutlook(s.ticker, horizon);
  }

  function handleHorizon(h: OutlookHorizon) {
    setHorizon(h);
    if (selected) fetchOutlook(selected.ticker, h);
  }

  return (
    <div>
      <StockPicker stocks={stocks} onSelect={handleSelect} />

      <div className="mt-4 flex flex-wrap gap-1 rounded-lg border border-border bg-surface-2 p-1 w-fit">
        {HORIZONS.map((h) => (
          <button
            key={h.value}
            onClick={() => handleHorizon(h.value)}
            className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
              horizon === h.value ? "bg-accent text-black" : "text-muted hover:text-foreground"
            }`}
          >
            {h.label}
          </button>
        ))}
      </div>

      {!selected && (
        <p className="mt-8 text-sm text-muted text-center py-10">
          Search for a stock above to see the AI&apos;s outlook.
        </p>
      )}

      {selected && loading && (
        <div className="mt-8 h-64 rounded-xl border border-border bg-surface-2 animate-pulse" />
      )}

      {selected && error && (
        <p className="mt-8 text-sm text-red text-center py-10">{error}</p>
      )}

      {selected && !loading && !error && outlook && (
        <div className="mt-8 space-y-6">
          <div className="flex items-baseline gap-3">
            <h2 className="text-xl font-bold">{outlook.company}</h2>
            <span className="text-xs text-muted font-mono">{outlook.ticker}</span>
            <span className="font-mono text-lg font-semibold">
              ₹{outlook.current_price.toLocaleString(undefined, { maximumFractionDigits: 2 })}
            </span>
          </div>

          {outlook.mode === "quantitative" ? (
            <div className="rounded-xl border border-border bg-surface p-5">
              <p className="text-xs uppercase tracking-wide text-muted font-semibold mb-3">
                Price Target — {outlook.horizon_label}
              </p>
              <StatGrid>
                <StatTile
                  label="Predicted Price"
                  value={`₹${outlook.predicted_price!.toLocaleString(undefined, { maximumFractionDigits: 2 })}`}
                />
                <StatTile
                  label="Expected Move"
                  value={`${outlook.expected_move_pct! >= 0 ? "+" : ""}${outlook.expected_move_pct!.toFixed(2)}%`}
                  color={outlook.expected_move_pct! >= 0 ? "green" : "red"}
                />
                <StatTile label="Direction" value={outlook.direction!} color={outlook.direction === "UP" ? "green" : outlook.direction === "DOWN" ? "red" : "amber"} />
                <StatTile label="Confidence" value={outlook.price_confidence!} />
              </StatGrid>
            </div>
          ) : (
            <div className="rounded-xl border border-amber/30 bg-amber/5 p-4 text-sm text-foreground/90">
              {outlook.disclaimer}
            </div>
          )}

          <div className="rounded-xl border border-border bg-surface p-5">
            <div className="flex items-center justify-between mb-3">
              <p className="text-xs uppercase tracking-wide text-muted font-semibold">
                AI Decision (current)
              </p>
              <Badge tone={ACTION_TONE[outlook.decision.action]}>{outlook.decision.action}</Badge>
            </div>
            {outlook.decision.narrative?.headline && (
              <p className="text-sm text-foreground/90 leading-relaxed">
                {outlook.decision.narrative.headline}
              </p>
            )}
          </div>

          <div className="rounded-xl border border-border bg-surface p-5">
            <p className="text-xs uppercase tracking-wide text-muted font-semibold mb-3">
              What&apos;s Driving This (SHAP)
            </p>
            <ShapChart items={outlook.shap_ranked} />
          </div>

          <div className="rounded-xl border border-border bg-surface p-5">
            <p className="text-xs uppercase tracking-wide text-muted font-semibold mb-3">
              News Sentiment
            </p>
            <StatGrid>
              <StatTile label="Bullish Headlines" value={String(outlook.news_summary.bull_count)} color="green" />
              <StatTile label="Bearish Headlines" value={String(outlook.news_summary.bear_count)} color="red" />
            </StatGrid>
            {outlook.news_summary.top_bullish && (
              <p className="mt-3 text-sm text-foreground/80">
                <span className="text-green">▲</span> {outlook.news_summary.top_bullish.headline}
              </p>
            )}
            {outlook.news_summary.top_bearish && (
              <p className="mt-2 text-sm text-foreground/80">
                <span className="text-red">▼</span> {outlook.news_summary.top_bearish.headline}
              </p>
            )}
          </div>

          <p className="text-[11px] text-muted leading-relaxed">
            Informational only — Sensei AI does not place trades or connect to
            a broker. Predictions are AI-generated estimates, not investment
            advice.
          </p>
        </div>
      )}
    </div>
  );
}
