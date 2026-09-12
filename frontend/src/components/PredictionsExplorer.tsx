"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { PredictionStockCard } from "@/components/PredictionStockCard";
import type { MoverPrediction, PredictionHorizon } from "@/lib/types";

const HORIZONS: { label: string; value: PredictionHorizon }[] = [
  { label: "Tomorrow", value: "1d" },
  { label: "Next 7 Days", value: "7d" },
  { label: "This Month", value: "30d" },
];

const DIRECTIONS: { label: string; value: "UP" | "DOWN" | "ALL" }[] = [
  { label: "May Go Up", value: "UP" },
  { label: "May Go Down", value: "DOWN" },
  { label: "All", value: "ALL" },
];

export function PredictionsExplorer({ initialData }: { initialData: MoverPrediction[] }) {
  const [horizon, setHorizon] = useState<PredictionHorizon>("30d");
  const [direction, setDirection] = useState<"UP" | "DOWN" | "ALL">("UP");
  const [data, setData] = useState<MoverPrediction[]>(initialData);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    api
      .predictions(horizon, direction)
      .then((res) => {
        if (!cancelled) setData(res);
      })
      .catch(() => {
        if (!cancelled) setData([]);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [horizon, direction]);

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-3 mb-6">
        <div className="inline-flex rounded-lg border border-border bg-surface-2 p-1">
          {HORIZONS.map((h) => (
            <button
              key={h.value}
              onClick={() => setHorizon(h.value)}
              className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
                horizon === h.value ? "bg-accent text-black" : "text-muted hover:text-foreground"
              }`}
            >
              {h.label}
            </button>
          ))}
        </div>
        <div className="inline-flex rounded-lg border border-border bg-surface-2 p-1">
          {DIRECTIONS.map((d) => (
            <button
              key={d.value}
              onClick={() => setDirection(d.value)}
              className={`rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
                direction === d.value ? "bg-accent text-black" : "text-muted hover:text-foreground"
              }`}
            >
              {d.label}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {loading &&
          Array.from({ length: 8 }).map((_, i) => (
            <div key={i} className="h-32 rounded-xl border border-border bg-surface-2 animate-pulse" />
          ))}
        {!loading && data.length === 0 && (
          <div className="col-span-full text-sm text-muted py-10 text-center">
            No stocks match this filter right now.
          </div>
        )}
        {!loading &&
          data.map((stock) => (
            <PredictionStockCard
              key={stock.ticker}
              ticker={stock.ticker}
              company={stock.company}
              currentPrice={stock.current_price}
              todayChangePct={stock.today_change_pct}
              direction={stock.direction}
              confidence={stock.confidence}
              predictedPrice={stock.predicted_price}
              horizonLabel={stock.horizon_label}
            />
          ))}
      </div>
    </div>
  );
}
