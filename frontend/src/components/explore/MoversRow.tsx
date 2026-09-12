"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { PredictionStockCard } from "@/components/PredictionStockCard";
import type { CapSegment, MoverPrediction } from "@/lib/types";

const SEGMENTS: { label: string; value: CapSegment }[] = [
  { label: "Large Cap", value: "large" },
  { label: "Mid Cap", value: "mid" },
  { label: "Small Cap", value: "small" },
];

export function MoversRow({
  title,
  type,
  initialSegment,
  initialData,
}: {
  title: string;
  type: "gainers" | "losers";
  initialSegment: CapSegment;
  initialData: MoverPrediction[];
}) {
  const [segment, setSegment] = useState<CapSegment>(initialSegment);
  const [data, setData] = useState<MoverPrediction[]>(initialData);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (segment === initialSegment) {
      setData(initialData);
      return;
    }
    let cancelled = false;
    setLoading(true);
    api
      .byCap(segment)
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
  }, [segment]);

  const ranked = [...data]
    .filter((r) => (type === "gainers" ? r.expected_move_pct > 0 : r.expected_move_pct < 0))
    .sort((a, b) =>
      type === "gainers"
        ? b.expected_move_pct - a.expected_move_pct
        : a.expected_move_pct - b.expected_move_pct
    );
  const top4 = ranked.slice(0, 4);

  return (
    <div className="mb-8">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-semibold text-foreground">{title}</h2>
        <div className="inline-flex rounded-lg border border-border bg-surface-2 p-1">
          {SEGMENTS.map((s) => (
            <button
              key={s.value}
              onClick={() => setSegment(s.value)}
              className={`rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
                segment === s.value
                  ? "bg-accent text-black"
                  : "text-muted hover:text-foreground"
              }`}
            >
              {s.label}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 xl:grid-cols-5 gap-4">
        {loading &&
          Array.from({ length: 4 }).map((_, i) => (
            <div key={i} className="h-40 rounded-xl border border-border bg-surface-2 animate-pulse" />
          ))}
        {!loading &&
          top4.map((stock) => (
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
        {!loading && ranked.length === 0 && (
          <div className="col-span-full text-sm text-muted py-6 text-center">
            No {type} in this segment right now.
          </div>
        )}
        {!loading && ranked.length > 0 && (
          <Link
            href={`/explore/movers?type=${type}&segment=${segment}`}
            className="flex flex-col items-center justify-center rounded-xl border border-dashed border-border bg-surface-2 p-4 text-sm font-medium text-muted hover:text-accent hover:border-accent/50 transition-colors"
          >
            View All →
          </Link>
        )}
      </div>
    </div>
  );
}
