"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { Badge } from "@/components/ui/Badge";
import type { CapSegment, ScreenerRow, ScreenFilter, ScreenInfo } from "@/lib/types";

const SEGMENTS: { label: string; value: CapSegment | "all" }[] = [
  { label: "All", value: "all" },
  { label: "Large Cap", value: "large" },
  { label: "Mid Cap", value: "mid" },
  { label: "Small Cap", value: "small" },
];

export function ScreenerList({
  screens,
  initialFilter,
  initialData,
}: {
  screens: ScreenInfo[];
  initialFilter: ScreenFilter;
  initialData: ScreenerRow[];
}) {
  const [filter, setFilter] = useState<ScreenFilter>(initialFilter);
  const [segment, setSegment] = useState<CapSegment | "all">("all");
  const [data, setData] = useState<ScreenerRow[]>(initialData);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (filter === initialFilter && segment === "all") {
      setData(initialData);
      return;
    }
    let cancelled = false;
    setLoading(true);
    api
      .screener(filter, segment)
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
  }, [filter, segment]);

  const activeScreen = screens.find((s) => s.id === filter);

  return (
    <div>
      <h2 className="text-sm font-semibold text-foreground mb-3">Trading Screens</h2>
      <div className="rounded-xl border border-border bg-surface divide-y divide-border mb-6">
        {screens.map((screen) => (
          <button
            key={screen.id}
            onClick={() => setFilter(screen.id)}
            className={`flex w-full items-center justify-between px-5 py-3.5 text-left transition-colors ${
              filter === screen.id ? "bg-surface-2" : "hover:bg-surface-2"
            }`}
          >
            <div className="flex items-center gap-3">
              <Badge tone={screen.bias === "Bullish" ? "green" : "red"}>{screen.bias}</Badge>
              <span className="text-sm font-medium text-foreground">{screen.label}</span>
            </div>
            {filter === screen.id && <span className="text-accent text-xs">Selected</span>}
          </button>
        ))}
      </div>

      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-semibold text-foreground">
          {activeScreen?.label ?? "Results"}
        </h3>
        <div className="inline-flex rounded-lg border border-border bg-surface-2 p-1">
          {SEGMENTS.map((s) => (
            <button
              key={s.value}
              onClick={() => setSegment(s.value)}
              className={`rounded-md px-2.5 py-1 text-xs font-medium transition-colors ${
                segment === s.value ? "bg-accent text-black" : "text-muted hover:text-foreground"
              }`}
            >
              {s.label}
            </button>
          ))}
        </div>
      </div>

      <div className="rounded-xl border border-border bg-surface divide-y divide-border">
        {loading && (
          <div className="p-6 text-sm text-muted text-center">Loading…</div>
        )}
        {!loading && data.length === 0 && (
          <div className="p-6 text-sm text-muted text-center">
            No stocks match this screen right now.
          </div>
        )}
        {!loading &&
          data.map((row) => (
            <Link
              key={row.ticker}
              href={`/stock/${encodeURIComponent(row.ticker)}`}
              className="flex items-center justify-between px-5 py-3.5 hover:bg-surface-2 transition-colors"
            >
              <div className="min-w-0">
                <div className="text-sm font-semibold text-foreground truncate">
                  {row.company}
                </div>
                <div className="text-xs text-muted font-mono">{row.ticker}</div>
              </div>
              <div className="flex items-center gap-4 shrink-0">
                <span className="font-mono text-sm font-semibold">
                  ₹{row.price.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                </span>
                <Badge tone={activeScreen?.bias === "Bullish" ? "green" : "red"}>
                  RSI {row.rsi.toFixed(1)}
                </Badge>
              </div>
            </Link>
          ))}
      </div>
    </div>
  );
}
