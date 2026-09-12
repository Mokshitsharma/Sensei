"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { StatTile } from "@/components/ui/StatTile";
import type { AccuracySummary } from "@/lib/types";

const VISIBLE_COUNT = 5;

const HORIZON_LABEL: Record<string, string> = {
  "1d": "1 Day",
  "7d": "7 Days",
  "30d": "1 Month",
};

export function PredictionAccuracyRow() {
  const [data, setData] = useState<AccuracySummary | null | "error">(null);
  const [showAll, setShowAll] = useState(false);

  useEffect(() => {
    api
      .predictionsAccuracy()
      .then(setData)
      .catch(() => setData("error"));
  }, []);

  if (data === "error") return null;

  return (
    <div className="mb-8">
      <h2 className="text-sm font-semibold text-foreground mb-3">
        Prediction Track Record
      </h2>

      {data === null && (
        <div className="h-32 rounded-xl border border-border bg-surface-2 animate-pulse" />
      )}

      {data !== null && data.rows.length === 0 && (
        <div className="rounded-xl border border-border bg-surface p-6 text-sm text-muted text-center">
          {data.pending > 0
            ? `${data.pending} prediction${data.pending === 1 ? "" : "s"} logged and waiting on their target date — check back once one arrives.`
            : "No predictions to evaluate yet — check back after the next trading day."}
        </div>
      )}

      {data !== null && data.rows.length > 0 && (
        <div className="rounded-xl border border-border bg-surface overflow-hidden">
          {data.pct_correct !== null && (
            <div className="p-4 border-b border-border">
              <div className="max-w-[180px]">
                <StatTile
                  label={`Correct (${data.total} calls)`}
                  value={`${data.pct_correct.toFixed(0)}%`}
                  color={data.pct_correct >= 50 ? "green" : "red"}
                />
              </div>
            </div>
          )}
          <div className="divide-y divide-border">
            {(showAll ? data.rows : data.rows.slice(0, VISIBLE_COUNT)).map((row, i) => (
              <Link
                key={`${row.ticker}-${row.horizon}-${row.target_date}-${i}`}
                href={`/stock/${encodeURIComponent(row.ticker)}`}
                className="flex items-center justify-between px-5 py-3.5 hover:bg-surface-2 transition-colors text-sm"
              >
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="font-semibold text-foreground truncate">{row.company}</span>
                    <span className="shrink-0 rounded border border-border px-1.5 py-0.5 text-[10px] text-muted uppercase">
                      {HORIZON_LABEL[row.horizon] ?? row.horizon}
                    </span>
                  </div>
                  <div className="text-xs text-muted font-mono">
                    {row.ticker} · called {row.predicted_at} for {row.target_date}
                  </div>
                </div>
                <div className="flex items-center gap-5 font-mono text-xs shrink-0">
                  <div className="text-right">
                    <div className="text-muted">Called</div>
                    <div className="text-foreground">
                      ₹{row.price_at_prediction.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-muted">Predicted</div>
                    <div className="text-foreground">
                      ₹{row.predicted_price.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-muted">Actual</div>
                    <div className="text-foreground">
                      ₹{row.actual_price.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                    </div>
                  </div>
                  <span
                    className={`text-base ${row.correct ? "text-green" : "text-red"}`}
                    title={row.correct ? "Correct direction" : "Wrong direction"}
                  >
                    {row.correct ? "✓" : "✗"}
                  </span>
                </div>
              </Link>
            ))}
          </div>
          {data.rows.length > VISIBLE_COUNT && (
            <button
              onClick={() => setShowAll((v) => !v)}
              className="w-full border-t border-border px-5 py-3 text-sm font-medium text-accent hover:bg-surface-2 transition-colors"
            >
              {showAll ? "Show Less" : `View All (${data.rows.length})`}
            </button>
          )}
        </div>
      )}
    </div>
  );
}
