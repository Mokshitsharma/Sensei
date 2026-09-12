"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import { StatTile } from "@/components/ui/StatTile";
import type { AccuracyBreakdown } from "@/lib/types";

const HORIZON_LABEL: Record<string, string> = {
  "1d": "1 Day",
  "7d": "7 Days",
  "30d": "1 Month",
};

// Below this many evaluated calls, a horizon's % is noise, not a track
// record — flagged in the UI rather than presented at face value.
const MIN_SAMPLE = 20;

function tileColor(pct: number | null): "green" | "red" | "default" {
  if (pct === null) return "default";
  return pct >= 50 ? "green" : "red";
}

function TrendChart({ trend }: { trend: AccuracyBreakdown["trend"] }) {
  const points = trend.filter((t) => t.cumulative_pct_correct !== null);
  if (points.length < 2) {
    return (
      <div className="flex h-48 items-center justify-center text-sm text-muted">
        Not enough evaluated days yet to plot a trend — check back once more
        predictions resolve.
      </div>
    );
  }

  const W = 600;
  const H = 200;
  const PAD = 24;

  const xFor = (i: number) => PAD + (i / (points.length - 1)) * (W - 2 * PAD);
  const yFor = (pct: number) => PAD + (1 - pct / 100) * (H - 2 * PAD);

  const linePath = points
    .map((p, i) => `${i === 0 ? "M" : "L"} ${xFor(i)} ${yFor(p.cumulative_pct_correct!)}`)
    .join(" ");

  const fiftyY = yFor(50);
  const last = points[points.length - 1];

  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" preserveAspectRatio="none">
        {/* 50% reference line — the "no real signal" baseline for a directional call */}
        <line
          x1={PAD}
          y1={fiftyY}
          x2={W - PAD}
          y2={fiftyY}
          stroke="currentColor"
          className="text-border"
          strokeWidth={1}
          strokeDasharray="4 4"
        />
        <text
          x={W - PAD}
          y={fiftyY - 6}
          textAnchor="end"
          className="fill-muted text-[10px]"
        >
          50% (coin flip)
        </text>

        <path d={linePath} fill="none" className="stroke-accent" strokeWidth={2} />

        {points.map((p, i) => (
          <circle
            key={p.date}
            cx={xFor(i)}
            cy={yFor(p.cumulative_pct_correct!)}
            r={2.5}
            className="fill-accent"
          />
        ))}

        <text x={PAD} y={H - 6} className="fill-muted text-[10px]">
          {points[0].date}
        </text>
        <text x={W - PAD} y={H - 6} textAnchor="end" className="fill-muted text-[10px]">
          {last.date}
        </text>
      </svg>
      <div className="mt-2 text-center text-xs text-muted">
        Cumulative directional accuracy over time ({last.cumulative_total} calls
        evaluated as of {last.date})
      </div>
    </div>
  );
}

export function AccuracyBreakdownView() {
  const [data, setData] = useState<AccuracyBreakdown | null | "error">(null);

  useEffect(() => {
    api
      .predictionsAccuracyBreakdown()
      .then(setData)
      .catch(() => setData("error"));
  }, []);

  if (data === "error") {
    return (
      <div className="rounded-xl border border-border bg-surface p-6 text-sm text-muted text-center">
        Couldn't load the accuracy breakdown right now.
      </div>
    );
  }

  if (data === null) {
    return <div className="h-64 rounded-xl border border-border bg-surface-2 animate-pulse" />;
  }

  const overallTotal = data.by_horizon.reduce((s, h) => s + h.total, 0);
  const overallCorrect = data.by_horizon.reduce((s, h) => s + h.correct, 0);
  const overallPending = data.by_horizon.reduce((s, h) => s + h.pending, 0);
  const overallPct = overallTotal ? (overallCorrect / overallTotal) * 100 : null;

  return (
    <div className="space-y-6">
      <div className="rounded-xl border border-border bg-surface p-5">
        <div className="mb-4 flex items-baseline justify-between">
          <h2 className="text-sm font-semibold text-foreground">Overall Track Record</h2>
          <span className="text-xs text-muted">{overallPending} calls still pending</span>
        </div>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          <StatTile
            label={`Overall (${overallTotal} calls)`}
            value={overallPct === null ? "—" : `${overallPct.toFixed(1)}%`}
            color={tileColor(overallPct)}
          />
          {data.by_horizon.map((h) => (
            <StatTile
              key={h.horizon}
              label={`${HORIZON_LABEL[h.horizon] ?? h.horizon} (${h.total} calls)`}
              value={h.pct_correct === null ? "—" : `${h.pct_correct.toFixed(1)}%`}
              color={h.total < MIN_SAMPLE ? "amber" : tileColor(h.pct_correct)}
            />
          ))}
        </div>
        {data.by_horizon.some((h) => h.total > 0 && h.total < MIN_SAMPLE) && (
          <p className="mt-3 text-xs text-amber">
            Amber = fewer than {MIN_SAMPLE} evaluated calls — not enough
            samples yet to treat that number as a real track record.
          </p>
        )}
      </div>

      <div className="rounded-xl border border-border bg-surface p-5">
        <h2 className="mb-4 text-sm font-semibold text-foreground">Accuracy Over Time</h2>
        <TrendChart trend={data.trend} />
      </div>
    </div>
  );
}
