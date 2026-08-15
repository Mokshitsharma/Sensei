import clsx from "clsx";
import type { NarrativeShapItem } from "@/lib/types";

const LABELS: Record<string, string> = {
  rsi_norm: "RSI",
  ema_spread: "EMA Crossover",
  macd_diff: "MACD Divergence",
  atr_pct: "ATR (Volatility)",
  volatility_10: "10-day Volatility",
  return_1: "1-day Return",
  return_5: "5-day Return",
  return_10: "10-day Return",
  range_position: "Price Range Position",
};

export function ShapChart({ items }: { items: NarrativeShapItem[] }) {
  if (!items || items.length === 0) {
    return <p className="text-sm text-muted">SHAP data not available.</p>;
  }

  const max = Math.max(...items.map((i) => Math.abs(i.shap)), 0.001);

  return (
    <div className="space-y-2">
      {items.map((item) => {
        const pct = (Math.abs(item.shap) / max) * 100;
        const positive = item.shap >= 0;
        return (
          <div key={item.feature} className="flex items-center gap-3 text-sm">
            <span className="w-32 shrink-0 text-muted truncate">
              {LABELS[item.feature] ?? item.feature}
            </span>
            <div className="relative flex-1 h-5 flex items-center">
              <div className="absolute left-1/2 top-0 bottom-0 w-px bg-border" />
              <div
                className={clsx(
                  "absolute top-0 bottom-0 rounded-sm",
                  positive ? "bg-green" : "bg-red"
                )}
                style={
                  positive
                    ? { left: "50%", width: `${pct / 2}%` }
                    : { right: "50%", width: `${pct / 2}%` }
                }
              />
            </div>
            <span
              className={clsx(
                "w-16 shrink-0 text-right font-mono text-xs",
                positive ? "text-green" : "text-red"
              )}
            >
              {item.shap >= 0 ? "+" : ""}
              {item.shap.toFixed(3)}
            </span>
          </div>
        );
      })}
    </div>
  );
}
