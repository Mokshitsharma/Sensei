import Link from "next/link";
import { Badge } from "@/components/ui/Badge";

const DIRECTION_LABEL = {
  UP: "May go up",
  DOWN: "May go down",
  FLAT: "Hold",
} as const;

const DIRECTION_TONE = {
  UP: "green",
  DOWN: "red",
  FLAT: "amber",
} as const;

const CONFIDENCE_PCT: Record<string, number> = {
  HIGH: 85,
  MEDIUM: 60,
  LOW: 30,
};

function initials(name: string) {
  return name
    .split(" ")
    .filter(Boolean)
    .slice(0, 2)
    .map((w) => w[0])
    .join("")
    .toUpperCase();
}

// Deterministic placeholder "logo" color from the ticker, so each stock
// gets a stable, distinct avatar color without needing real logo assets.
function avatarHue(ticker: string) {
  let hash = 0;
  for (let i = 0; i < ticker.length; i++) hash = (hash * 31 + ticker.charCodeAt(i)) % 360;
  return hash;
}

export function PredictionStockCard({
  ticker,
  company,
  currentPrice,
  todayChangePct,
  direction,
  confidence,
  predictedPrice,
}: {
  ticker: string;
  company: string;
  currentPrice: number;
  todayChangePct?: number;
  direction: "UP" | "DOWN" | "FLAT";
  confidence: "HIGH" | "MEDIUM" | "LOW";
  predictedPrice?: number;
  /** Kept optional for callers passing it; no longer rendered inline —
   * horizon context comes from the page/section, not repeated per card. */
  horizonLabel?: string;
}) {
  const hue = avatarHue(ticker);
  const confidencePct = CONFIDENCE_PCT[confidence] ?? 50;

  return (
    <Link
      href={`/stock/${encodeURIComponent(ticker)}`}
      className="block rounded-xl border border-border bg-surface p-4 transition-colors hover:border-accent/50 hover:bg-surface-2"
    >
      <div className="flex items-center gap-2.5">
        <div
          className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-[11px] font-bold text-black"
          style={{ backgroundColor: `hsl(${hue} 70% 60%)` }}
        >
          {initials(company)}
        </div>
        <div className="min-w-0 flex-1">
          <div className="text-sm font-semibold text-foreground truncate">{company}</div>
          <div className="text-[11px] text-muted font-mono">{ticker}</div>
        </div>
      </div>

      <div className="mt-3 flex items-baseline justify-between">
        <span className="font-mono text-base font-semibold">
          ₹{currentPrice.toLocaleString(undefined, { maximumFractionDigits: 2 })}
        </span>
        {todayChangePct !== undefined && (
          <span
            className={`font-mono text-xs ${todayChangePct >= 0 ? "text-green" : "text-red"}`}
          >
            {todayChangePct >= 0 ? "▲" : "▼"}
            {todayChangePct >= 0 ? "+" : ""}
            {todayChangePct.toFixed(2)}%
          </span>
        )}
      </div>

      <div className="mt-2.5">
        <Badge tone={DIRECTION_TONE[direction]}>{DIRECTION_LABEL[direction]}</Badge>
      </div>
      {predictedPrice !== undefined && (
        <div className="mt-1.5 text-xs text-muted font-mono">
          → ₹{predictedPrice.toLocaleString(undefined, { maximumFractionDigits: 2 })}
        </div>
      )}

      <div className="mt-3 flex items-center gap-2">
        <div className="h-1 flex-1 rounded-full bg-surface-2 overflow-hidden">
          <div
            className={`h-full rounded-full ${
              confidence === "HIGH" ? "bg-green" : confidence === "MEDIUM" ? "bg-amber" : "bg-red"
            }`}
            style={{ width: `${confidencePct}%` }}
          />
        </div>
        <span className="text-[10px] text-muted shrink-0">{confidence}</span>
      </div>
    </Link>
  );
}
