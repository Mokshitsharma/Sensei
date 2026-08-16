import type { Decision, TradeSetup } from "@/lib/types";
import { Badge } from "./ui/Badge";

const TONE = {
  BUY: "green",
  SELL: "red",
  HOLD: "amber",
} as const;

const GLOW_VAR = {
  BUY: "var(--green)",
  SELL: "var(--red)",
  HOLD: "var(--amber)",
} as const;

export function AiSignalPanel({
  decision,
  swingSetup,
}: {
  decision: Decision;
  swingSetup: TradeSetup;
}) {
  const hasLevels = !(
    swingSetup.error &&
    swingSetup.entry_zone[0] === 0 &&
    swingSetup.entry_zone[1] === 0
  );

  return (
    <div className="animate-fade-in-up sticky top-4 rounded-xl border border-border bg-surface p-5">
      <p className="text-xs uppercase tracking-wide text-muted font-semibold">
        AI Recommendation
      </p>
      <p className="text-sm text-muted mt-2">
        {(decision.confidence * 100).toFixed(1)}% confidence
      </p>

      <div>
        <div className="mt-3">
          <span
            className="animate-glow-pulse inline-block rounded-md"
            style={{ "--tone": GLOW_VAR[decision.action] } as React.CSSProperties}
          >
            <Badge tone={TONE[decision.action]}>
              <span className="text-base">{decision.action}</span>
            </Badge>
          </span>
        </div>

        {decision.narrative?.headline && (
          <p className="mt-4 text-sm text-foreground/90 leading-relaxed border-t border-border pt-4">
            {decision.narrative.headline}
          </p>
        )}

        {hasLevels && (
          <div className="mt-4 border-t border-border pt-4 space-y-2.5">
            <p className="text-xs uppercase tracking-wide text-muted font-semibold mb-1">
              Swing Trade Levels
            </p>
            <LevelRow label="Entry Zone" value={`₹${swingSetup.entry_zone[0].toFixed(2)}–${swingSetup.entry_zone[1].toFixed(2)}`} />
            <LevelRow label="Stop Loss" value={`₹${swingSetup.stop_loss.toFixed(2)}`} tone="red" />
            <LevelRow label="Target 1" value={`₹${swingSetup.target_1.toFixed(2)}`} tone="green" />
            <LevelRow label="Risk / Reward" value={`${swingSetup.risk_reward.toFixed(1)}x`} />
            <LevelRow label="Validity" value={swingSetup.validity} />
          </div>
        )}
      </div>

      <p className="mt-5 text-[11px] text-muted leading-relaxed border-t border-border pt-3">
        Informational only — Sensei AI does not place trades or connect to a
        broker. Levels are AI-generated suggestions, not investment advice.
      </p>
    </div>
  );
}

function LevelRow({
  label,
  value,
  tone,
}: {
  label: string;
  value: string;
  tone?: "green" | "red";
}) {
  return (
    <div className="flex items-center justify-between text-sm">
      <span className="text-muted">{label}</span>
      <span
        className={`font-mono font-semibold ${
          tone === "green" ? "text-green" : tone === "red" ? "text-red" : ""
        }`}
      >
        {value}
      </span>
    </div>
  );
}
