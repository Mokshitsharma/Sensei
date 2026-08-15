import type { TradeSetup } from "@/lib/types";
import { Card } from "./ui/Card";
import { Badge } from "./ui/Badge";
import { StatGrid, StatTile } from "./ui/StatTile";

const BIAS_TONE = {
  BULLISH: "green",
  BEARISH: "red",
  NEUTRAL: "amber",
} as const;

export function TradeSetupCard({ setup }: { setup: TradeSetup }) {
  if (setup.error && setup.entry_zone[0] === 0 && setup.entry_zone[1] === 0) {
    return (
      <div className="rounded-lg border border-amber/30 bg-amber/10 px-4 py-3 text-sm text-amber">
        {setup.plan}
      </div>
    );
  }

  return (
    <div>
      <div className="flex items-center gap-2 mb-4">
        <Badge tone={BIAS_TONE[setup.bias]}>{setup.bias}</Badge>
        <span className="text-sm text-muted">— {setup.pattern}</span>
      </div>

      <StatGrid>
        <StatTile
          label="Entry Zone"
          value={`₹${setup.entry_zone[0].toFixed(2)}–${setup.entry_zone[1].toFixed(2)}`}
        />
        <StatTile label="Stop Loss" value={`₹${setup.stop_loss.toFixed(2)}`} color="red" />
        <StatTile label="Target 1" value={`₹${setup.target_1.toFixed(2)}`} color="green" />
        <StatTile
          label="Risk / Reward"
          value={`${setup.risk_reward.toFixed(1)}x`}
          color={setup.risk_reward >= 2 ? "green" : "amber"}
        />
      </StatGrid>

      {Object.keys(setup.key_levels).length > 0 && (
        <div className="mt-4">
          <p className="text-xs uppercase tracking-wide text-muted font-semibold mb-2">
            Key levels
          </p>
          <StatGrid>
            {Object.entries(setup.key_levels).map(([label, value]) => (
              <StatTile key={label} label={label} value={`₹${value.toLocaleString()}`} />
            ))}
          </StatGrid>
        </div>
      )}

      <div className="mt-5 rounded-lg border border-border bg-surface-2 p-4">
        <p className="text-xs uppercase tracking-wide text-muted font-semibold mb-1">
          Trade plan &middot; valid {setup.validity}
        </p>
        <p className="text-sm text-foreground/90 leading-relaxed">{setup.plan}</p>
      </div>
    </div>
  );
}
