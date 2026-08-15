import type { Decision, Signals } from "@/lib/types";
import { Card } from "./ui/Card";

function Row({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color?: "green" | "red";
}) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-border last:border-0">
      <span className="text-sm text-muted">{label}</span>
      <span
        className={`font-mono text-sm font-semibold ${
          color === "green" ? "text-green" : color === "red" ? "text-red" : ""
        }`}
      >
        {value}
      </span>
    </div>
  );
}

const PPO_LABELS: Record<string, string> = { "0": "Hold", "1": "Accumulate", "2": "Reduce" };
function ppoLabel(action: unknown): string {
  return PPO_LABELS[String(action)] ?? String(action);
}

export function RawModelSignals({
  signals,
  decision,
}: {
  signals: Signals;
  decision: Decision;
}) {
  return (
    <Card title="⚙ Raw Model Signals">
      <div className="text-center mb-4">
        <div className="text-xs uppercase tracking-wide text-muted">
          Prediction Probability
        </div>
        <div className="text-3xl font-mono font-bold text-green mt-1">
          {(signals.ml_prob_up * 100).toFixed(1)}%
        </div>
      </div>
      <Row
        label="LSTM (Long-Short)"
        value={`${signals.lstm_return >= 0 ? "+" : ""}${(signals.lstm_return * 100).toFixed(1)}%`}
        color={signals.lstm_return >= 0 ? "green" : "red"}
      />
      <Row
        label="TCN (Temporal Conv)"
        value={`${signals.tcn_return >= 0 ? "+" : ""}${(signals.tcn_return * 100).toFixed(1)}%`}
        color={signals.tcn_return >= 0 ? "green" : "red"}
      />
      <Row label="HMM Regime" value={signals.regime} color={signals.regime === "BULL" ? "green" : "red"} />
      <Row label="PPO (RL Agent)" value={ppoLabel(signals.ppo_action)} />
      <div className="mt-4 flex items-center justify-between">
        <span className="text-xs uppercase tracking-wide text-muted">
          Ensemble Score
        </span>
        <span className="font-mono text-sm font-bold text-green">
          {decision.score.toFixed(1)}/5
        </span>
      </div>
    </Card>
  );
}
