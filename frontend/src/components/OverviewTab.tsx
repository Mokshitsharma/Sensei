import type { BacktestResponse, Fundamentals, Signals } from "@/lib/types";
import { Card } from "./ui/Card";
import { StatGrid, StatTile } from "./ui/StatTile";

export function OverviewTab({
  signals,
  fundamentals,
  backtest,
}: {
  signals: Signals;
  fundamentals: Fundamentals;
  backtest: BacktestResponse;
}) {
  const marketCapCr = fundamentals.market_cap / 1e7;

  return (
    <div className="space-y-5">
      <Card title="Key Metrics">
        <StatGrid>
          <StatTile label="ML Prob (Up)" value={signals.ml_prob_up.toFixed(2)} />
          <StatTile
            label="LSTM Return"
            value={signals.lstm_return.toFixed(3)}
            color={signals.lstm_return >= 0 ? "green" : "red"}
          />
          <StatTile
            label="TCN Return"
            value={signals.tcn_return.toFixed(3)}
            color={signals.tcn_return >= 0 ? "green" : "red"}
          />
          <StatTile
            label="Market Regime"
            value={signals.regime}
            color={signals.regime === "BULL" ? "green" : "red"}
          />
        </StatGrid>
      </Card>

      <Card title="Company Fundamentals">
        <StatGrid>
          <StatTile label="Current Price" value={`₹${fundamentals.current_price}`} />
          <StatTile label="Market Cap" value={`₹${marketCapCr.toLocaleString(undefined, { maximumFractionDigits: 0 })} Cr`} />
          <StatTile label="ROE" value={fundamentals.roe.toFixed(2)} />
          <StatTile
            label="52W Range"
            value={`${fundamentals["52_week_low"]} – ${fundamentals["52_week_high"]}`}
          />
        </StatGrid>
      </Card>

      <Card title="Strategy Backtest">
        <EquitySparkline data={backtest.equity_curve} />
        <StatGrid>
          <StatTile
            label="Total Return"
            value={`${(backtest.metrics.total_return * 100).toFixed(2)}%`}
            color={backtest.metrics.total_return >= 0 ? "green" : "red"}
          />
          <StatTile label="Sharpe Ratio" value={backtest.metrics.sharpe_ratio.toFixed(2)} />
          <StatTile
            label="Max Drawdown"
            value={`${(backtest.metrics.max_drawdown * 100).toFixed(2)}%`}
            color="red"
          />
        </StatGrid>
      </Card>
    </div>
  );
}

function EquitySparkline({ data }: { data: number[] }) {
  if (data.length === 0) return null;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const range = max - min || 1;
  const points = data
    .map((v, i) => {
      const x = (i / (data.length - 1)) * 100;
      const y = 100 - ((v - min) / range) * 100;
      return `${x},${y}`;
    })
    .join(" ");

  const up = data[data.length - 1] >= data[0];

  return (
    <div className="h-32 mb-4">
      <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="w-full h-full">
        <polyline
          points={points}
          fill="none"
          stroke={up ? "#4ade80" : "#f87171"}
          strokeWidth="1.5"
          vectorEffect="non-scaling-stroke"
        />
      </svg>
    </div>
  );
}
