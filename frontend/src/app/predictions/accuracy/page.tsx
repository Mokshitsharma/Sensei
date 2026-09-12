import { api } from "@/lib/api";
import { AppShell } from "@/components/AppShell";
import { AccuracyBreakdownView } from "@/components/AccuracyBreakdownView";

export default async function PredictionsAccuracyPage() {
  const [indices, stocks] = await Promise.all([api.indices(), api.stocks()]);

  return (
    <AppShell indices={indices} stocks={stocks}>
      <div className="mx-auto max-w-4xl px-6 py-8">
        <h1 className="text-lg font-semibold mb-1">Prediction Accuracy</h1>
        <p className="text-sm text-muted mb-6">
          Every AI price call is logged the day it's made and graded once its
          target date actually arrives — nothing is scored early, and nothing
          is scored twice. This is the model's real track record, not a
          backtest.
        </p>
        <AccuracyBreakdownView />
      </div>
    </AppShell>
  );
}
