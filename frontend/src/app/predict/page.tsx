import { api } from "@/lib/api";
import { AppShell } from "@/components/AppShell";
import { PredictStockExplorer } from "@/components/PredictStockExplorer";

export default async function PredictPage() {
  const [indices, stocks] = await Promise.all([api.indices(), api.stocks()]);

  return (
    <AppShell indices={indices} stocks={stocks}>
      <div className="mx-auto max-w-4xl px-6 py-8">
        <h1 className="text-lg font-semibold mb-1">Predict a Stock</h1>
        <p className="text-sm text-muted mb-6">
          Pick a stock and a time horizon. Short horizons get a real price
          target; long horizons get an explained directional outlook instead
          of a fabricated number.
        </p>
        <PredictStockExplorer stocks={stocks} />
      </div>
    </AppShell>
  );
}
