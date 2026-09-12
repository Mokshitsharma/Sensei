import { api } from "@/lib/api";
import { AppShell } from "@/components/AppShell";
import { PredictionsExplorer } from "@/components/PredictionsExplorer";

export default async function PredictionsPage() {
  const [indices, stocks, initialData] = await Promise.all([
    api.indices(),
    api.stocks(),
    api.predictions("30d", "UP"),
  ]);

  return (
    <AppShell indices={indices} stocks={stocks}>
      <div className="mx-auto max-w-6xl px-6 py-8">
        <h1 className="text-lg font-semibold mb-1">Predictions</h1>
        <p className="text-sm text-muted mb-6">
          Stocks the AI expects to move, by horizon and direction.
        </p>
        <PredictionsExplorer initialData={initialData} />
      </div>
    </AppShell>
  );
}
