import Link from "next/link";
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
        <div className="mb-6 flex items-start justify-between gap-4">
          <div>
            <h1 className="text-lg font-semibold mb-1">Predictions</h1>
            <p className="text-sm text-muted">
              Stocks the AI expects to move, by horizon and direction.
            </p>
          </div>
          <Link
            href="/predictions/accuracy"
            className="shrink-0 rounded-lg border border-border px-3 py-2 text-xs font-medium text-accent hover:bg-surface-2 transition-colors"
          >
            View Track Record
          </Link>
        </div>
        <PredictionsExplorer initialData={initialData} />
      </div>
    </AppShell>
  );
}
