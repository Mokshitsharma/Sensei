import { api } from "@/lib/api";
import { AppShell } from "@/components/AppShell";
import { ThemeToggle } from "@/components/ThemeToggle";

export default async function SettingsPage() {
  const [indices, stocks] = await Promise.all([api.indices(), api.stocks()]);

  return (
    <AppShell indices={indices} stocks={stocks}>
      <div className="mx-auto max-w-2xl px-6 py-8">
        <h1 className="text-lg font-semibold mb-6">Settings</h1>

        <div className="rounded-xl border border-border bg-surface p-5">
          <p className="text-sm font-semibold text-foreground">Appearance</p>
          <p className="text-xs text-muted mt-1 mb-4">
            Switch between the dark and light theme. Your choice is saved on
            this device.
          </p>
          <ThemeToggle />
        </div>
      </div>
    </AppShell>
  );
}
