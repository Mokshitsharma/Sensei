import clsx from "clsx";

export function StatTile({
  label,
  value,
  color,
  mono = true,
}: {
  label: string;
  value: string;
  color?: "green" | "red" | "amber" | "default";
  mono?: boolean;
}) {
  const colorClass =
    color === "green"
      ? "text-green"
      : color === "red"
        ? "text-red"
        : color === "amber"
          ? "text-amber"
          : "text-foreground";

  return (
    <div className="rounded-lg border border-border bg-surface-2 px-4 py-3">
      <div className="text-[11px] uppercase tracking-wide text-muted font-medium">
        {label}
      </div>
      <div
        className={clsx(
          "mt-1 text-lg font-semibold",
          colorClass,
          mono && "font-mono"
        )}
      >
        {value}
      </div>
    </div>
  );
}

export function StatGrid({ children }: { children: React.ReactNode }) {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">{children}</div>
  );
}
