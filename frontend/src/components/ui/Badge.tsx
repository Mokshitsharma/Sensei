import clsx from "clsx";

export function Badge({
  children,
  tone = "default",
}: {
  children: React.ReactNode;
  tone?: "green" | "red" | "amber" | "default";
}) {
  const toneClass =
    tone === "green"
      ? "bg-green/10 text-green border-green/30"
      : tone === "red"
        ? "bg-red/10 text-red border-red/30"
        : tone === "amber"
          ? "bg-amber/10 text-amber border-amber/30"
          : "bg-surface-2 text-muted border-border";

  return (
    <span
      className={clsx(
        "inline-flex items-center rounded-md border px-2 py-1 text-xs font-semibold tracking-wide font-mono",
        toneClass
      )}
    >
      {children}
    </span>
  );
}

export function Change({
  value,
  pct,
  showValue = true,
}: {
  value: number;
  pct: number;
  showValue?: boolean;
}) {
  const up = value >= 0;
  return (
    <span
      className={clsx(
        "font-mono text-sm font-medium",
        up ? "text-green" : "text-red"
      )}
    >
      {up ? "▲" : "▼"}
      {showValue && ` ${Math.abs(value).toLocaleString(undefined, { maximumFractionDigits: 2 })}`}{" "}
      ({pct >= 0 ? "+" : ""}
      {pct.toFixed(2)}%)
    </span>
  );
}
