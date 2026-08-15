import type { SrLevel, SupportResistance as SrType } from "@/lib/types";
import { Card } from "./ui/Card";
import { Badge } from "./ui/Badge";

function LevelRow({ level, label }: { level: SrLevel; label: string }) {
  const tone =
    level.strength === "Strong"
      ? "green"
      : level.strength === "Moderate"
        ? "amber"
        : "default";
  return (
    <div className="flex items-center justify-between py-2 border-b border-border last:border-0 text-sm">
      <span className="text-muted w-10">{label}</span>
      <span className="font-mono font-semibold flex-1">
        ₹{level.price.toLocaleString(undefined, { maximumFractionDigits: 2 })}
      </span>
      <Badge tone={tone}>{level.strength}</Badge>
      <span className="text-muted text-xs w-32 text-right truncate">
        {level.methods.slice(0, 2).join(", ")}
      </span>
    </div>
  );
}

export function SupportResistance({ data }: { data: SrType }) {
  const pivotEntries = Object.entries(data.pivot_data ?? {});

  return (
    <Card title="Support & Resistance">
      <div className="grid md:grid-cols-2 gap-6">
        <div>
          <p className="text-xs uppercase tracking-wide text-muted font-semibold mb-2">
            Resistance
          </p>
          {data.resistances.map((r, i) => (
            <LevelRow key={i} level={r} label={`R${i + 1}`} />
          ))}
        </div>
        <div>
          <p className="text-xs uppercase tracking-wide text-muted font-semibold mb-2">
            Support
          </p>
          {data.supports.map((s, i) => (
            <LevelRow key={i} level={s} label={`S${i + 1}`} />
          ))}
        </div>
      </div>

      {pivotEntries.length > 0 && (
        <div className="mt-6 pt-4 border-t border-border">
          <p className="text-xs uppercase tracking-wide text-muted font-semibold mb-3">
            Pivot Points (standard)
          </p>
          <div className="flex flex-wrap gap-3">
            {pivotEntries.map(([key, value]) => (
              <div
                key={key}
                className="rounded-md border border-border bg-surface-2 px-3 py-2 text-center min-w-[64px]"
              >
                <div className="text-[10px] text-muted uppercase">{key}</div>
                <div className="font-mono text-sm font-semibold">{value}</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </Card>
  );
}
