import { ReactNode } from "react";
import clsx from "clsx";

export function Card({
  children,
  className,
  title,
  badge,
}: {
  children: ReactNode;
  className?: string;
  title?: ReactNode;
  badge?: ReactNode;
}) {
  return (
    <div
      className={clsx(
        "rounded-xl border border-border bg-surface",
        className
      )}
    >
      {(title || badge) && (
        <div className="flex items-center justify-between px-5 py-4 border-b border-border">
          <div className="text-sm font-semibold text-foreground flex items-center gap-2">
            {title}
          </div>
          {badge}
        </div>
      )}
      <div className="p-5">{children}</div>
    </div>
  );
}
