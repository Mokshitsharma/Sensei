"use client";

import { ReactNode, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import clsx from "clsx";

export function Tabs({
  tabs,
  defaultIndex = 0,
}: {
  tabs: { label: string; content: ReactNode }[];
  defaultIndex?: number;
}) {
  const [active, setActive] = useState(defaultIndex);

  return (
    <div>
      <div className="flex gap-6 border-b border-border overflow-x-auto">
        {tabs.map((t, i) => (
          <button
            key={t.label}
            onClick={() => setActive(i)}
            className={clsx(
              "relative whitespace-nowrap pb-3 pt-1 text-sm font-semibold uppercase tracking-wide transition-colors",
              i === active
                ? "text-accent"
                : "text-muted hover:text-foreground"
            )}
          >
            {t.label}
            {i === active && (
              <motion.div
                layoutId="tab-underline"
                className="absolute left-0 right-0 -bottom-px h-0.5 bg-accent"
                transition={{ type: "spring", stiffness: 500, damping: 40 }}
              />
            )}
          </button>
        ))}
      </div>
      <div className="pt-6 grid">
        <AnimatePresence initial={false}>
          <motion.div
            key={active}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15, ease: "easeOut" }}
            className="col-start-1 row-start-1"
          >
            {tabs[active]?.content}
          </motion.div>
        </AnimatePresence>
      </div>
    </div>
  );
}
