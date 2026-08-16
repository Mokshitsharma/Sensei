"use client";

import { useEffect, useState } from "react";

type Theme = "dark" | "light";

function applyTheme(theme: Theme) {
  document.documentElement.dataset.theme = theme;
  localStorage.setItem("sensei-theme", theme);
}

export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme | null>(null);

  useEffect(() => {
    const stored = localStorage.getItem("sensei-theme");
    setTheme(stored === "light" ? "light" : "dark");
  }, []);

  if (theme === null) {
    return <div className="h-10 w-full max-w-xs rounded-lg bg-surface-2 animate-pulse" />;
  }

  return (
    <div className="inline-flex rounded-lg border border-border bg-surface-2 p-1">
      <button
        onClick={() => {
          setTheme("dark");
          applyTheme("dark");
        }}
        className={`flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-colors ${
          theme === "dark"
            ? "bg-accent text-black"
            : "text-muted hover:text-foreground"
        }`}
      >
        🌙 Dark
      </button>
      <button
        onClick={() => {
          setTheme("light");
          applyTheme("light");
        }}
        className={`flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-colors ${
          theme === "light"
            ? "bg-accent text-black"
            : "text-muted hover:text-foreground"
        }`}
      >
        ☀️ Light
      </button>
    </div>
  );
}
