"use client";

import Link from "next/link";
import { useEffect } from "react";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error(error);
  }, [error]);

  return (
    <div className="flex min-h-screen flex-col items-center justify-center gap-4 px-6 text-center">
      <span className="text-2xl font-bold text-accent">Sensei AI</span>
      <h1 className="text-lg font-semibold">Something went wrong</h1>
      <p className="max-w-sm text-sm text-muted">
        The AI engine hit a snag loading this page — it might be a temporary
        data-source hiccup. Try again, or head back to Explore.
      </p>
      <div className="flex gap-3 mt-2">
        <button
          onClick={() => reset()}
          className="rounded-lg bg-accent px-4 py-2 text-sm font-semibold text-black hover:bg-accent-hover"
        >
          Try again
        </button>
        <Link
          href="/explore"
          className="rounded-lg border border-border px-4 py-2 text-sm font-semibold text-foreground hover:bg-surface-2"
        >
          Back to Explore
        </Link>
      </div>
    </div>
  );
}
