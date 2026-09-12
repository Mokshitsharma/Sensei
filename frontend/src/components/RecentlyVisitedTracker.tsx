"use client";

import { useEffect } from "react";
import { pushRecent } from "@/lib/recentlyVisited";

export function RecentlyVisitedTracker({
  ticker,
  name,
}: {
  ticker: string;
  name: string;
}) {
  useEffect(() => {
    pushRecent({ ticker, name });
  }, [ticker, name]);

  return null;
}
