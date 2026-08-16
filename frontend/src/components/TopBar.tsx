import Link from "next/link";
import { SignInButton, UserButton } from "@clerk/nextjs";
import { auth } from "@clerk/nextjs/server";
import type { StockListItem } from "@/lib/types";
import { SearchBar } from "./SearchBar";

const NAV_ITEMS = [
  { label: "Explore", href: "/explore", enabled: true },
  { label: "Portfolio", href: "/portfolio", enabled: false },
  { label: "Screener", href: "/screener", enabled: false },
];

export async function TopBar({ stocks }: { stocks: StockListItem[] }) {
  const { userId } = await auth();

  return (
    <header className="flex items-center gap-3 sm:gap-5 border-b border-border bg-surface px-3 sm:px-4 py-3">
      <Link href="/" className="text-lg sm:text-xl font-bold text-accent shrink-0">
        Sensei AI
      </Link>

      <nav className="hidden md:flex items-center gap-1 shrink-0">
        {NAV_ITEMS.map((item) =>
          item.enabled ? (
            <Link
              key={item.label}
              href={item.href}
              className="rounded-md px-2.5 py-1.5 text-sm font-medium text-muted hover:text-foreground hover:bg-surface-2"
            >
              {item.label}
            </Link>
          ) : (
            <span
              key={item.label}
              title="Coming soon"
              className="flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-sm font-medium text-muted/40 cursor-not-allowed"
            >
              {item.label}
              <span className="text-[9px] uppercase tracking-wide border border-border rounded px-1 py-0.5">
                Soon
              </span>
            </span>
          )
        )}
      </nav>

      <div className="flex-1 min-w-0">
        <SearchBar stocks={stocks} />
      </div>
      <div className="flex items-center gap-3 sm:gap-4 shrink-0 text-muted">
        <button aria-label="Notifications" className="hover:text-foreground">
          🔔
        </button>
        <Link href="/settings" aria-label="Settings" className="hover:text-foreground">
          ⚙
        </Link>
        {userId ? (
          <UserButton
            appearance={{
              elements: { avatarBox: "h-8 w-8" },
            }}
          />
        ) : (
          <SignInButton mode="modal">
            <button className="rounded-lg bg-accent px-3 py-1.5 text-sm font-semibold text-black hover:bg-accent-hover">
              Sign in
            </button>
          </SignInButton>
        )}
      </div>
    </header>
  );
}
