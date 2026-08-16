import Link from "next/link";
import { SignInButton, UserButton } from "@clerk/nextjs";
import { auth } from "@clerk/nextjs/server";
import type { StockListItem } from "@/lib/types";
import { SearchBar } from "./SearchBar";

export async function TopBar({ stocks }: { stocks: StockListItem[] }) {
  const { userId } = await auth();

  return (
    <header className="flex items-center gap-3 border-b border-border bg-surface px-3 sm:px-4 py-3">
      <Link href="/explore" className="text-lg sm:text-xl font-bold text-accent shrink-0">
        Sensei AI
      </Link>
      <div className="flex-1 min-w-0">
        <SearchBar stocks={stocks} />
      </div>
      <div className="flex items-center gap-3 sm:gap-4 shrink-0 text-muted">
        <button aria-label="Notifications" className="hover:text-foreground">
          🔔
        </button>
        <button aria-label="Settings" className="hover:text-foreground">
          ⚙
        </button>
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
