import Link from "next/link";

const NAV_ITEMS = [
  { label: "Home", href: "/", icon: "⌂", active: true, enabled: true },
  { label: "Portfolio", href: "/portfolio", icon: "▤", enabled: false },
  { label: "Screener", href: "/screener", icon: "≡", enabled: false },
  { label: "Settings", href: "/settings", icon: "⚙", enabled: false },
];

export function Sidebar() {
  return (
    <aside className="hidden md:flex md:w-16 lg:w-56 shrink-0 flex-col border-r border-border bg-surface">
      <nav className="flex-1 py-6 flex flex-col gap-1 px-2 lg:px-3">
        {NAV_ITEMS.map((item) =>
          item.enabled ? (
            <Link
              key={item.label}
              href={item.href}
              className={`flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors ${
                item.active
                  ? "bg-accent/10 text-accent border-l-2 border-accent"
                  : "text-muted hover:text-foreground hover:bg-surface-2"
              }`}
            >
              <span className="text-base w-5 text-center">{item.icon}</span>
              <span className="hidden lg:inline">{item.label}</span>
            </Link>
          ) : (
            <div
              key={item.label}
              title="Coming soon"
              className="flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium text-muted/40 cursor-not-allowed"
            >
              <span className="text-base w-5 text-center">{item.icon}</span>
              <span className="hidden lg:inline flex-1">{item.label}</span>
              <span className="hidden lg:inline text-[9px] uppercase tracking-wide border border-border rounded px-1 py-0.5">
                Soon
              </span>
            </div>
          )
        )}
      </nav>
    </aside>
  );
}
