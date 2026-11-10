import { Link, useRouterState } from "@tanstack/react-router";
import { useState, useEffect } from "react";
import { Menu, X, Sparkles } from "lucide-react";

const links = [
  { to: "/", label: "Home" },
  { to: "/about", label: "About" },
  { to: "/features", label: "Features" },
  { to: "/ai-demo", label: "AI Demo" },
  { to: "/contact", label: "Contact" },
] as const;

export function SiteNav() {
  const [open, setOpen] = useState(false);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const pathname = useRouterState({ select: (s) => s.location.pathname });

  // Check login status whenever the route changes
  useEffect(() => {
    setIsLoggedIn(!!localStorage.getItem("cognify_student_id"));
  }, [pathname]);

  return (
    <header className="sticky top-0 z-50 px-4 pt-4">
      <nav className="glass mx-auto flex max-w-7xl items-center justify-between rounded-full px-4 py-2.5 sm:px-6">
        <Link to="/" className="flex items-center gap-2 pl-1">
          <span className="grid h-8 w-8 place-items-center rounded-xl gradient-primary shadow-soft">
            <Sparkles className="h-4 w-4 text-primary-foreground" />
          </span>
          <span className="font-display text-lg font-extrabold tracking-tight">Cognify</span>
        </Link>

        {/* Desktop Nav Links */}
        <div className="hidden items-center gap-1 lg:flex">
          {links.map((l) => {
            const active = pathname === l.to;
            return (
              <Link
                key={l.to}
                to={l.to}
                className={`rounded-full px-3.5 py-1.5 text-sm font-medium transition-colors ${
                  active
                    ? "bg-primary/10 text-primary"
                    : "text-foreground/70 hover:bg-foreground/5 hover:text-foreground"
                }`}
              >
                {l.label}
              </Link>
            );
          })}
        </div>

        {/* Desktop Auth/Button Area */}
        <div className="hidden items-center gap-2 lg:flex">
          {isLoggedIn ? (
            <Link
              to="/dashboard"
              className="rounded-full gradient-primary px-5 py-2 text-sm font-semibold text-primary-foreground shadow-soft"
            >
              Dashboard
            </Link>
          ) : (
            <>
              <Link
                to="/auth"
                className="rounded-full px-4 py-1.5 text-sm font-medium text-foreground/80 hover:text-foreground"
              >
                Log in
              </Link>
              <Link
                to="/auth"
                search={{ mode: "signup" } as never}
                className="rounded-full gradient-primary px-4 py-2 text-sm font-semibold text-primary-foreground shadow-soft"
              >
                Sign up
              </Link>
            </>
          )}
        </div>

        {/* Mobile Menu Button */}
        <button
          onClick={() => setOpen((v) => !v)}
          className="grid h-9 w-9 place-items-center rounded-full bg-foreground/5 lg:hidden"
          aria-label="Toggle menu"
        >
          {open ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
        </button>
      </nav>

      {/* Mobile Dropdown Nav */}
      {open && (
        <div className="glass mx-auto mt-2 max-w-7xl rounded-3xl p-3 lg:hidden">
          <div className="flex flex-col">
            {links.map((l) => (
              <Link
                key={l.to}
                to={l.to}
                onClick={() => setOpen(false)}
                className={`rounded-2xl px-4 py-2.5 text-sm font-medium ${
                  pathname === l.to ? "bg-primary/10 text-primary" : "text-foreground/80"
                }`}
              >
                {l.label}
              </Link>
            ))}
            <div className="mt-2 flex gap-2 border-t border-border/50 pt-3">
              {isLoggedIn ? (
                <Link
                  to="/dashboard"
                  onClick={() => setOpen(false)}
                  className="flex-1 rounded-full gradient-primary px-4 py-2 text-center text-sm font-semibold text-primary-foreground"
                >
                  Go to Dashboard
                </Link>
              ) : (
                <>
                  <Link
                    to="/auth"
                    onClick={() => setOpen(false)}
                    className="flex-1 rounded-full border border-border bg-background/60 px-4 py-2 text-center text-sm font-medium"
                  >
                    Log in
                  </Link>
                  <Link
                    to="/auth"
                    search={{ mode: "signup" } as never}
                    onClick={() => setOpen(false)}
                    className="flex-1 rounded-full gradient-primary px-4 py-2 text-center text-sm font-semibold text-primary-foreground"
                  >
                    Sign up
                  </Link>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </header>
  );
}