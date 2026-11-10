import { Link } from "@tanstack/react-router";
import { Sparkles } from "lucide-react";

export function SiteFooter() {
  return (
    <footer className="mt-24 px-4 pb-8">
      <div className="glass mx-auto max-w-7xl rounded-3xl px-6 py-10 sm:px-10">
        <div className="grid gap-10 md:grid-cols-4">
          <div className="md:col-span-2">
            <Link to="/" className="flex items-center gap-2">
              <span className="grid h-8 w-8 place-items-center rounded-xl gradient-primary">
                <Sparkles className="h-4 w-4 text-primary-foreground" />
              </span>
              <span className="font-display text-lg font-extrabold">Cognify</span>
            </Link>
            <p className="mt-3 max-w-sm text-sm text-muted-foreground">
              Calm, AI-powered adaptive learning for children on the autism spectrum. Zero
              sensory overload — just meaningful progress.
            </p>
          </div>
          
          <div>
            <div className="mb-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Product
            </div>
            <ul className="space-y-2 text-sm">
              <li><Link to="/features" className="hover:text-primary">Features</Link></li>
              <li><Link to="/ai-demo" className="hover:text-primary">AI demo</Link></li>
            </ul>
          </div>
          
          <div>
            <div className="mb-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Company
            </div>
            <ul className="space-y-2 text-sm">
              <li><Link to="/about" className="hover:text-primary">About</Link></li>
              <li><Link to="/contact" className="hover:text-primary">Contact</Link></li>
            </ul>
          </div>
        </div>
        
        <div className="mt-8 flex flex-col items-start justify-between gap-2 border-t border-border/50 pt-6 text-xs text-muted-foreground sm:flex-row sm:items-center">
          <p>© {new Date().getFullYear()} Cognify. Learning that adapts to every mind.</p>
          <p>Made with care for neurodivergent learners.</p>
        </div>
      </div>
    </footer>
  );
}