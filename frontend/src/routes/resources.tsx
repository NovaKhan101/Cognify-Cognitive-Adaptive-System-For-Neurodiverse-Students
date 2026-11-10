import { createFileRoute } from "@tanstack/react-router";
import { BookOpen, FileText, Video, Users } from "lucide-react";

export const Route = createFileRoute("/resources")({
  head: () => ({
    meta: [
      { title: "Resources — Cognify" },
      { name: "description", content: "Guides, research, and tools for families and educators supporting children with ASD." },
      { property: "og:title", content: "Cognify resources" },
      { property: "og:description", content: "Helpful reading and tools for families and educators." },
    ],
  }),
  component: Resources,
});

const items = [
  { icon: BookOpen, tag: "Guide", t: "A gentle intro to adaptive learning", d: "How AI-based pacing supports children on the spectrum." },
  { icon: FileText, tag: "Research", t: "Sensory-safe UI: what we've learned", d: "Design choices that reduce overload and improve retention." },
  { icon: Video, tag: "Video", t: "Setting up Cognify with your family", d: "A 4-minute walkthrough for first-time parents." },
  { icon: Users, tag: "Community", t: "Therapist collaboration guide", d: "Sharing progress across caregivers, gently and privately." },
  { icon: BookOpen, tag: "Guide", t: "Reading progress at a glance", d: "How to interpret the dashboard without pressure." },
  { icon: FileText, tag: "Research", t: "Focus signals and calm loops", d: "The engine behind the Cognify Intelligence pill." },
];

function Resources() {
  return (
    <div className="mx-auto max-w-7xl px-4 pt-16 sm:px-6">
      <p className="text-xs font-semibold uppercase tracking-widest text-primary">Resources</p>
      <h1 className="mt-2 font-display text-4xl font-extrabold sm:text-5xl">
        Learn along <span className="text-gradient">with us</span>.
      </h1>
      <p className="mt-4 max-w-2xl text-lg text-foreground/70">
        Guides, research, and family-friendly explainers for the calm learning path.
      </p>

      <div className="mt-10 grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
        {items.map((r) => (
          <article key={r.t} className="glass rounded-3xl p-6">
            <div className="flex items-center gap-3">
              <span className="grid h-11 w-11 place-items-center rounded-2xl gradient-primary">
                <r.icon className="h-5 w-5 text-white" />
              </span>
              <span className="rounded-full bg-foreground/5 px-3 py-1 text-xs font-semibold text-muted-foreground">
                {r.tag}
              </span>
            </div>
            <h3 className="mt-4 font-semibold">{r.t}</h3>
            <p className="mt-1 text-sm text-muted-foreground">{r.d}</p>
            <a href="#" className="mt-4 inline-flex text-sm font-semibold text-primary">Read →</a>
          </article>
        ))}
      </div>
    </div>
  );
}