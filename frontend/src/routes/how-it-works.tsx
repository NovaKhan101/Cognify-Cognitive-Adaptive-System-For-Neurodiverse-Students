import { createFileRoute } from "@tanstack/react-router";

export const Route = createFileRoute("/how-it-works")({
  head: () => ({
    meta: [
      { title: "How Cognify works — Adaptive AI for ASD learners" },
      { name: "description", content: "From onboarding to daily learning: how Cognify's AI adapts to each child in real time." },
      { property: "og:title", content: "How Cognify works" },
      { property: "og:description", content: "The step-by-step of adaptive, sensory-safe learning." },
    ],
  }),
  component: HowItWorks,
});

const steps = [
  { n: "01", t: "Gentle onboarding", d: "A short, low-pressure assessment builds a comfort profile — interests, sensory preferences, and starting level." },
  { n: "02", t: "Personalized path", d: "Cognify Intelligence generates a first learning path with the right pacing and content type for the child." },
  { n: "03", t: "Adaptive sessions", d: "During each session, the AI reads focus signals and adjusts difficulty, domain, or task type." },
  { n: "04", t: "Reflect & celebrate", d: "Sessions end with soft, positive summaries. Progress compounds without pressure." },
  { n: "05", t: "Caregiver insights", d: "Parents and therapists get clear analytics and suggestions in the dashboard." },
];

function HowItWorks() {
  return (
    <div className="mx-auto max-w-4xl px-4 pt-16 sm:px-6">
      <p className="text-xs font-semibold uppercase tracking-widest text-primary">How it works</p>
      <h1 className="mt-2 font-display text-4xl font-extrabold sm:text-5xl">
        A calm loop that <span className="text-gradient">learns with you</span>.
      </h1>
      <p className="mt-4 max-w-2xl text-lg text-foreground/70">
        Cognify's engine watches, adapts, and encourages — five gentle steps that repeat
        every session.
      </p>

      <div className="mt-12 space-y-5">
        {steps.map((s) => (
          <div key={s.n} className="glass flex gap-5 rounded-3xl p-6">
            <div className="grid h-14 w-14 shrink-0 place-items-center rounded-2xl gradient-primary font-display text-lg font-extrabold text-primary-foreground">
              {s.n}
            </div>
            <div className="min-w-0">
              <h3 className="text-lg font-semibold">{s.t}</h3>
              <p className="mt-1 text-sm text-muted-foreground">{s.d}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
