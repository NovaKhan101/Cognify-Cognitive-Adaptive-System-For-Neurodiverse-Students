import { createFileRoute, Link } from "@tanstack/react-router";
import {
  Brain,
  Heart,
  Shield,
  Sparkles,
  ArrowRight,
  Puzzle,
  LineChart,
  Play,
} from "lucide-react";
import { IntelligencePill } from "@/components/intelligence-pill";

export const Route = createFileRoute("/")({
  component: Home,
});

function Home() {
  return (
    <div className="mx-auto max-w-7xl px-4 pt-10 sm:px-6 sm:pt-16">
      {/* HERO */}
      <section className="relative overflow-hidden rounded-[2.5rem] gradient-hero px-6 py-16 sm:px-12 sm:py-24">
        <div className="relative z-10 mx-auto max-w-3xl text-center">
          <div className="mb-6 flex justify-center">
            <IntelligencePill />
          </div>
          <h1 className="font-display text-4xl font-extrabold tracking-tight sm:text-6xl">
            Learning that <span className="text-gradient">adapts</span> to every mind.
          </h1>
          <p className="mx-auto mt-5 max-w-2xl text-lg text-foreground/70 sm:text-xl">
            Cognify is an AI-powered adaptive learning platform designed for children on the
            autism spectrum. Calm pacing, personalized paths, real progress.
          </p>
          <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
            <Link
              to="/auth"
              search={{ mode: "signup" } as never}
              className="inline-flex items-center gap-2 rounded-full gradient-primary px-6 py-3 text-sm font-semibold text-primary-foreground shadow-soft transition-transform hover:scale-[1.02]"
            >
              Get started free <ArrowRight className="h-4 w-4" />
            </Link>
            <Link
              to="/ai-demo"
              className="inline-flex items-center gap-2 rounded-full glass px-6 py-3 text-sm font-semibold"
            >
              <Play className="h-4 w-4" /> Try the AI demo
            </Link>
          </div>
        </div>

        {/* Floating glass tiles */}
        <div className="pointer-events-none absolute -left-10 top-16 hidden animate-float md:block">
          <div className="glass rounded-3xl p-4">
            <div className="flex items-center gap-3">
              <div className="grid h-10 w-10 place-items-center rounded-2xl gradient-accent">
                <Heart className="h-5 w-5 text-white" />
              </div>
              <div>
                <div className="text-xs text-muted-foreground">Comfort level</div>
                <div className="text-sm font-semibold">Excellent</div>
              </div>
            </div>
          </div>
        </div>
        <div
          className="pointer-events-none absolute -right-8 bottom-16 hidden animate-float md:block"
          style={{ animationDelay: "1.5s" }}
        >
          <div className="glass rounded-3xl p-4">
            <div className="flex items-center gap-3">
              <div className="grid h-10 w-10 place-items-center rounded-2xl gradient-primary">
                <Brain className="h-5 w-5 text-white" />
              </div>
              <div>
                <div className="text-xs text-muted-foreground">AI path</div>
                <div className="text-sm font-semibold">Adjusting…</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* WHY */}
      <section className="mt-20">
        <div className="mb-10 text-center">
          <p className="text-xs font-semibold uppercase tracking-widest text-primary">
            Built for calm
          </p>
          <h2 className="mt-2 text-3xl font-bold sm:text-4xl">
            Designed with neurodivergent learners in mind
          </h2>
        </div>
        <div className="grid gap-5 md:grid-cols-3">
          {[
            {
              icon: Brain,
              title: "Adaptive AI engine",
              body: "Difficulty, pacing and content shift in real time based on focus and behavior signals.",
              grad: "gradient-primary",
            },
            {
              icon: Heart,
              title: "Sensory-safe design",
              body: "Muted palette, soft motion, generous spacing — nothing overstimulating.",
              grad: "gradient-accent",
            },
            {
              icon: Shield,
              title: "Progress you can trust",
              body: "Clear analytics for parents and therapists. Milestones celebrated gently.",
              grad: "gradient-primary",
            },
          ].map((f) => (
            <div key={f.title} className="glass rounded-3xl p-6">
              <div className={`grid h-12 w-12 place-items-center rounded-2xl ${f.grad}`}>
                <f.icon className="h-6 w-6 text-white" />
              </div>
              <h3 className="mt-4 text-lg font-semibold">{f.title}</h3>
              <p className="mt-2 text-sm text-muted-foreground">{f.body}</p>
            </div>
          ))}
        </div>
      </section>

      {/* STATS */}
      <section className="mt-20">
        <div className="glass grid gap-6 rounded-3xl p-8 sm:grid-cols-4 sm:p-10">
          {[
            ["93%", "Learner engagement"],
            ["4.8×", "Faster milestone reach"],
            ["12k+", "Adaptive tasks"],
            ["100%", "Sensory-safe UI"],
          ].map(([n, l]) => (
            <div key={l} className="text-center">
              <div className="font-display text-3xl font-extrabold text-gradient sm:text-4xl">
                {n}
              </div>
              <div className="mt-1 text-xs uppercase tracking-wider text-muted-foreground">
                {l}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* HOW SNAPSHOT */}
      <section className="mt-20 grid gap-6 md:grid-cols-3">
        {[
          { icon: Puzzle, step: "01", title: "Meet the learner", body: "A gentle onboarding builds a profile of strengths, interests and comfort zones." },
          { icon: Sparkles, step: "02", title: "AI shapes the path", body: "Cognify chooses tasks, videos and difficulty that fit the moment." },
          { icon: LineChart, step: "03", title: "Progress unfolds", body: "Small wins compound. Parents and educators see growth in real time." },
        ].map((s) => (
          <div key={s.step} className="glass rounded-3xl p-6">
            <div className="flex items-center justify-between">
              <div className="grid h-11 w-11 place-items-center rounded-2xl gradient-primary">
                <s.icon className="h-5 w-5 text-white" />
              </div>
              <span className="font-display text-2xl font-extrabold text-primary/30">
                {s.step}
              </span>
            </div>
            <h3 className="mt-4 text-lg font-semibold">{s.title}</h3>
            <p className="mt-2 text-sm text-muted-foreground">{s.body}</p>
          </div>
        ))}
      </section>

      {/* CTA */}
      <section className="mt-20">
        <div className="relative overflow-hidden rounded-[2rem] gradient-primary p-10 text-center text-primary-foreground shadow-soft sm:p-14">
          <h2 className="font-display text-3xl font-extrabold sm:text-4xl">
            A calmer path to learning starts today.
          </h2>
          <p className="mx-auto mt-3 max-w-xl text-primary-foreground/85">
            Try Cognify with your child. Free while we're in early access.
          </p>
        </div>
      </section>
    </div>
  );
}
