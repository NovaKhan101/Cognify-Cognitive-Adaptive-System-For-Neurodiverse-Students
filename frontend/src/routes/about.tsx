import { createFileRoute } from "@tanstack/react-router";
import { Heart, Users, Lightbulb, Target } from "lucide-react";

export const Route = createFileRoute("/about")({
  head: () => ({
    meta: [
      { title: "About Cognify — Calm adaptive learning for ASD" },
      { name: "description", content: "Our mission: build a calm, adaptive learning space for children on the autism spectrum, powered by ethical AI." },
      { property: "og:title", content: "About Cognify" },
      { property: "og:description", content: "The team and mission behind Cognify." },
    ],
  }),
  component: About,
});

function About() {
  return (
    <div className="mx-auto max-w-5xl px-4 pt-16 sm:px-6">
      <p className="text-xs font-semibold uppercase tracking-widest text-primary">About Cognify</p>
      <h1 className="mt-2 font-display text-4xl font-extrabold sm:text-5xl">
        Learning shouldn't feel <span className="text-gradient">overwhelming</span>.
      </h1>
      <p className="mt-5 max-w-2xl text-lg text-foreground/70">
        Cognify was built alongside educators, therapists, and families of children on the
        autism spectrum. We believe technology should meet learners where they are — with
        empathy, patience, and adaptive intelligence.
      </p>

      <div className="mt-12 grid gap-5 md:grid-cols-2">
        {[
          { icon: Heart, title: "Empathy first", body: "Every interaction is designed to reduce anxiety and celebrate effort, not just outcomes." },
          { icon: Target, title: "Personal, not one-size", body: "Our AI shapes each session to focus, mood, and progress — a truly individual path." },
          { icon: Users, title: "Family-inclusive", body: "Parents and therapists get transparent insights without invasive tracking." },
          { icon: Lightbulb, title: "Evidence-informed", body: "Grounded in research on ASD learning, executive function, and sensory processing." },
        ].map((v) => (
          <div key={v.title} className="glass rounded-3xl p-6">
            <div className="grid h-11 w-11 place-items-center rounded-2xl gradient-primary">
              <v.icon className="h-5 w-5 text-white" />
            </div>
            <h3 className="mt-4 text-lg font-semibold">{v.title}</h3>
            <p className="mt-2 text-sm text-muted-foreground">{v.body}</p>
          </div>
        ))}
      </div>

      <div className="mt-14 glass rounded-3xl p-8">
        <h2 className="font-display text-2xl font-bold">Our story</h2>
        <p className="mt-3 text-foreground/70">
          Cognify started as a research project into how adaptive AI could reduce learner
          stress for children with ASD. Today, it's a platform that quietly adjusts thousands
          of small decisions — pacing, contrast, task order, sensory intensity — so every
          child can learn in a way that feels safe.
        </p>
      </div>
    </div>
  );
}
