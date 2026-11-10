import { createFileRoute } from "@tanstack/react-router";
import { Brain, Sparkles, Shield, Gauge, Palette, Users, Video, Award } from "lucide-react";

export const Route = createFileRoute("/features")({
  head: () => ({
    meta: [
      { title: "Features — Cognify adaptive learning" },
      { name: "description", content: "Adaptive AI, sensory-safe design, live analytics, video recommendations, and more." },
      { property: "og:title", content: "Cognify features" },
      { property: "og:description", content: "Everything Cognify offers for calmer, adaptive learning." },
    ],
  }),
  component: Features,
});

const features = [
  { icon: Brain, title: "Adaptive AI engine", body: "Cognify Intelligence continuously tunes tasks, difficulty, and pacing to each child's focus level." },
  { icon: Palette, title: "Sensory-safe UI", body: "Muted palette, gentle motion, generous space. No harsh reds, no flashing." },
  { icon: Gauge, title: "Focus-aware pacing", body: "The engine slows down or shifts domain when signs of overload appear." },
  { icon: Video, title: "Smart video paths", body: "Curated video content recommended based on interest and past behavior." },
  { icon: Shield, title: "Private by design", body: "No ads. No third-party tracking. Behavioral data stays with the family." },
  { icon: Users, title: "Family + therapist view", body: "Shared dashboard so caregivers and specialists stay aligned." },
  { icon: Award, title: "Gentle rewards", body: "Progress is celebrated softly — no over-stimulating sounds or animations." },
  { icon: Sparkles, title: "Cognify Intelligence pill", body: "A subtle indicator lets you see when the AI adapts — visible but never intrusive." },
];

function Features() {
  return (
    <div className="mx-auto max-w-7xl px-4 pt-16 sm:px-6">
      <p className="text-xs font-semibold uppercase tracking-widest text-primary">Features</p>
      <h1 className="mt-2 font-display text-4xl font-extrabold sm:text-5xl">
        Every detail, <span className="text-gradient">designed for calm</span>.
      </h1>
      <p className="mt-4 max-w-2xl text-lg text-foreground/70">
        Cognify combines adaptive AI with sensory-considered design. Here's what makes it
        different.
      </p>

      <div className="mt-12 grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
        {features.map((f) => (
          <div key={f.title} className="glass rounded-3xl p-6">
            <div className="grid h-11 w-11 place-items-center rounded-2xl gradient-primary">
              <f.icon className="h-5 w-5 text-white" />
            </div>
            <h3 className="mt-4 font-semibold">{f.title}</h3>
            <p className="mt-2 text-sm text-muted-foreground">{f.body}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
