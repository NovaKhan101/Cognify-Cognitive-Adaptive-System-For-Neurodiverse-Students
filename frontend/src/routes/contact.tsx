import { createFileRoute } from "@tanstack/react-router";
import { Mail, MessageCircle, MapPin } from "lucide-react";
import { useState } from "react";

export const Route = createFileRoute("/contact")({
  head: () => ({
    meta: [
      { title: "Contact — Cognify" },
      { name: "description", content: "Talk to the Cognify team about adaptive learning for children with ASD." },
      { property: "og:title", content: "Contact Cognify" },
      { property: "og:description", content: "Reach out to our team." },
    ],
  }),
  component: Contact,
});

function Contact() {
  const [sent, setSent] = useState(false);
  return (
    <div className="mx-auto max-w-6xl px-4 pt-16 sm:px-6">
      <p className="text-xs font-semibold uppercase tracking-widest text-primary">Contact</p>
      <h1 className="mt-2 font-display text-4xl font-extrabold sm:text-5xl">
        Let's <span className="text-gradient">talk</span>.
      </h1>
      <p className="mt-4 max-w-2xl text-lg text-foreground/70">
        Questions, partnerships, or feedback — we'd love to hear from you.
      </p>

      <div className="mt-12 grid gap-6 md:grid-cols-[1.4fr_1fr]">
        <form
          onSubmit={(e) => { e.preventDefault(); setSent(true); }}
          className="glass rounded-3xl p-6 sm:p-8"
        >
          {sent ? (
            <div className="py-12 text-center">
              <div className="mx-auto grid h-14 w-14 place-items-center rounded-2xl gradient-accent">
                <MessageCircle className="h-7 w-7 text-white" />
              </div>
              <h2 className="mt-4 font-display text-2xl font-bold">Thank you</h2>
              <p className="mt-2 text-muted-foreground">We'll get back to you within one working day.</p>
            </div>
          ) : (
            <>
              <div className="grid gap-4 sm:grid-cols-2">
                <label className="block">
                  <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Name</span>
                  <input required className="mt-1 w-full rounded-2xl border border-border bg-white/70 px-4 py-3 text-sm outline-none focus:border-primary" />
                </label>
                <label className="block">
                  <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Email</span>
                  <input required type="email" className="mt-1 w-full rounded-2xl border border-border bg-white/70 px-4 py-3 text-sm outline-none focus:border-primary" />
                </label>
              </div>
              <label className="mt-4 block">
                <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Message</span>
                <textarea required rows={5} className="mt-1 w-full rounded-2xl border border-border bg-white/70 px-4 py-3 text-sm outline-none focus:border-primary" />
              </label>
              <button className="mt-5 inline-flex items-center gap-2 rounded-full gradient-primary px-6 py-3 text-sm font-semibold text-primary-foreground shadow-soft">
                Send message
              </button>
            </>
          )}
        </form>

        <aside className="space-y-4">
          {[
            { icon: Mail, t: "Email", d: "hello@cognify.app" },
            { icon: MessageCircle, t: "Support", d: "support@cognify.app" },
            { icon: MapPin, t: "HQ", d: "Remote-first · Global" },
          ].map((c) => (
            <div key={c.t} className="glass flex items-center gap-4 rounded-3xl p-5">
              <div className="grid h-11 w-11 place-items-center rounded-2xl gradient-primary">
                <c.icon className="h-5 w-5 text-white" />
              </div>
              <div>
                <div className="text-xs uppercase tracking-wider text-muted-foreground">{c.t}</div>
                <div className="text-sm font-semibold">{c.d}</div>
              </div>
            </div>
          ))}
        </aside>
      </div>
    </div>
  );
}
