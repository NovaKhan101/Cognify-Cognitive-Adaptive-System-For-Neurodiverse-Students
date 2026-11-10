import { useEffect, useState } from "react";
import { Sparkles } from "lucide-react";

const messages = [
  "Adapting to your focus level…",
  "Tuning difficulty gently…",
  "Recommending a calmer path…",
  "Learning your preferences…",
  "Balancing challenge & comfort…",
];

export function IntelligencePill({ className = "" }: { className?: string }) {
  const [i, setI] = useState(0);
  const [visible, setVisible] = useState(true);

  useEffect(() => {
    const t = setInterval(() => {
      setVisible(false);
      setTimeout(() => {
        setI((v) => (v + 1) % messages.length);
        setVisible(true);
      }, 350);
    }, 4200);
    return () => clearInterval(t);
  }, []);

  return (
    <div
      className={`inline-flex items-center gap-2 rounded-full border border-primary/20 bg-primary/8 px-3 py-1.5 text-xs font-medium text-primary shadow-sm backdrop-blur-md transition-opacity duration-300 ${
        visible ? "opacity-100" : "opacity-0"
      } ${className}`}
      style={{ background: "color-mix(in oklab, var(--color-primary) 10%, white)" }}
    >
      <span className="relative flex h-2 w-2">
        <span className="absolute inline-flex h-full w-full animate-pulse-soft rounded-full bg-accent opacity-75" />
        <span className="relative inline-flex h-2 w-2 rounded-full bg-accent" />
      </span>
      <Sparkles className="h-3 w-3" />
      <span>{messages[i]}</span>
    </div>
  );
}
