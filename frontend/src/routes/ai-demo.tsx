import { createFileRoute } from "@tanstack/react-router";
import { useEffect, useMemo, useState } from "react";
import { Sparkles, Check, X, Volume2, Heart, Trophy } from "lucide-react";
import { IntelligencePill } from "@/components/intelligence-pill";

export const Route = createFileRoute("/ai-demo")({
  head: () => ({
    meta: [
      { title: "AI Learning Demo — Cognify" },
      { name: "description", content: "Try Cognify's adaptive AI with four real mini-lessons: shape matching, word building, emotion recognition, and counting." },
      { property: "og:title", content: "Cognify AI demo" },
      { property: "og:description", content: "Real interactive mini-lessons powered by our adaptive learning engine." },
    ],
  }),
  component: AIDemo,
});

type Mood = "calm" | "focused" | "restless" | "tired";
type DemoId = "shapes" | "words" | "emotions" | "count";

const moodTuning: Record<Mood, { length: string; sensory: string; difficulty: string; hintDelay: number }> = {
  calm:     { length: "6 min", sensory: "Standard calm", difficulty: "Comfort +1", hintDelay: 6000 },
  focused:  { length: "8 min", sensory: "Standard calm", difficulty: "Slight challenge", hintDelay: 8000 },
  restless: { length: "2 min", sensory: "Ultra-soft",    difficulty: "Restorative", hintDelay: 3500 },
  tired:    { length: "3 min", sensory: "Ultra-soft",    difficulty: "Restful", hintDelay: 5000 },
};

function AIDemo() {
  const [mood, setMood] = useState<Mood>("calm");
  const [demo, setDemo] = useState<DemoId>("shapes");
  const [xp, setXp] = useState(0);
  const [reasoning, setReasoning] = useState<string[]>([
    "Learner state: calm. Choosing a gentle visual task.",
  ]);

  const addReason = (line: string) =>
    setReasoning((r) => [line, ...r].slice(0, 6));

  const onMood = (m: Mood) => {
    setMood(m);
    addReason(
      m === "restless" ? "Detected restless signals — shortening session, softer visuals."
      : m === "tired"   ? "Low energy — switching to a restful, listening-forward task."
      : m === "focused" ? "Focus is high — nudging difficulty up by one step."
      : "Calm baseline — comfort +1, standard sensory mode."
    );
  };

  const onDemo = (d: DemoId) => {
    setDemo(d);
    addReason(
      d === "shapes"   ? "Domain: Visual reasoning. Loading pattern-match set."
      : d === "words"  ? "Domain: Language. Loading 3-letter word tiles."
      : d === "emotions" ? "Domain: Social. Loading emotion-recognition cards."
      : "Domain: Numbers. Loading counting task with soft visuals."
    );
  };

  const tune = moodTuning[mood];

  return (
    <div className="mx-auto max-w-6xl px-4 pt-16 pb-24 sm:px-6">
      <div className="flex flex-wrap items-center gap-3">
        <p className="text-xs font-semibold uppercase tracking-widest text-primary">AI Learning Demo</p>
        <IntelligencePill />
      </div>
      <h1 className="mt-2 font-display text-4xl font-extrabold sm:text-5xl">
        Try Cognify <span className="text-gradient">for real</span>.
      </h1>
      <p className="mt-4 max-w-2xl text-lg text-foreground/70">
        Pick a focus state and a lesson. Each task is a real mini-activity — the same kind
        your child would see. Watch the adaptive rationale update as you play.
      </p>

      {/* Controls */}
      <div className="mt-8 glass rounded-3xl p-6 sm:p-8">
        <div className="grid gap-6 sm:grid-cols-2">
          <div>
            <div className="text-sm font-medium text-muted-foreground">Learner state</div>
            <div className="mt-3 flex flex-wrap gap-2">
              {(Object.keys(moodTuning) as Mood[]).map((m) => (
                <button
                  key={m}
                  onClick={() => onMood(m)}
                  className={`rounded-full px-4 py-2 text-sm font-medium capitalize transition ${
                    mood === m
                      ? "gradient-primary text-primary-foreground shadow-soft"
                      : "bg-foreground/5 text-foreground/70 hover:bg-foreground/10"
                  }`}
                >
                  {m}
                </button>
              ))}
            </div>
          </div>
          <div>
            <div className="text-sm font-medium text-muted-foreground">Lesson</div>
            <div className="mt-3 flex flex-wrap gap-2">
              {([
                ["shapes", "Shape match"],
                ["words", "Word building"],
                ["emotions", "Feelings"],
                ["count", "Counting"],
              ] as [DemoId, string][]).map(([id, label]) => (
                <button
                  key={id}
                  onClick={() => onDemo(id)}
                  className={`rounded-full px-4 py-2 text-sm font-medium transition ${
                    demo === id
                      ? "gradient-accent text-primary-foreground shadow-soft"
                      : "bg-foreground/5 text-foreground/70 hover:bg-foreground/10"
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* XP bar */}
        <div className="mt-6 flex items-center gap-3">
          <Trophy className="h-4 w-4 text-warning" />
          <div className="flex-1">
            <div className="mb-1 flex justify-between text-xs text-muted-foreground">
              <span>Session progress</span>
              <span>{xp} / 100 XP</span>
            </div>
            <div className="h-2 w-full overflow-hidden rounded-full bg-foreground/10">
              <div
                className="h-full gradient-primary transition-all duration-500"
                style={{ width: `${Math.min(100, xp)}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Task + rationale */}
      <div className="mt-6 grid gap-6 lg:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
        <div className="glass rounded-3xl p-6 sm:p-8">
          {demo === "shapes"   && <ShapeMatch key={mood} mood={mood} onWin={(n) => { setXp((x) => x + n); addReason(`Correct match — reinforcing visual pattern recognition (+${n} XP).`); }} onMiss={() => addReason("Miss — softening next distractor.")} />}
          {demo === "words"    && <WordBuild key={mood} mood={mood} onWin={(n) => { setXp((x) => x + n); addReason(`Word completed — vocabulary node strengthened (+${n} XP).`); }} onMiss={() => addReason("Misplaced tile — offering a gentle hint.")} />}
          {demo === "emotions" && <EmotionPick key={mood} mood={mood} onWin={(n) => { setXp((x) => x + n); addReason(`Emotion recognized — social-cognition score +1 (+${n} XP).`); }} onMiss={() => addReason("Emotion mismatch — will re-teach with clearer examples.")} />}
          {demo === "count"    && <CountingGame key={mood} mood={mood} onWin={(n) => { setXp((x) => x + n); addReason(`Counted correctly — numeracy improving (+${n} XP).`); }} onMiss={() => addReason("Off by one — offering a slower recount.")} />}
        </div>

        <div className="glass rounded-3xl p-6">
          <div className="flex items-center gap-2 text-xs uppercase tracking-wider text-primary">
            <Sparkles className="h-3.5 w-3.5" /> Adaptive rationale
          </div>
          <ul className="mt-4 space-y-2 text-sm">
            <li><span className="text-muted-foreground">Session length:</span> <b>{tune.length}</b></li>
            <li><span className="text-muted-foreground">Difficulty:</span> <b>{tune.difficulty}</b></li>
            <li><span className="text-muted-foreground">Sensory mode:</span> <b>{tune.sensory}</b></li>
          </ul>
          <div className="mt-5 text-xs uppercase tracking-wider text-muted-foreground">Live decisions</div>
          <ul className="mt-2 space-y-2">
            {reasoning.map((r, i) => (
              <li
                key={i}
                className="rounded-2xl border border-border/50 bg-white/60 px-3 py-2 text-sm"
                style={{ opacity: 1 - i * 0.12 }}
              >
                {r}
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}

/* ---------------- Shape Match ---------------- */
type Shape = { id: string; icon: string; color: string };
const SHAPES: Shape[] = [
  { id: "circle",   icon: "●", color: "text-primary" },
  { id: "square",   icon: "■", color: "text-secondary" },
  { id: "triangle", icon: "▲", color: "text-accent" },
  { id: "star",     icon: "★", color: "text-warning" },
];

function ShapeMatch({ mood, onWin, onMiss }: { mood: Mood; onWin: (n: number) => void; onMiss: () => void }) {
  const [round, setRound] = useState(0);
  const target = SHAPES[round % SHAPES.length];
  const [picked, setPicked] = useState<string | null>(null);
  const options = useMemo(() => {
    const count = mood === "restless" || mood === "tired" ? 3 : 4;
    return [...SHAPES].sort(() => Math.random() - 0.5).slice(0, count).includes(target)
      ? [...SHAPES].sort(() => Math.random() - 0.5).slice(0, count)
      : [target, ...SHAPES.filter((s) => s.id !== target.id)].slice(0, count).sort(() => Math.random() - 0.5);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [round, mood]);

  const pick = (id: string) => {
    setPicked(id);
    if (id === target.id) {
      onWin(10);
      setTimeout(() => { setPicked(null); setRound((r) => r + 1); }, 900);
    } else {
      onMiss();
      setTimeout(() => setPicked(null), 900);
    }
  };

  return (
    <div>
      <div className="text-xs uppercase tracking-wider text-muted-foreground">Visual reasoning · Round {round + 1}</div>
      <h3 className="mt-1 font-display text-2xl font-bold">Which one matches?</h3>
      <div className="mt-6 flex items-center justify-center">
        <div className="grid h-32 w-32 place-items-center rounded-3xl gradient-hero text-6xl">
          <span className={target.color}>{target.icon}</span>
        </div>
      </div>
      <div className="mt-8 grid grid-cols-2 gap-3 sm:grid-cols-4">
        {options.map((o) => {
          const state =
            picked === o.id ? (o.id === target.id ? "correct" : "wrong") : "idle";
          return (
            <button
              key={o.id + round}
              onClick={() => !picked && pick(o.id)}
              className={`group relative grid aspect-square place-items-center rounded-3xl border transition ${
                state === "correct" ? "border-accent bg-accent/15" :
                state === "wrong"   ? "border-destructive bg-destructive/10" :
                "border-border/60 bg-white/70 hover:-translate-y-0.5 hover:shadow-soft"
              }`}
            >
              <span className={`text-5xl ${o.color}`}>{o.icon}</span>
              {state === "correct" && <Check className="absolute right-2 top-2 h-4 w-4 text-accent" />}
              {state === "wrong"   && <X     className="absolute right-2 top-2 h-4 w-4 text-destructive" />}
            </button>
          );
        })}
      </div>
    </div>
  );
}

/* ---------------- Word Building ---------------- */
const WORDS = ["CAT", "SUN", "DOG", "HAT", "BUS"];
function WordBuild({ mood, onWin, onMiss }: { mood: Mood; onWin: (n: number) => void; onMiss: () => void }) {
  const [wordIdx, setWordIdx] = useState(0);
  const word = WORDS[wordIdx % WORDS.length];
  const [slots, setSlots] = useState<(string | null)[]>(() => Array(word.length).fill(null));
  const [tiles, setTiles] = useState<string[]>(() => shuffleTiles(word, mood));

  useEffect(() => {
    setSlots(Array(word.length).fill(null));
    setTiles(shuffleTiles(word, mood));
  }, [wordIdx, mood, word]);

  const place = (letter: string, tileIdx: number) => {
    const empty = slots.findIndex((s) => s === null);
    if (empty === -1) return;
    if (word[empty] === letter) {
      const next = [...slots]; next[empty] = letter; setSlots(next);
      const nextTiles = [...tiles]; nextTiles.splice(tileIdx, 1); setTiles(nextTiles);
      if (next.every((s) => s !== null)) {
        onWin(15);
        setTimeout(() => setWordIdx((i) => i + 1), 900);
      }
    } else {
      onMiss();
    }
  };

  return (
    <div>
      <div className="flex items-center justify-between">
        <div>
          <div className="text-xs uppercase tracking-wider text-muted-foreground">Language · Word {wordIdx + 1}</div>
          <h3 className="mt-1 font-display text-2xl font-bold">Build the word</h3>
        </div>
        <button className="inline-flex items-center gap-1.5 rounded-full bg-foreground/5 px-3 py-1.5 text-xs font-medium hover:bg-foreground/10">
          <Volume2 className="h-3.5 w-3.5" /> Hear it
        </button>
      </div>

      <div className="mt-6 flex justify-center gap-3">
        {slots.map((s, i) => (
          <div key={i} className={`grid h-16 w-14 place-items-center rounded-2xl border-2 border-dashed font-display text-3xl font-extrabold ${
            s ? "border-accent bg-accent/10 text-accent-foreground" : "border-border/70 bg-white/60 text-muted-foreground"
          }`}>
            {s ?? "·"}
          </div>
        ))}
      </div>

      <div className="mt-8 flex flex-wrap justify-center gap-2">
        {tiles.map((t, i) => (
          <button
            key={t + i}
            onClick={() => place(t, i)}
            className="grid h-12 w-12 place-items-center rounded-2xl bg-white/80 font-display text-xl font-bold shadow-soft transition hover:-translate-y-0.5"
          >
            {t}
          </button>
        ))}
      </div>
    </div>
  );
}
function shuffleTiles(word: string, mood: Mood) {
  const distract = mood === "restless" || mood === "tired" ? 0 : 2;
  const extras = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    .split("").filter((l) => !word.includes(l))
    .sort(() => Math.random() - 0.5).slice(0, distract);
  return [...word.split(""), ...extras].sort(() => Math.random() - 0.5);
}

/* ---------------- Emotions ---------------- */
const EMOTIONS = [
  { face: "😊", label: "Happy",   options: ["Happy", "Sad", "Angry"] },
  { face: "😢", label: "Sad",     options: ["Excited", "Sad", "Calm"] },
  { face: "😮", label: "Surprised", options: ["Surprised", "Sleepy", "Angry"] },
  { face: "😌", label: "Calm",    options: ["Scared", "Calm", "Happy"] },
];
function EmotionPick({ mood, onWin, onMiss }: { mood: Mood; onWin: (n: number) => void; onMiss: () => void }) {
  const [idx, setIdx] = useState(0);
  const item = EMOTIONS[idx % EMOTIONS.length];
  const [picked, setPicked] = useState<string | null>(null);
  const opts = mood === "restless" ? item.options.slice(0, 2) : item.options;

  const pick = (o: string) => {
    setPicked(o);
    if (o === item.label) {
      onWin(12);
      setTimeout(() => { setPicked(null); setIdx((i) => i + 1); }, 900);
    } else {
      onMiss();
      setTimeout(() => setPicked(null), 900);
    }
  };

  return (
    <div>
      <div className="text-xs uppercase tracking-wider text-muted-foreground">Social · Card {idx + 1}</div>
      <h3 className="mt-1 font-display text-2xl font-bold">How is this friend feeling?</h3>
      <div className="mt-6 flex items-center justify-center">
        <div className="grid h-40 w-40 place-items-center rounded-full gradient-hero text-8xl">
          {item.face}
        </div>
      </div>
      <div className="mt-8 flex flex-wrap justify-center gap-3">
        {opts.map((o) => {
          const state = picked === o ? (o === item.label ? "correct" : "wrong") : "idle";
          return (
            <button
              key={o}
              onClick={() => !picked && pick(o)}
              className={`rounded-2xl border px-6 py-3 text-sm font-semibold transition ${
                state === "correct" ? "border-accent bg-accent/15" :
                state === "wrong"   ? "border-destructive bg-destructive/10" :
                "border-border/60 bg-white/70 hover:-translate-y-0.5 hover:shadow-soft"
              }`}
            >
              {o}
            </button>
          );
        })}
      </div>
    </div>
  );
}

/* ---------------- Counting ---------------- */
function CountingGame({ mood, onWin, onMiss }: { mood: Mood; onWin: (n: number) => void; onMiss: () => void }) {
  const [round, setRound] = useState(0);
  const max = mood === "restless" || mood === "tired" ? 5 : 8;
  const target = useMemo(() => 2 + Math.floor(Math.random() * (max - 1)), [round, max]);
  const options = useMemo(() => {
    const set = new Set<number>([target]);
    while (set.size < 4) set.add(Math.max(1, target + (Math.floor(Math.random() * 5) - 2)));
    return [...set].sort(() => Math.random() - 0.5);
  }, [target]);
  const [picked, setPicked] = useState<number | null>(null);

  const pick = (n: number) => {
    setPicked(n);
    if (n === target) {
      onWin(10);
      setTimeout(() => { setPicked(null); setRound((r) => r + 1); }, 900);
    } else {
      onMiss();
      setTimeout(() => setPicked(null), 900);
    }
  };

  return (
    <div>
      <div className="text-xs uppercase tracking-wider text-muted-foreground">Numbers · Round {round + 1}</div>
      <h3 className="mt-1 font-display text-2xl font-bold">How many hearts?</h3>
      <div className="mt-6 flex min-h-32 flex-wrap items-center justify-center gap-3 rounded-3xl gradient-hero p-6">
        {Array.from({ length: target }).map((_, i) => (
          <Heart key={i} className="h-8 w-8 fill-accent text-accent" />
        ))}
      </div>
      <div className="mt-6 flex flex-wrap justify-center gap-3">
        {options.map((n) => {
          const state = picked === n ? (n === target ? "correct" : "wrong") : "idle";
          return (
            <button
              key={n}
              onClick={() => picked === null && pick(n)}
              className={`grid h-14 w-14 place-items-center rounded-2xl border font-display text-2xl font-bold transition ${
                state === "correct" ? "border-accent bg-accent/15" :
                state === "wrong"   ? "border-destructive bg-destructive/10" :
                "border-border/60 bg-white/80 hover:-translate-y-0.5 hover:shadow-soft"
              }`}
            >
              {n}
            </button>
          );
        })}
      </div>
    </div>
  );
}
