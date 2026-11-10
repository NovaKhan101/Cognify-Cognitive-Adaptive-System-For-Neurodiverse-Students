import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useState, useEffect } from "react";
import { Sparkles, LogOut, Play, Brain, AlertCircle } from "lucide-react";
import { BarChart, Bar, XAxis, ResponsiveContainer, YAxis, Tooltip } from "recharts";
import { api, type UserProfile } from "@/lib/api";

export const Route = createFileRoute("/dashboard")({
  head: () => ({
    meta: [
      { title: "Dashboard — Cognify" },
      { name: "description", content: "Your personalized adaptive learning dashboard." },
    ],
  }),
  component: Dashboard,
});

// 5 hardcoded local videos as per spec
const VIDEOS = [
  { title: "Washing Hands", src: "/videos/washing-hands.mp4" },
  { title: "Saying Hello", src: "/videos/saying-hello.mp4" },
  { title: "Taking Turns", src: "/videos/taking-turns.mp4" },
  { title: "Calm Breathing", src: "/videos/calm-breathing.mp4" },
  { title: "Brushing Teeth", src: "/videos/brushing-teeth.mp4" },
];

// Intelligent formatter to keep chart labels clean without ugly "..." slicing
const formatDomainName = (name: string) => {
  switch (name) {
    case "Executive Function": return "Exec Func";
    case "Social Cognition": return "Social Cog";
    case "Sensory Processing": return "Sensory";
    case "Motor Skills": return "Motor";
    case "Emotional Regulation": return "Emot Reg";
    default: return name;
  }
};

function Dashboard() {
  const navigate = useNavigate();
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [domainData, setDomainData] = useState<{ name: string; v: number }[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeVideo, setActiveVideo] = useState<string | null>(null);
  const [videoErrors, setVideoErrors] = useState<string[]>([]);

  const studentId = localStorage.getItem("cognify_student_id");

  useEffect(() => {
    // If no student ID in localStorage, lock the dashboard
    if (!studentId) {
      setIsLoading(false);
      return;
    }

    const fetchData = async () => {
      try {
        const [profileData, ksData] = await Promise.all([
          api.getProfile(studentId),
          api.getKnowledgeState(studentId),
        ]);

        setProfile(profileData);

        // Map knowledge_state Record<string, number> to chart array format
        const chartData = Object.entries(ksData.knowledge_state).map(([name, val]) => ({
          name: name, // Keep full name for Tooltip, format in XAxis
          v: Math.round(val),
        }));
        setDomainData(chartData);
      } catch (err: any) {
        setError(err.message || "Failed to load dashboard data.");
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, [studentId]);

  const handleLogout = () => {
    localStorage.removeItem("cognify_student_id");
    localStorage.removeItem("cognify_student_name");
    localStorage.removeItem("cognify_cognitive_profile");
    navigate({ to: "/auth" });
  };

  const handleVideoError = (src: string) => {
    setVideoErrors((prev) => [...prev, src]);
    setActiveVideo(null);
  };

  // ==================== LOADING STATE ====================
  if (isLoading) {
    return (
      <div className="flex min-h-[60vh] items-center justify-center">
        <div className="flex flex-col items-center gap-3">
          <Sparkles className="h-8 w-8 animate-pulse text-primary" />
          <p className="text-sm text-muted-foreground">Loading your learning profile...</p>
        </div>
      </div>
    );
  }

  // ==================== LOCKED STATE (NO AUTH) ====================
  if (!studentId || !profile) {
    return (
      <div className="mx-auto flex min-h-[70vh] max-w-md flex-col items-center justify-center px-4 text-center">
        <div className="glass w-full rounded-3xl p-8">
          <div className="mx-auto mb-6 grid h-12 w-12 place-items-center rounded-full bg-foreground/5">
            <LogOut className="h-5 w-5 text-muted-foreground" />
          </div>
          <h1 className="font-display text-xl font-bold">Please Log In First</h1>
          <p className="mt-2 text-sm text-muted-foreground">
            You need a learning profile to access the dashboard.
          </p>
          <button
            onClick={() => navigate({ to: "/auth" })}
            className="mt-6 w-full rounded-full gradient-primary px-6 py-3 text-sm font-semibold text-primary-foreground shadow-soft"
          >
            Go to Login
          </button>
        </div>
      </div>
    );
  }

  if (error) {
    return <div className="py-20 text-center text-red-500">{error}</div>;
  }

  // ==================== MAIN DASHBOARD ====================
  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 lg:px-8">
      {/* Header */}
      <div className="mb-8 flex flex-col justify-between gap-4 sm:flex-row sm:items-center">
        <div>
          <div className="flex items-center gap-2">
            <h1 className="font-display text-2xl font-extrabold">Welcome back, {profile.name}</h1>
            {profile.cognitive_profile && (
              <span className="flex items-center gap-1 rounded-full bg-primary/10 px-2 py-0.5 text-xs font-semibold text-primary">
                <Brain className="h-3 w-3" />
                {profile.profile_friendly_name}
              </span>
            )}
          </div>
          <p className="mt-1 text-sm text-muted-foreground">
            System Mode: <span className="font-semibold capitalize">{profile.system_mode.replace(/_/g, " ")}</span>
            <span className="mx-2">•</span>
            Total Tasks: <span className="font-semibold">{profile.total_attempts}</span>
          </p>
        </div>
        <button
          onClick={handleLogout}
          className="flex items-center justify-center gap-2 rounded-full border border-border bg-white/70 px-4 py-2 text-sm font-medium text-muted-foreground transition hover:bg-foreground/5"
        >
          <LogOut className="h-4 w-4" /> Log Out
        </button>
      </div>

      {/* Bento Grid Layout */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* 1. Hero Card - Start Session */}
        <div className="glass relative overflow-hidden rounded-3xl p-8 lg:col-span-2 lg:row-span-2">
          <div className="relative z-10 flex h-full flex-col justify-between">
            <div>
              <span className="inline-flex items-center gap-1 rounded-full bg-primary/10 px-3 py-1 text-xs font-semibold text-primary">
                <Sparkles className="h-3 w-3" /> Ready to learn
              </span>
              <h2 className="mt-4 font-display text-3xl font-extrabold leading-tight">
                Continue your personalized learning journey
              </h2>
              <p className="mt-2 max-w-md text-sm text-muted-foreground">
                The system will adapt to your cognitive profile, fatigue level, and current mastery to provide the perfect next task.
              </p>
            </div>

            <div className="mt-8">
              <button
                onClick={() => navigate({ to: "/session" })}
                className="flex items-center justify-center gap-2 rounded-full gradient-primary px-8 py-4 text-sm font-bold text-primary-foreground shadow-soft transition hover:shadow-lg"
              >
                Start Session
              </button>
            </div>
          </div>

          {/* Decorative blobs */}
          <div className="absolute -right-20 -top-20 h-64 w-64 rounded-full bg-primary/10 blur-3xl"></div>
          <div className="absolute -bottom-20 -left-20 h-64 w-64 rounded-full bg-accent/10 blur-3xl"></div>
        </div>

        {/* 2. Domain Mastery Chart */}
        <div className="glass rounded-3xl p-6">
          <div className="mb-4 flex items-center justify-between">
            <h3 className="font-display text-lg font-bold">Domain Mastery</h3>
            <span className="text-xs text-muted-foreground">DKT Model</span>
          </div>

          {domainData.length > 0 ? (
            <div className="h-[200px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={domainData} margin={{ top: 5, right: 5, left: -20, bottom: 5 }}>
                  <XAxis
                    dataKey="name"
                    tickFormatter={formatDomainName}
                    tick={{ fontSize: 10, fill: "#888" }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis hide domain={[0, 100]} />
                  <Tooltip
                    cursor={{ fill: "transparent" }}
                    contentStyle={{ borderRadius: "12px", border: "1px solid #eee", fontSize: "12px" }}
                  />
                  <Bar dataKey="v" fill="oklch(0.62 0.19 285)" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <div className="flex h-[200px] items-center justify-center text-sm text-muted-foreground">
              No data yet
            </div>
          )}
        </div>

        {/* 3. Videos Section */}
        <div className="glass rounded-3xl p-6 lg:col-span-3">
          <div className="mb-6 flex items-center justify-between">
            <div>
              <h3 className="font-display text-lg font-bold">Daily Living Skills</h3>
              <p className="text-xs text-muted-foreground">Watch and learn with these short visual guides</p>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
            {VIDEOS.map((video) => {
              const hasError = videoErrors.includes(video.src);
              return (
                <div key={video.src} className="group cursor-pointer">
                  <div
                    className={`relative aspect-video overflow-hidden rounded-2xl bg-foreground/5 ${
                      hasError ? "cursor-not-allowed opacity-60" : ""
                    }`}
                    onClick={() => !hasError && setActiveVideo(activeVideo === video.src ? null : video.src)}
                  >
                    {activeVideo === video.src && !hasError ? (
                      <video
                        src={video.src}
                        className="h-full w-full object-cover"
                        autoPlay
                        controls
                        loop
                        onError={() => handleVideoError(video.src)}
                      />
                    ) : (
                      <div className="flex h-full flex-col items-center justify-center p-4 text-center transition group-hover:bg-foreground/10">
                        <div className="grid h-10 w-10 place-items-center rounded-full bg-primary/10 text-primary">
                          {hasError ? (
                            <AlertCircle className="h-4 w-4 text-red-500" />
                          ) : (
                            <Play className="h-4 w-4" />
                          )}
                        </div>
                        <p className="mt-2 text-xs font-semibold">
                          {hasError ? "Unavailable" : video.title}
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}