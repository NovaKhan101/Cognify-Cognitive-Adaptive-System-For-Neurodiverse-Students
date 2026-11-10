import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useState, useEffect } from "react";
import { z } from "zod";
import { Sparkles, ChevronRight, ChevronDown, Home, RotateCw, ClipboardCheck, BarChart2, AlertTriangle } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip } from "recharts";
import { api, type SessionSummaryResponse, type SessionReportResponse } from "@/lib/api";

const search = z.object({
  session_id: z.string(),
});

export const Route = createFileRoute("/session-complete")({
  validateSearch: search,
  head: () => ({
    meta: [
      { title: "Session Complete — Cognify" },
      { name: "description", content: "Review your learning session progress." },
    ],
  }),
  component: SessionComplete,
});

// Reused intelligent formatter to keep chart labels clean
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

function SessionComplete() {
  const { session_id: sessionId } = Route.useSearch();
  const navigate = useNavigate();
  
  const [summary, setSummary] = useState<SessionSummaryResponse | null>(null);
  const [report, setReport] = useState<SessionReportResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isParentOpen, setIsParentOpen] = useState(false);

  useEffect(() => {
    // Guard: If no session ID in URL, redirect to dashboard
    if (!sessionId) {
      navigate({ to: "/dashboard" });
      return;
    }

    const fetchData = async () => {
      try {
        const [summaryData, reportData] = await Promise.all([
          api.getSessionSummary(sessionId),
          api.getSessionReport(sessionId),
        ]);
        setSummary(summaryData);
        setReport(reportData);
      } catch (err: any) {
        setError(err.message || "Failed to load session results.");
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, [sessionId, navigate]);

  // ==================== LOADING STATE ====================
  if (isLoading) {
    return (
      <div className="flex min-h-[80vh] items-center justify-center">
        <div className="flex flex-col items-center gap-3">
          <Sparkles className="h-8 w-8 animate-pulse text-primary" />
          <p className="text-sm text-muted-foreground">Generating your session report...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="mx-auto flex min-h-[70vh] max-w-md flex-col items-center justify-center px-4 text-center">
        <div className="glass w-full rounded-3xl p-8">
          <div className="mx-auto mb-6 grid h-12 w-12 place-items-center rounded-full bg-red-500/10">
            <AlertTriangle className="h-5 w-5 text-red-500" />
          </div>
          <h1 className="font-display text-xl font-bold">Something went wrong</h1>
          <p className="mt-2 text-sm text-muted-foreground">{error}</p>
          <button
            onClick={() => navigate({ to: "/dashboard" })}
            className="mt-6 w-full rounded-full gradient-primary px-6 py-3 text-sm font-semibold text-primary-foreground shadow-soft"
          >
            Back to Dashboard
          </button>
        </div>
      </div>
    );
  }

  if (!summary || !report) return null;

  // Map domain_breakdown to chart array format
  const domainChartData = Object.entries(report.domain_breakdown).map(([name, data]) => ({
    name: name,
    v: Math.round(data.accuracy_pct),
  }));

  // Metrics array for the parent grid
  const parentMetrics = [
    { label: "Total Attempts", value: report.metrics.total_attempts },
    { label: "Overall Accuracy", value: `${report.metrics.overall_accuracy_pct.toFixed(1)}%` },
    { label: "Avg Response Time", value: `${report.metrics.average_response_time_sec.toFixed(1)}s` },
    { label: "Hints Used", value: report.metrics.total_hints_used },
    { label: "Frustration Flags", value: report.metrics.total_frustration_flags },
    { label: "Attention Drops", value: report.metrics.total_attention_drops },
    { label: "Idle Time Ratio", value: `${report.metrics.idle_time_ratio_pct.toFixed(1)}%` },
    { label: "Avg Fatigue", value: report.metrics.avg_fatigue.toFixed(2) },
    { label: "Max Fatigue", value: report.metrics.max_fatigue.toFixed(2) },
  ];

  return (
    <div className="mx-auto max-w-3xl px-4 py-8 sm:px-6">
      
      {/* ==================== CHILD-FACING SECTION ==================== */}
      <div className="glass rounded-3xl p-8 text-center">
        <div className="mx-auto mb-6 grid h-16 w-16 place-items-center rounded-full bg-green-500/20">
          <Sparkles className="h-8 w-8 text-green-600" />
        </div>
        
        <h1 className="font-display text-3xl font-extrabold text-primary">
          {summary.summary.encouragement || "Great work today!"}
        </h1>
        
        <div className="mt-8 space-y-4 text-left">
          <div className="rounded-2xl bg-foreground/5 p-6 space-y-3">
            {summary.summary.bullet_1 && (
              <p className="text-sm text-foreground flex items-start gap-2">
                <span className="text-primary font-bold">•</span> {summary.summary.bullet_1}
              </p>
            )}
            {summary.summary.bullet_2 && (
              <p className="text-sm text-foreground flex items-start gap-2">
                <span className="text-primary font-bold">•</span> {summary.summary.bullet_2}
              </p>
            )}
            {summary.summary.bullet_3 && (
              <p className="text-sm text-foreground flex items-start gap-2">
                <span className="text-primary font-bold">•</span> {summary.summary.bullet_3}
              </p>
            )}
          </div>
        </div>

        {/* Quick Stats */}
        <div className="mt-6 grid grid-cols-3 gap-4">
          <div className="rounded-2xl border border-border bg-white/70 p-4">
            <p className="text-xs text-muted-foreground">Accuracy</p>
            <p className="font-display text-2xl font-bold text-foreground">{summary.stats.accuracy.toFixed(0)}%</p>
          </div>
          <div className="rounded-2xl border border-border bg-white/70 p-4">
            <p className="text-xs text-muted-foreground">Tasks</p>
            <p className="font-display text-2xl font-bold text-foreground">{summary.stats.total_tasks}</p>
          </div>
          <div className="rounded-2xl border border-border bg-white/70 p-4">
            <p className="text-xs text-muted-foreground">Duration</p>
            <p className="font-display text-2xl font-bold text-foreground">{summary.stats.duration_mins}m</p>
          </div>
        </div>
      </div>

      {/* Domain Breakdown Chart */}
      <div className="glass rounded-3xl p-6 mt-6">
        <div className="mb-4 flex items-center justify-between">
          <h3 className="font-display text-lg font-bold">Domain Breakdown</h3>
          <BarChart2 className="h-5 w-5 text-muted-foreground" />
        </div>
        {domainChartData.length > 0 ? (
          <div className="h-[200px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={domainChartData} margin={{ top: 5, right: 5, left: -20, bottom: 5 }}>
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
            No domain data available
          </div>
        )}
      </div>

      {/* ==================== PARENT / TEACHER SECTION ==================== */}
      <div className="mt-6 overflow-hidden rounded-3xl border border-gray-200 bg-white">
        <button
          onClick={() => setIsParentOpen(!isParentOpen)}
          className="flex w-full items-center justify-between bg-gray-50 p-5 text-left transition hover:bg-gray-100"
        >
          <span className="flex items-center gap-2 font-display font-bold text-gray-700">
            <ClipboardCheck className="h-5 w-5 text-gray-500" />
            For Parents & Teachers
          </span>
          <div className="flex items-center gap-2">
            <span className="text-xs font-medium text-gray-400">
              {isParentOpen ? "Hide" : "Show"} Details
            </span>
            {isParentOpen ? <ChevronDown className="h-5 w-5 text-gray-500" /> : <ChevronRight className="h-5 w-5 text-gray-500" />}
          </div>
        </button>

        {isParentOpen && (
          <div className="space-y-6 p-6 transition">
            {/* Clinical Insights */}
            <div className="space-y-4">
              <div>
                <h4 className="text-xs font-semibold uppercase tracking-wider text-gray-500">Clinical Insight</h4>
                <p className="mt-1 text-sm text-gray-800">{report.clinical_insight}</p>
              </div>
              <div>
                <h4 className="text-xs font-semibold uppercase tracking-wider text-gray-500">Clinical Feedback</h4>
                <p className="mt-1 text-sm text-gray-800">{report.clinical_feedback}</p>
              </div>
              <div className="rounded-xl border border-amber-100 bg-amber-50 p-4">
                <h4 className="text-xs font-semibold uppercase tracking-wider text-amber-700">AI Teacher Note</h4>
                <p className="mt-1 text-sm text-amber-900">{summary.teacher_note}</p>
              </div>
            </div>

            {/* Highlights (Best/Weak/Fatigue) */}
            <div className="grid grid-cols-1 gap-4 border-t border-gray-100 pt-6 sm:grid-cols-3">
              <div className="rounded-xl bg-green-50 p-4 text-center">
                <p className="text-xs font-semibold uppercase tracking-wider text-green-600">Strongest Domain</p>
                <p className="mt-1 font-bold text-green-900">{summary.stats.best_domain}</p>
              </div>
              <div className="rounded-xl bg-orange-50 p-4 text-center">
                <p className="text-xs font-semibold uppercase tracking-wider text-orange-600">Needs Work</p>
                <p className="mt-1 font-bold text-orange-900">{summary.stats.weak_domain}</p>
              </div>
              <div className="rounded-xl bg-blue-50 p-4 text-center">
                <p className="text-xs font-semibold uppercase tracking-wider text-blue-600">Fatigue Level</p>
                <p className="mt-1 font-bold text-blue-900">{summary.stats.fatigue_level}</p>
              </div>
            </div>

            {/* Full Metrics Grid */}
            <div className="border-t border-gray-100 pt-6">
              <h4 className="mb-3 text-xs font-semibold uppercase tracking-wider text-gray-500">Full Session Metrics</h4>
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
                {parentMetrics.map((metric) => (
                  <div key={metric.label} className="rounded-lg bg-gray-50 p-3">
                    <p className="text-[10px] uppercase tracking-wide text-gray-500">{metric.label}</p>
                    <p className="mt-1 font-display text-lg font-bold text-gray-900">{metric.value}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* ==================== ACTIONS ==================== */}
      <div className="mt-8 flex flex-col gap-4 sm:flex-row">
        <button
          onClick={() => navigate({ to: "/dashboard" })}
          className="flex flex-1 items-center justify-center gap-2 rounded-full border border-border bg-white/70 px-6 py-3 text-sm font-semibold text-foreground shadow-sm transition hover:bg-foreground/5"
        >
          <Home className="h-4 w-4" /> Back to Dashboard
        </button>
        <button
          onClick={() => navigate({ to: "/session" })}
          className="flex flex-1 items-center justify-center gap-2 rounded-full gradient-primary px-6 py-3 text-sm font-semibold text-primary-foreground shadow-soft transition hover:shadow-lg"
        >
          <RotateCw className="h-4 w-4" /> Start Another Session
        </button>
      </div>
    </div>
  );
}