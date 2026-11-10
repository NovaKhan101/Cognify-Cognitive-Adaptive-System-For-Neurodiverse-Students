import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useState, useEffect, useRef, useCallback } from "react";
import { Sparkles, LogOut, Lightbulb, Check, X } from "lucide-react";
import { api, type NextTaskResponse } from "@/lib/api";

export const Route = createFileRoute("/session")({
  head: () => ({
    meta: [
      { title: "Learning Session — Cognify" },
      { name: "description", content: "Adaptive learning session in progress." },
    ],
  }),
  component: Session,
});

// Formulas and constants exactly as specified
const SENSORY_LOAD: Record<string, number> = {
  "Attention": 2, "Memory": 1, "Language": 1, "Executive Function": 2,
  "Social Cognition": 2, "Sensory Processing": 3, "Motor Skills": 1, "Emotional Regulation": 3
};

const computeEngagementScore = (
  hintsUsed: number, 
  frustrationFlag: boolean, 
  attentionDropFlag: boolean, 
  consecutiveCorrect: number
): number => {
  let score = 0.75;
  if (frustrationFlag) score -= 0.12;
  if (attentionDropFlag) score -= 0.08;
  score += consecutiveCorrect * 0.03;
  score += Math.random() * 0.08 - 0.04;
  return Math.max(0.1, Math.min(1.0, score));
};

function Session() {
  const navigate = useNavigate();
  
  // Core state
  const [studentId, setStudentId] = useState<string | null>(null);
  const [cognitiveProfile, setCognitiveProfile] = useState<string>("unknown");
  
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [currentTask, setCurrentTask] = useState<NextTaskResponse["data"] | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedAnswer, setSelectedAnswer] = useState<string | null>(null);
  const [isAnswerLocked, setIsAnswerLocked] = useState(false);

  // Tracking state
  const taskStartTime = useRef<number>(Date.now());
  const lastInteractionTime = useRef<number>(Date.now());
  const hintsUsed = useRef<number>(0);
  const [hintShown, setHintShown] = useState(false);
  
  const attemptNumber = useRef<number>(1);
  const sequenceStep = useRef<number>(1);
  const consecutiveCorrect = useRef<number>(0);
  const domainSwitchCount = useRef<number>(0);
  const previousDomain = useRef<string | null>(null);
  const idleTimeSec = useRef<number>(0);

  // Moved handleEndSession up to avoid any hoisting/linting issues 
  const handleEndSession = useCallback(async () => {
    if (!sessionId) return;
    try {
      await api.endSession(sessionId);
    } catch (err) {
      console.error("Failed to end session properly", err);
    } finally {
      navigate({ to: "/session-complete", search: { session_id: sessionId } });
    }
  }, [sessionId, navigate]);

  // FIX: Wrapped fetchNextTask in useCallback to satisfy exhaustive-deps and prevent stale closures
  const fetchNextTask = useCallback(async (sId: string) => {
    setIsLoading(true);
    setError(null);
    setSelectedAnswer(null);
    setIsAnswerLocked(false);
    setHintShown(false);
    hintsUsed.current = 0;
    taskStartTime.current = Date.now();
    lastInteractionTime.current = Date.now();

    try {
      const res = await api.getNextTask(studentId!, sId);
      setCurrentTask(res.data);
      
      if (previousDomain.current && res.data.predicted_domain !== previousDomain.current) {
        domainSwitchCount.current += 1;
      }
      previousDomain.current = res.data.predicted_domain;
    } catch (err: any) {
      setError(err.message || "Failed to load next task.");
    } finally {
      setIsLoading(false);
    }
  }, [studentId]);

  const handleAnswer = async (option: string) => {
    if (!currentTask || !sessionId || !studentId || isAnswerLocked) return;

    setSelectedAnswer(option);
    setIsAnswerLocked(true);

    const isCorrect = option === currentTask.task.correct_answer;
    const responseTimeMs = Date.now() - taskStartTime.current;
    idleTimeSec.current = (Date.now() - lastInteractionTime.current) / 1000;
    
    const attentionDropFlag = idleTimeSec.current > 5;
    const frustrationFlag = hintsUsed.current >= 2 && !isCorrect;
    const engagementScore = computeEngagementScore(
      hintsUsed.current, frustrationFlag, attentionDropFlag, consecutiveCorrect.current
    );
    const sessionPositionNorm = sequenceStep.current / 20;
    const sessionFatigueIndex = Math.min(
      1.0, 
      (sequenceStep.current / 20) * (cognitiveProfile === "routine-seeking" ? 1.5 : 1.0) * 0.7
    );

    if (isCorrect) {
      consecutiveCorrect.current += 1;
    } else {
      consecutiveCorrect.current = 0;
    }

    const taskId = String(sequenceStep.current);

    const attemptData = {
      student_id: studentId,
      session_id: sessionId,
      task_id: taskId,
      domain_id: currentTask.predicted_domain,
      sequence_step: sequenceStep.current,
      correct: isCorrect ? 1 : 0,
      response_time_ms: responseTimeMs,
      response_time_sec: responseTimeMs / 1000,
      difficulty: currentTask.recommended_difficulty,
      engagement_score: engagementScore,
      hints_used: hintsUsed.current,
      skips: 0,
      frustration_flag: frustrationFlag,
      attention_drop_flag: attentionDropFlag,
      idle_time_sec: idleTimeSec.current,
      attempt_number: attemptNumber.current,
      session_fatigue_index: sessionFatigueIndex,
      perceived_difficulty: currentTask.recommended_difficulty,
      sensory_load: SENSORY_LOAD[currentTask.predicted_domain] || 2,
      domain_switch_count: domainSwitchCount.current,
      session_position_norm: sessionPositionNorm,
      cognitive_profile: cognitiveProfile,
      task_source: currentTask.task?.source || "db"
    };

    try {
      await api.recordAttempt(attemptData);
      
      attemptNumber.current += 1;
      sequenceStep.current += 1;

      setTimeout(async () => {
        if (!sessionId) return;
        
        try {
          const status = await api.getSessionStatus(sessionId);
          if (status.should_end) {
            handleEndSession();
          } else {
            fetchNextTask(sessionId);
          }
        } catch (err) {
          fetchNextTask(sessionId);
        }
      }, 1200);

    } catch (err: any) {
      setError(err.message || "Failed to record attempt.");
      setIsAnswerLocked(false);
    }
  };

  const handleShowHint = () => {
    if (!hintShown) {
      hintsUsed.current += 1;
      setHintShown(true);
    }
  };

  // 0. Load user session from localStorage (Client-side only)
  useEffect(() => {
    const id = localStorage.getItem("cognify_student_id");
    const profile = localStorage.getItem("cognify_cognitive_profile") || "unknown";
    
    if (!id) {
      navigate({ to: "/auth" });
      return;
    }
    
    setStudentId(id);
    setCognitiveProfile(profile);
  }, [navigate]);

  // 1. Initialize Session (Waits until studentId is loaded)
  useEffect(() => {
    if (!studentId) return;

    const initSession = async () => {
      try {
        const sessionRes = await api.startSession(studentId);
        setSessionId(sessionRes.session_id);
        await fetchNextTask(sessionRes.session_id);
      } catch (err: any) {
        setError(err.message || "Failed to start session.");
        setIsLoading(false);
      }
    };

    initSession();
  }, [studentId, navigate, fetchNextTask]); // FIX: Added fetchNextTask to dependencies

  // 2. Track Idle Time
  useEffect(() => {
    const updateInteraction = () => { lastInteractionTime.current = Date.now(); };
    window.addEventListener("mousemove", updateInteraction);
    window.addEventListener("mousedown", updateInteraction);
    window.addEventListener("keydown", updateInteraction);
    return () => {
      window.removeEventListener("mousemove", updateInteraction);
      window.removeEventListener("mousedown", updateInteraction);
      window.removeEventListener("keydown", updateInteraction);
    };
  }, []);

  // 3. Poll Session Status Every 30s
  useEffect(() => {
    if (!sessionId) return;
    const interval = setInterval(async () => {
      try {
        const status = await api.getSessionStatus(sessionId);
        if (status.should_end) {
          handleEndSession();
        }
      } catch (err) {
        console.error("Failed to poll session status");
      }
    }, 30000);
    return () => clearInterval(interval);
  }, [sessionId, handleEndSession]);

  // ==================== LOADING STATE ====================
  if (isLoading && !currentTask) {
    return (
      <div className="flex min-h-[80vh] items-center justify-center">
        <div className="flex flex-col items-center gap-3">
          <Sparkles className="h-8 w-8 animate-pulse text-primary" />
          <p className="text-sm text-muted-foreground">Preparing your next task...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex min-h-[80vh] items-center justify-center text-center">
        <div>
          <p className="text-red-500">{error}</p>
          <button onClick={() => navigate({ to: "/dashboard" })} className="mt-4 text-sm text-primary underline">
            Return to Dashboard
          </button>
        </div>
      </div>
    );
  }

  if (!currentTask) return null;

  // UI Adaptation flags
  const ui = currentTask.ui_adaptation || {};
  const showEncouragement = ui.encouragement_frequency === "high" || (selectedAnswer !== null && currentTask.task.correct_answer === selectedAnswer);

  return (
    <div className={`mx-auto max-w-3xl px-4 py-8 sm:px-6 ${ui.color_mode === "calm" ? "opacity-90 saturate-50" : ""}`}>
      {/* Top Bar */}
      <div className="mb-8 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="grid h-8 w-8 place-items-center rounded-lg gradient-primary shadow-soft">
            <Sparkles className="h-4 w-4 text-primary-foreground" />
          </span>
          <span className="font-display text-sm font-bold capitalize">{currentTask.predicted_domain}</span>
          <span className="rounded-full bg-foreground/5 px-2 py-0.5 text-xs text-muted-foreground">
            Difficulty: {currentTask.recommended_difficulty}
          </span>
        </div>
        <button
          onClick={handleEndSession}
          className="flex items-center gap-2 rounded-full border border-border bg-white/70 px-4 py-2 text-xs font-medium text-muted-foreground transition hover:bg-red-50 hover:text-red-600"
        >
          <LogOut className="h-3 w-3" /> End Session
        </button>
      </div>

      {/* Task Card */}
      <div className={`glass rounded-3xl p-8 transition ${ui.simplify_layout ? "simple-layout" : ""}`}>
        <h1 className="font-display text-2xl font-extrabold leading-tight">
          {currentTask.task.instruction}
        </h1>

        {hintShown && (
          <div className="mt-4 flex items-start gap-3 rounded-2xl bg-amber-100/50 p-4 text-sm text-amber-800">
            <Lightbulb className="h-4 w-4 mt-0.5 shrink-0" />
            <p>{currentTask.task.hint}</p>
          </div>
        )}

        {/* Options Grid */}
        <div className="mt-8 grid grid-cols-1 gap-4 sm:grid-cols-2">
          {currentTask.task.options.map((option, idx) => {
            let buttonClass = "border-border bg-white/70 hover:bg-foreground/5 text-foreground";
            
            if (isAnswerLocked) {
              if (option === currentTask.task.correct_answer) {
                buttonClass = "border-green-500 bg-green-50 text-green-700";
              } else if (option === selectedAnswer) {
                buttonClass = "border-red-500 bg-red-50 text-red-700";
              } else {
                buttonClass = "border-border bg-white/50 opacity-50";
              }
            }

            return (
              <button
                key={idx}
                onClick={() => handleAnswer(option)}
                disabled={isAnswerLocked}
                className={`flex items-center justify-between rounded-2xl border-2 px-5 py-4 text-left font-medium transition ${buttonClass} ${
                  isAnswerLocked ? "cursor-not-allowed" : ""
                }`}
              >
                <span>{option}</span>
                {isAnswerLocked && option === currentTask.task.correct_answer && <Check className="h-5 w-5" />}
                {isAnswerLocked && option === selectedAnswer && option !== currentTask.task.correct_answer && <X className="h-5 w-5" />}
              </button>
            );
          })}
        </div>

        {/* Hint & Encouragement Row */}
        <div className="mt-6 flex min-h-[40px] items-center justify-between">
          {!isAnswerLocked ? (
            <button
              onClick={handleShowHint}
              disabled={hintShown}
              className="flex items-center gap-2 text-sm font-medium text-muted-foreground transition hover:text-primary disabled:opacity-50"
            >
              <Lightbulb className="h-4 w-4" /> {hintShown ? "Hint shown" : "Show hint"}
            </button>
          ) : (
            <div className="text-sm font-medium text-muted-foreground">
              {selectedAnswer === currentTask.task.correct_answer ? (
                <span className="text-green-600">Correct!</span>
              ) : (
                <span className="text-red-600">Not quite right.</span>
              )}
            </div>
          )}

          {showEncouragement && isAnswerLocked && (
            <div className="animate-pulse-soft text-sm font-semibold text-primary">
              {currentTask.task.encouragement}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}