import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { useState, useEffect } from "react";
import { Sparkles, ChevronRight, ArrowLeft, Check } from "lucide-react";
import { z } from "zod";
import { api, type StudentListItem, type QuizQuestion, type UserProfile } from "@/lib/api";

const search = z.object({
  mode: z.enum(["login", "signup"]).optional(),
});

export const Route = createFileRoute("/auth")({
  validateSearch: search,
  head: () => ({
    meta: [
      { title: "Welcome to Cognify" },
      { name: "description", content: "Set up your personalized learning profile." },
    ],
  }),
  component: Auth,
});

// Helper to save user session
const saveSession = (profile: UserProfile) => {
  localStorage.setItem("cognify_student_id", profile.student_id);
  localStorage.setItem("cognify_student_name", profile.name || "Learner");
  localStorage.setItem("cognify_cognitive_profile", profile.cognitive_profile || "unknown");
};

function Auth() {
  const { mode: initial } = Route.useSearch();
  const [mode, setMode] = useState<"login" | "signup">(initial ?? "login");
  
  return (
    <div className="mx-auto flex min-h-[80vh] max-w-md items-center px-4 pt-16 sm:px-6">
      <div className="glass w-full rounded-3xl p-8">
        <div className="mb-6 flex items-center gap-2">
          <span className="grid h-9 w-9 place-items-center rounded-xl gradient-primary shadow-soft">
            <Sparkles className="h-4 w-4 text-primary-foreground" />
          </span>
          <span className="font-display text-lg font-extrabold">Cognify</span>
        </div>

        {mode === "login" ? (
          <LoginView setMode={setMode} />
        ) : (
          <SignupView setMode={setMode} />
        )}
      </div>
    </div>
  );
}

// ================= LOGIN VIEW =================
function LoginView({ setMode }: { setMode: (m: "login" | "signup") => void }) {
  const navigate = useNavigate();
  const [students, setStudents] = useState<StudentListItem[]>([]);
  const [selectedId, setSelectedId] = useState<string>("");
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.getStudents()
      .then(data => {
        setStudents(data);
        if (data.length > 0) setSelectedId(data[0].student_id);
      })
      .catch(err => setError(err.message))
      .finally(() => setIsLoading(false));
  }, []);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedId) return;
    setIsLoading(true);
    setError(null);
    try {
      const profile = await api.getProfile(selectedId);
      saveSession(profile);
      navigate({ to: "/dashboard" });
    } catch (err: any) {
      setError(err.message || "Failed to log in.");
      setIsLoading(false);
    }
  };

  if (isLoading && students.length === 0) {
    return <div className="py-8 text-center text-muted-foreground">Loading students...</div>;
  }

  return (
    <>
      <h1 className="font-display text-2xl font-extrabold">Welcome Back</h1>
      <p className="mt-1 text-sm text-muted-foreground">Select your profile to continue learning.</p>

      {students.length === 0 ? (
        <div className="mt-8 rounded-2xl bg-foreground/5 p-6 text-center">
          <p className="text-sm text-muted-foreground">No students registered yet.</p>
          <button onClick={() => setMode("signup")} className="mt-4 text-sm font-semibold text-primary">
            Set up a Learning Profile →
          </button>
        </div>
      ) : (
        <form onSubmit={handleLogin} className="mt-6 space-y-4">
          <label className="block">
            <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Student Profile</span>
            <select 
              value={selectedId} 
              onChange={(e) => setSelectedId(e.target.value)}
              className="mt-1 w-full rounded-2xl border border-border bg-white/70 px-4 py-3 text-sm outline-none focus:border-primary"
            >
              {students.map(s => (
                <option key={s.student_id} value={s.student_id}>
                  {s.name} {s.cognitive_profile ? `(${s.cognitive_profile})` : ""}
                </option>
              ))}
            </select>
          </label>

          {error && <p className="text-sm text-red-500">{error}</p>}

          <button 
            type="submit" 
            disabled={isLoading}
            className="flex w-full items-center justify-center gap-2 rounded-full gradient-primary px-6 py-3 text-sm font-semibold text-primary-foreground shadow-soft disabled:opacity-50"
          >
            {isLoading ? "Loading..." : "Continue Learning"}
            {!isLoading && <ChevronRight className="h-4 w-4" />}
          </button>
        </form>
      )}

      <p className="mt-6 text-center text-xs text-muted-foreground">
        New to Cognify?{" "}
        <button onClick={() => setMode("signup")} className="font-semibold text-primary">
          Set up Learning Profile
        </button>
      </p>
    </>
  );
}

// ================= SIGNUP VIEW =================
function SignupView({ setMode }: { setMode: (m: "login" | "signup") => void }) {
  const navigate = useNavigate();
  const [step, setStep] = useState(1);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Step 1 State
  const [name, setName] = useState("");
  const [age, setAge] = useState("");
  const [gender, setGender] = useState("");
  const [diagnosisLevel, setDiagnosisLevel] = useState("Level 1");
  const [guardianName, setGuardianName] = useState("");
  const [guardianContact, setGuardianContact] = useState("");

  // Step 2 State
  const [quizQuestions, setQuizQuestions] = useState<QuizQuestion[]>([]);
  const [quizAnswers, setQuizAnswers] = useState<Record<string, boolean>>({});
  const [currentQ, setCurrentQ] = useState(0);

  // Step 3 State
  const [profile, setProfile] = useState<UserProfile | null>(null);

  const handleStep1Submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!name || !age) return;
    setIsLoading(true);
    setError(null);
    try {
      const quiz = await api.getQuiz();
      setQuizQuestions(quiz.questions);
      setStep(2);
    } catch (err: any) {
      setError("Failed to load quiz. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleQuizAnswer = async (answer: boolean) => {
    const questionId = quizQuestions[currentQ].question_id;
    const newAnswers = { ...quizAnswers, [questionId]: answer };
    setQuizAnswers(newAnswers);

    if (currentQ < quizQuestions.length - 1) {
      setCurrentQ(currentQ + 1);
    } else {
      // Last question — submit registration
      setIsLoading(true);
      setError(null);
      try {
        const result = await api.registerStudent({
          name,
          age: Number(age),
          gender: gender || undefined,
          diagnosis_level: diagnosisLevel as "Level 1" | "Level 2" | "Level 3",
          quiz_answers: newAnswers,
          guardian_name: guardianName || undefined,
          guardian_contact: guardianContact || undefined,
        });

        const fullProfile = await api.getProfile(result.student_id);
        setProfile(fullProfile);
        setStep(3);
      } catch (err: any) {
        setError(err.message || "Registration failed.");
      } finally {
        setIsLoading(false);
      }
    }
  };

  const finishSignup = () => {
    if (profile) {
      saveSession(profile);
      navigate({ to: "/dashboard" });
    }
  };

  return (
    <>
      {/* Progress Dots */}
      <div className="mb-6 flex items-center justify-center gap-2">
        {[1, 2, 3].map((s) => (
          <div 
            key={s} 
            className={`h-2 w-2 rounded-full transition ${step >= s ? "bg-primary" : "bg-foreground/20"}`}
          />
        ))}
      </div>

      {error && <p className="mb-4 text-sm text-red-500">{error}</p>}

      {step === 1 && (
        <>
          <h1 className="font-display text-2xl font-extrabold">Student Information</h1>
          <p className="mt-1 text-sm text-muted-foreground">Tell us about the learner.</p>
          
          <form onSubmit={handleStep1Submit} className="mt-6 space-y-4">
            <label className="block">
              <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Child's Name</span>
              <input required value={name} onChange={(e) => setName(e.target.value)} className="mt-1 w-full rounded-2xl border border-border bg-white/70 px-4 py-3 text-sm outline-none focus:border-primary" />
            </label>
            
            <div className="grid grid-cols-2 gap-4">
              <label className="block">
                <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Age</span>
                <input required type="number" min="3" max="25" value={age} onChange={(e) => setAge(e.target.value)} className="mt-1 w-full rounded-2xl border border-border bg-white/70 px-4 py-3 text-sm outline-none focus:border-primary" />
              </label>
              <label className="block">
                <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Gender</span>
                <select value={gender} onChange={(e) => setGender(e.target.value)} className="mt-1 w-full rounded-2xl border border-border bg-white/70 px-4 py-3 text-sm outline-none focus:border-primary">
                  <option value="">Prefer not to say</option>
                  <option value="Male">Male</option>
                  <option value="Female">Female</option>
                  <option value="Other">Other</option>
                </select>
              </label>
            </div>

            <label className="block">
              <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">ASD Diagnosis Level</span>
              <select value={diagnosisLevel} onChange={(e) => setDiagnosisLevel(e.target.value)} className="mt-1 w-full rounded-2xl border border-border bg-white/70 px-4 py-3 text-sm outline-none focus:border-primary">
                <option value="Level 1">Level 1 (Requires Support)</option>
                <option value="Level 2">Level 2 (Requires Substantial Support)</option>
                <option value="Level 3">Level 3 (Requires Very Substantial Support)</option>
              </select>
            </label>

            <button 
              type="submit" 
              disabled={isLoading}
              className="flex w-full items-center justify-center gap-2 rounded-full gradient-primary px-6 py-3 text-sm font-semibold text-primary-foreground shadow-soft disabled:opacity-50"
            >
              {isLoading ? "Loading..." : "Next: Learning Quiz"}
              {!isLoading && <ChevronRight className="h-4 w-4" />}
            </button>
          </form>
        </>
      )}

      {step === 2 && quizQuestions.length > 0 && (
        <>
          {currentQ > 0 && (
            <button onClick={() => setCurrentQ(currentQ - 1)} className="mb-4 flex items-center text-xs text-muted-foreground hover:text-primary">
              <ArrowLeft className="h-3 w-3 mr-1" /> Back
            </button>
          )}
          
          <div className="mb-2 text-xs font-semibold uppercase tracking-wider text-primary">
            Question {currentQ + 1} of {quizQuestions.length}
          </div>
          
          <h1 className="font-display text-2xl font-extrabold leading-tight">
            {quizQuestions[currentQ].question}
          </h1>
          
          {/* Progress dots for quiz */}
          <div className="mt-4 mb-8 flex items-center justify-center gap-1.5">
            {quizQuestions.map((_, idx) => (
              <div 
                key={idx} 
                className={`h-1.5 w-1.5 rounded-full transition ${idx <= currentQ ? "bg-primary" : "bg-foreground/20"}`}
              />
            ))}
          </div>

          <div className="mt-6 grid grid-cols-2 gap-4">
            <button 
              onClick={() => handleQuizAnswer(true)}
              disabled={isLoading}
              className="rounded-2xl border-2 border-green-500/50 bg-green-500/10 py-6 text-lg font-bold text-green-700 transition hover:bg-green-500/20 disabled:opacity-50"
            >
              Yes
            </button>
            <button 
              onClick={() => handleQuizAnswer(false)}
              disabled={isLoading}
              className="rounded-2xl border-2 border-red-500/50 bg-red-500/10 py-6 text-lg font-bold text-red-700 transition hover:bg-red-500/20 disabled:opacity-50"
            >
              No
            </button>
          </div>

          {isLoading && <p className="mt-6 text-center text-sm text-muted-foreground">Analyzing answers...</p>}
        </>
      )}

      {step === 3 && profile && (
        <div className="text-center">
          <div className="mx-auto mb-6 grid h-16 w-16 place-items-center rounded-full bg-green-500/20">
            <Check className="h-8 w-8 text-green-600" />
          </div>
          
          <h1 className="font-display text-2xl font-extrabold">Profile Ready!</h1>
          
          <div className="mt-6 rounded-2xl bg-foreground/5 p-6">
            <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Learning Profile</span>
            <h2 className="mt-1 font-display text-xl font-bold text-primary">
              {profile.profile_friendly_name}
            </h2>
            <p className="mt-2 text-sm text-muted-foreground">
              {profile.name}'s learning journey is now personalized.
            </p>
          </div>

          <button 
            onClick={finishSignup}
            className="mt-8 flex w-full items-center justify-center gap-2 rounded-full gradient-primary px-6 py-3 text-sm font-semibold text-primary-foreground shadow-soft"
          >
            Start Learning Journey
            <ChevronRight className="h-4 w-4" />
          </button>
        </div>
      )}

      {step === 1 && (
        <p className="mt-6 text-center text-xs text-muted-foreground">
          Already have a profile?{" "}
          <button onClick={() => setMode("login")} className="font-semibold text-primary">
            Log in
          </button>
        </p>
      )}
    </>
  );
}