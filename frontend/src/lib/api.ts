const API_BASE_URL = "http://localhost:8000";

// Helper to handle JSON responses
async function fetchAPI<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
    ...options,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(errorData.detail || `API Error: ${response.statusText}`);
  }

  return response.json() as Promise<T>;
}

// ==================== TYPES ====================
export interface QuizQuestion {
  question_id: string;
  question: string;
}

export interface RegistrationData {
  name: string;
  age: number;
  gender?: string;
  diagnosis_level: "Level 1" | "Level 2" | "Level 3";
  quiz_answers: Record<string, boolean>;
  guardian_name?: string;
  guardian_contact?: string;
  notes?: string;
}

// Matches FastAPI StudentResponse (lighter schema, no mastery scores)
export interface StudentListItem {
  student_id: string;
  registration_date: string;
  name?: string;
  age: number;
  gender?: string;
  diagnosis_level?: string;
  notes?: string;
  asd_severity?: string;
  cognitive_profile?: string;
  guardian_name?: string;
  guardian_contact?: string;
}

// Matches FastAPI /students/{id}/profile (includes system mode + mastery)
export interface UserProfile extends StudentListItem {
  profile_friendly_name: string;
  system_mode: string;
  total_attempts: number;
  knowledge_states: Record<string, number>; // PLURAL (used in profile endpoint)
  registered_at: string;
}

export interface TaskContent {
  title: string;
  instruction: string;
  options: string[];
  correct_answer: string;
  hint: string;
  encouragement: string;
  source: string;
}

export interface UIAdaptation {
  reduce_visual_stimuli: boolean;
  hide_timer: boolean;
  encouragement_frequency: "high" | "normal";
  color_mode: "calm" | "normal";
  simplify_layout: boolean;
}

export interface StudentContext {
  cognitive_profile: string;
  avg_fatigue: number;
  avg_perceived_difficulty: number;
  avg_sensory_load: number;
  domain_switch_count: number;
  recent_accuracy: number;
  frustration_rate: number;
  attention_drop_rate: number;
  consecutive_correct: number;
  session_position: number;
  weak_domains: string[];
  asd_severity?: string;
  domain?: string;
  difficulty?: number;
}

export interface NextTaskResponse {
  status: string;
  data: {
    predicted_domain: string;
    recommended_difficulty: number;
    model_used: string;
    task: TaskContent;
    ui_adaptation: UIAdaptation;
    system_mode: string;
    student_context: StudentContext;
  };
}

export interface AlertResponse {
  student_id: string;
  alert_triggered: boolean;
  alert_level: string | null;
  message: string | null;
  checked_at: string;
}

export interface SessionSummaryResponse {
  session_id: string;
  student_name: string;
  generated_at: string;
  summary: {
    bullet_1: string;
    bullet_2: string;
    bullet_3: string;
    encouragement: string;
  };
  teacher_note: string;
  stats: {
    accuracy: number;
    total_tasks: number;
    duration_mins: number;
    best_domain: string;
    weak_domain: string;
    fatigue_level: string;
  };
  source: string;
}

// Matches FastAPI MetricSummary schema
export interface SessionMetrics {
  total_attempts: number;
  overall_accuracy_pct: number;
  average_response_time_sec: number;
  total_frustration_flags: number;
  total_attention_drops: number;
  total_hints_used: number;
  idle_time_ratio_pct: number;
  avg_fatigue: number;
  max_fatigue: number;
  cognitive_profile: string;
}

// Matches FastAPI DomainMetric schema
export interface DomainMetric {
  attempts: number;
  accuracy_pct: number;
}

// Matches FastAPI SessionReportResponse schema
export interface SessionReportResponse {
  status: string;
  session_id: string;
  student_id: string;
  metrics: SessionMetrics;
  domain_breakdown: Record<string, DomainMetric>;
  clinical_insight: string;
  clinical_feedback: string;
}

export interface StudentProgress {
  student_id: string;
  total_attempts: number;
  domain_mastery: Record<string, number>;
}

export interface AnalyticsSummary {
  total_students: number;
  total_sessions: number;
  total_attempts: number;
  total_tasks: number;
}

// Matches FastAPI TaskAttemptCreate schema
export interface AttemptData {
  student_id: string;
  session_id: string;
  task_id: string;
  domain_id: string;
  sequence_step: number;
  correct: number; // 1 or 0
  response_time_ms: number;
  response_time_sec: number;
  difficulty: number;
  engagement_score: number;
  hints_used: number;
  skips: number;
  frustration_flag: boolean;
  attention_drop_flag: boolean;
  idle_time_sec: number;
  attempt_number: number;
  learning_type?: string;
  task_subtype?: string;
  perceived_difficulty?: number;
  session_fatigue_index?: number;
  domain_switch_count?: number;
  session_position_norm?: number;
  sensory_load?: number;
  task_source?: string;
  cognitive_profile?: string;
}

// ==================== API CALLS ====================
export const api = {
  // -------------------- Registration & Auth --------------------
  getQuiz: () => 
    fetchAPI<{ questions: QuizQuestion[] }>("/registration/quiz"),
  
  registerStudent: (data: RegistrationData) => 
    fetchAPI<{ success: boolean; student_id: string; name: string; cognitive_profile: string }>("/registration/register", {
      method: "POST",
      body: JSON.stringify(data),
    }),

  // Mock login: Verifies student exists by fetching profile
  loginWithStudentId: (studentId: string) => 
    fetchAPI<UserProfile>(`/students/${studentId}/profile`),

  getProfile: (studentId: string) => 
    fetchAPI<UserProfile>(`/students/${studentId}/profile`),

  // Plain student record (matches StudentListItem type)
  getStudent: (studentId: string) => 
    fetchAPI<StudentListItem>(`/students/${studentId}`),

  getStudents: () => 
    fetchAPI<StudentListItem[]>("/students"),

  // -------------------- Sessions --------------------
  startSession: (studentId: string) => 
    fetchAPI<{ session_id: string; student_id: string }>("/sessions", {
      method: "POST",
      body: JSON.stringify({ student_id: studentId }),
    }),

  endSession: (sessionId: string) => 
    fetchAPI<{ session: any; summary: SessionSummaryResponse }>(`/sessions/${sessionId}/end`, {
      method: "POST",
    }),

  // -------------------- Adaptive Learning --------------------
  recordAttempt: (attemptData: AttemptData) => 
    fetchAPI<any>("/attempts", {
      method: "POST",
      body: JSON.stringify(attemptData),
    }),

  getNextTask: (studentId: string, sessionId: string, temperature?: number) => 
    fetchAPI<NextTaskResponse>("/adaptive/next-task", {
      method: "POST",
      body: JSON.stringify({ 
        student_id: studentId, 
        session_id: sessionId,
        temperature: temperature // Added here
      }),
    }),

  // Rich analytics endpoint (charts, metrics, clinical insights)
  getSessionReport: (sessionId: string) => 
    fetchAPI<SessionReportResponse>("/adaptive/session-report", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId }),
    }),

  // -------------------- Dashboard & Analytics --------------------
  
  // NOTE: This endpoint specifically returns `knowledge_state` (SINGULAR) 
  // as per the FastAPI KnowledgeStateResponse schema.
  getKnowledgeState: (studentId: string) => 
    fetchAPI<{ student_id: string; knowledge_state: Record<string, number> }>(`/adaptive/knowledge-state/${studentId}`),

  // Domain mastery percentages per student
  getStudentProgress: (studentId: string) => 
    fetchAPI<StudentProgress>(`/adaptive/student-progress/${studentId}`),

  getSessionStatus: (sessionId: string) => 
    fetchAPI<{ session_id: string; should_end: boolean; reason: string | null; steps_completed: number; fatigue_level: string }>(`/adaptive/session-status/${sessionId}`),

  getAlerts: (studentId: string) => 
    fetchAPI<AlertResponse>(`/adaptive/alerts/${studentId}`),

  // LLM-generated 3-bullet-point summary
  getSessionSummary: (sessionId: string) => 
    fetchAPI<SessionSummaryResponse>(`/adaptive/session-summary/${sessionId}`),

  getStudentContext: (studentId: string) => 
    fetchAPI<{ student_id: string; context: StudentContext }>(`/adaptive/student-context/${studentId}`),

  getAnalyticsSummary: () => 
    fetchAPI<AnalyticsSummary>("/analytics/summary"),
};