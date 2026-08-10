from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime, date

# ==================== STUDENT SCHEMAS ====================

class StudentCreate(BaseModel):
    name: str = Field(..., min_length=2)
    age: int = Field(..., ge=3, le=25)
    gender: Optional[str] = None
    diagnosis_level: Optional[str] = None
    notes: Optional[str] = None
    asd_severity: Optional[str] = None
    cognitive_profile: Optional[str] = None
    guardian_name: Optional[str] = None
    guardian_contact: Optional[str] = None

class StudentResponse(BaseModel):
    student_id: str
    registration_date: date
    name: Optional[str] = None
    age: int
    gender: Optional[str] = None
    diagnosis_level: Optional[str] = None
    notes: Optional[str] = None
    asd_severity: Optional[str] = None
    cognitive_profile: Optional[str] = None
    guardian_name: Optional[str] = None
    guardian_contact: Optional[str] = None

    class Config:
        from_attributes = True

# ==================== SESSION SCHEMAS ====================

class SessionCreate(BaseModel):
    student_id: str

class SessionResponse(BaseModel):
    session_id: str
    student_id: str
    session_start: datetime
    session_end: Optional[datetime] = None
    total_steps: Optional[int] = 0
    avg_accuracy: Optional[float] = None
    avg_engagement: Optional[float] = None
    avg_frustration: Optional[float] = None
    avg_fatigue: Optional[float] = None
    domains_visited: Optional[list] = None

    class Config:
        from_attributes = True

# ==================== DOMAIN SCHEMAS ====================

class DomainResponse(BaseModel):
    domain_id: str
    domain_name: str

    class Config:
        from_attributes = True

# ==================== TASK SCHEMAS ====================

class TaskResponse(BaseModel):
    task_id: str
    task_name: str
    domain_id: str
    difficulty_level: Optional[int]
    modality: Optional[str]
    task_type: Optional[str]

    class Config:
        from_attributes = True

# ==================== TASK ATTEMPT SCHEMAS ====================

class TaskAttemptCreate(BaseModel):
    student_id:          str
    session_id:          str
    task_id:             str    
    domain_id:           Optional[str] = "Attention"
    sequence_step:       Optional[int] = 0
    correct:             int
    response_time_ms:    Optional[float] = 0.0
    response_time_sec:   float
    difficulty:          Optional[int] = 3
    engagement_score:    Optional[float] = 0.75
    hints_used:          int = 0
    skips:               int = 0
    frustration_flag:    bool = False
    attention_drop_flag: bool = False
    idle_time_sec:       float = 0.0
    attempt_number:      int = 1
    learning_type:       Optional[str] = None
    task_subtype:           Optional[str] = None
    perceived_difficulty:   Optional[int] = None
    session_fatigue_index:  Optional[float] = 0.0
    domain_switch_count:    Optional[int] = 0
    session_position_norm:  Optional[float] = None
    sensory_load:           Optional[int] = None
    task_source:            Optional[str] = "db"
    cognitive_profile:      Optional[str] = None

class TaskAttemptResponse(BaseModel):
    attempt_id:          int
    student_id:          str
    session_id:          str
    task_id:             str
    sequence_step:       int
    domain_id:           str
    correct:             bool
    response_time_ms:    float
    response_time_sec:   float
    difficulty:          int
    engagement_score:    float
    attempt_number:      int
    hints_used:          int
    skips:               int
    frustration_flag:    bool
    attention_drop_flag: bool
    idle_time_sec:       float
    learning_type:       Optional[str]
    timestamp:           datetime
    task_subtype:           Optional[str] = None
    perceived_difficulty:   Optional[int] = None
    session_fatigue_index:  Optional[float] = 0.0
    domain_switch_count:    Optional[int] = 0
    session_position_norm:  Optional[float] = None
    sensory_load:           Optional[int] = None
    task_source:            Optional[str] = "db"
    cognitive_profile:      Optional[str] = None

    class Config:
        from_attributes = True

# ==================== ADAPTIVE & V3 SCHEMAS ====================

class RegistrationRequest(BaseModel):
    name: str
    age: int = Field(..., ge=3, le=25)
    gender: Optional[str] = None
    diagnosis_level: str
    quiz_answers: dict
    guardian_name: Optional[str] = None
    guardian_contact: Optional[str] = None
    notes: Optional[str] = None

class SessionStatusResponse(BaseModel):
    session_id: str
    should_end: bool
    reason: Optional[str] = None
    steps_completed: int
    fatigue_level: str

class AlertResponse(BaseModel):
    student_id: str
    alert_triggered: bool
    alert_level: Optional[str] = None
    message: Optional[str] = None
    checked_at: str

class KnowledgeStateResponse(BaseModel):
    student_id: str
    knowledge_state: dict
    last_updated: Optional[str] = None

class MetricSummary(BaseModel):
    total_attempts: int
    overall_accuracy_pct: float
    average_response_time_sec: float
    total_frustration_flags: int
    total_attention_drops: int
    total_hints_used: int
    idle_time_ratio_pct: float
    avg_fatigue: Optional[float] = 0.0
    max_fatigue: Optional[float] = 0.0
    cognitive_profile: Optional[str] = None

class DomainMetric(BaseModel):
    attempts: int
    accuracy_pct: float

class NextTaskRequest(BaseModel):
    student_id: str
    session_id: str
    temperature: Optional[float] = 0.4

class SessionReportRequest(BaseModel):
    session_id: str

class SessionReportResponse(BaseModel):
    status: str
    session_id: str
    student_id: str
    metrics: MetricSummary
    domain_breakdown: Dict[str, DomainMetric]
    clinical_insight: str
    clinical_feedback: str