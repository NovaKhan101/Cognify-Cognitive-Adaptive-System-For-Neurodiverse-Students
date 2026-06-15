from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime, date

# ==================== STUDENT SCHEMAS ====================

class StudentCreate(BaseModel):
    age: int = Field(..., ge=3, le=18)
    gender: Optional[str] = None
    diagnosis_level: Optional[str] = None
    notes: Optional[str] = None

class StudentResponse(BaseModel):
    student_id: str
    registration_date: date
    age: int
    gender: Optional[str]
    diagnosis_level: Optional[str]
    notes: Optional[str]

    class Config:
        from_attributes = True

# ==================== SESSION SCHEMAS ====================

class SessionCreate(BaseModel):
    student_id: str

class SessionResponse(BaseModel):
    session_id: str
    student_id: str
    session_start: datetime
    session_end: Optional[datetime]

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
    correct:             int  # Frontend sends 1 or 0
    response_time_ms:    Optional[int] = 0
    response_time_sec:   float
    difficulty:          Optional[int] = 3
    engagement_score:    Optional[float] = 0.75
    hints_used:          int
    skips:               int
    frustration_flag:    bool
    attention_drop_flag: bool
    idle_time_sec:       float
    attempt_number:      int
    learning_type:       Optional[str] = None

class TaskAttemptResponse(BaseModel):
    attempt_id:          int
    student_id:          str
    session_id:          str
    task_id:             str
    sequence_step:       int            # Added to match models.py
    domain_id:           str            # Added to match models.py
    correct:             bool           # Validates the true SQLAlchemy Boolean type cleanly
    response_time_ms:    int            # Added to match models.py
    response_time_sec:   float
    difficulty:          int            # Added to match models.py
    engagement_score:    float          # Added to match models.py
    attempt_number:      int
    hints_used:          int
    skips:               int
    frustration_flag:    bool
    attention_drop_flag: bool
    idle_time_sec:       float
    learning_type:       Optional[str]
    timestamp:           datetime

    class Config:
        from_attributes = True