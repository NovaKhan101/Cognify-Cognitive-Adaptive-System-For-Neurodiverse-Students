"""
models.py
Cognify — SQLAlchemy ORM models
Updated: added sequence_step, response_time_ms, domain_id, engagement_score, difficulty, task_subtype
Added: GeneratedTask, StudentTaskHistory models
V3 Update: Added behavioral columns, KnowledgeState, Recommendation, TaskBank tables
Run ALTER TABLE migrations on Neon before restarting backend (see README).
"""

import uuid
from sqlalchemy import Column, String, Integer, Float, Boolean, Date, DateTime, ForeignKey, Numeric, Text, JSON  
from sqlalchemy.sql import func  
from database import Base

class Student(Base):  
    __tablename__ = "students"

    student_id          = Column(String(25), primary_key=True)  
    registration_date   = Column(Date, nullable=False, server_default=func.current_date())  
    age                 = Column(Integer, nullable=False)  
    gender              = Column(String(10))  
    diagnosis_level     = Column(String(20))  
    notes               = Column(Text)
    # V3 ADDITIONS:
    name                = Column(String(100))  
    asd_severity        = Column(String(20))  
    cognitive_profile   = Column(String(30))  
    guardian_name       = Column(String(100))  
    guardian_contact    = Column(String(50))  
    updated_at          = Column(DateTime, server_default=func.now())

class SessionModel(Base):  
    __tablename__ = "sessions"

    session_id          = Column(String(30), primary_key=True)  
    student_id          = Column(String(25), ForeignKey("students.student_id", ondelete="CASCADE"), nullable=False)  
    session_start       = Column(DateTime, nullable=False, server_default=func.now())  
    session_end         = Column(DateTime)
    # V3 ADDITIONS:
    total_steps         = Column(Integer, default=0)  
    avg_accuracy        = Column(Float)  
    avg_engagement      = Column(Float)  
    avg_frustration     = Column(Float)  
    avg_fatigue         = Column(Float)  
    domains_visited     = Column(JSON)  
    updated_at          = Column(DateTime, server_default=func.now())

class TaskDomain(Base):  
    __tablename__ = "task_domains"

    domain_id   = Column(String(20), primary_key=True)  
    domain_name = Column(String(50), nullable=False)

class Skill(Base):  
    __tablename__ = "skills"

    skill_id         = Column(String(50), primary_key=True)  
    domain_id        = Column(String(20), ForeignKey("task_domains.domain_id"), nullable=False)  
    skill_name       = Column(String(100), nullable=False)  
    difficulty_level = Column(Integer)

class Task(Base):  
    __tablename__ = "tasks"

    task_id          = Column(String(30), primary_key=True)  
    task_name        = Column(String(100), nullable=False)  
    domain_id        = Column(String(20), ForeignKey("task_domains.domain_id"), nullable=False)  
    difficulty_level = Column(Integer)  
    modality         = Column(String(15))  
    task_type        = Column(String(30))

class TaskSkill(Base):  
    __tablename__ = "task_skills"

    task_id    = Column(String(30), ForeignKey("tasks.task_id", ondelete="CASCADE"), primary_key=True)  
    skill_id   = Column(String(50), ForeignKey("skills.skill_id", ondelete="CASCADE"), primary_key=True)  
    weight     = Column(Numeric(3, 2), default=1.0)  
    is_primary = Column(Boolean, default=True)

class TaskAttempt(Base):  
    __tablename__ = "task_attempts"

    attempt_id  = Column(Integer, primary_key=True, autoincrement=True)  
    student_id  = Column(String(25), ForeignKey("students.student_id", ondelete="CASCADE"), nullable=False)  
    session_id  = Column(String(30), ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False)

    task_id     = Column(String(30), nullable=False)
    sequence_step = Column(Integer, default=0)

    # Performance  
    correct           = Column(Boolean, nullable=False)  
    response_time_ms  = Column(Integer, default=0)  
    response_time_sec = Column(Float, nullable=False)  
    attempt_number    = Column(Integer, default=1)  
    hints_used        = Column(Integer, default=0)  
    skips             = Column(Integer, default=0)  
    difficulty        = Column(Integer, default=3)

    # Behavioral  
    frustration_flag    = Column(Boolean, default=False)  
    attention_drop_flag = Column(Boolean, default=False)  
    idle_time_sec       = Column(Float, default=0.0)  
    engagement_score    = Column(Float, default=0.75)  
    domain_id           = Column(String(20), default="attention")  
    learning_type       = Column(String(50))
    task_subtype        = Column(String(50))

    # Metadata  
    timestamp = Column(DateTime, nullable=False, server_default=func.now())

    # V3 ADDITIONS:
    perceived_difficulty    = Column(Integer)  
    session_fatigue_index   = Column(Float, default=0.0)  
    domain_switch_count     = Column(Integer, default=0)  
    session_position_norm   = Column(Float)  
    sensory_load            = Column(Integer)  
    task_source             = Column(String(20), default="db")  
    gemini_prompt_context   = Column(Text)  
    cognitive_profile       = Column(String(30))

class GeneratedTask(Base):  
    __tablename__ = "generated_tasks"

    id           = Column(Integer, primary_key=True, autoincrement=True)  
    domain       = Column(String(50), nullable=False)  
    difficulty   = Column(Integer, nullable=False)  
    task_subtype = Column(String(50))  
    question     = Column(Text, nullable=False)  
    visual       = Column(Text)  
    hint         = Column(Text, nullable=False)  
    options      = Column(JSON, nullable=False)  
    source       = Column(String(20), default="gemini")  
    created_at   = Column(DateTime, server_default=func.now())

class StudentTaskHistory(Base):  
    __tablename__ = "student_task_history"

    id         = Column(Integer, primary_key=True, autoincrement=True)  
    student_id = Column(String(25), nullable=False)  
    task_db_id = Column(Integer, nullable=False)  
    seen_at    = Column(DateTime, server_default=func.now())

# V3 ADDITIONS:
class KnowledgeState(Base):  
    __tablename__ = "knowledge_states"

    id              = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))  
    student_id      = Column(String(25), ForeignKey("students.student_id", ondelete="CASCADE"), nullable=False)  
    session_id      = Column(String(30))  
    domain_id       = Column(String(50), nullable=False)  
    domain_name     = Column(String(50))  
    mastery_score   = Column(Float)  
    attempts_count  = Column(Integer, default=0)  
    last_updated    = Column(DateTime, server_default=func.now())

class Recommendation(Base):  
    __tablename__ = "recommendations"

    id                      = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))  
    student_id              = Column(String(25), ForeignKey("students.student_id", ondelete="CASCADE"), nullable=False)  
    session_id              = Column(String(30))  
    recommended_domain      = Column(String(50))  
    recommended_difficulty  = Column(Integer)  
    recommended_task_id     = Column(String(30))  
    task_source             = Column(String(20))  
    domain_confidence       = Column(Float)  
    difficulty_confidence   = Column(Float)  
    task_confidence         = Column(Float)  
    performance_pred        = Column(Float)  
    student_context         = Column(JSON)  
    created_at              = Column(DateTime, server_default=func.now())

class TaskBank(Base):  
    __tablename__ = "task_bank"

    id                  = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))  
    domain_name         = Column(String(50), nullable=False)  
    difficulty          = Column(Integer)  
    cognitive_profile   = Column(String(30))  
    title               = Column(String(200))  
    instruction         = Column(Text, nullable=False)  
    options             = Column(JSON)  
    correct_answer      = Column(Text)  
    hint                = Column(Text)  
    encouragement       = Column(Text)  
    sensory_load        = Column(Integer)  
    is_adaptive         = Column(Boolean, default=False)  
    usage_count         = Column(Integer, default=0)  
    created_at          = Column(DateTime, server_default=func.now())