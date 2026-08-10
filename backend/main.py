"""
main.py
Cognify — FastAPI Backend
Run: uvicorn main:app --reload
"""

import os  
import json  
import uuid
import logging  
from datetime import datetime  
from typing import List, Optional, Dict  
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Depends  
from fastapi.middleware.cors import CORSMiddleware  
from sqlalchemy.orm import Session  
from sqlalchemy import text

from database import get_db, engine  
from models import (  
    Base, Student, SessionModel, TaskDomain, Task,  
    TaskAttempt, KnowledgeState, Recommendation, TaskBank  
)  
from schemas import (  
    StudentCreate, StudentResponse,  
    SessionCreate, SessionResponse,  
    TaskAttemptCreate, TaskAttemptResponse,  
    TaskResponse, DomainResponse,  
    RegistrationRequest, SessionStatusResponse,  
    AlertResponse, KnowledgeStateResponse,  
    NextTaskRequest, SessionReportRequest, MetricSummary, 
    DomainMetric, SessionReportResponse
)  
from task_agent import get_task_for_student
from session_summary import generate_session_summary
from knowledge_state_updater import update_knowledge_states
from registration import register_student, get_student_profile, get_quiz_questions, validate_registration_data
from adaptive_logic import (
    check_frustration_trigger_recovery, should_end_session, 
    check_meltdown_alert, get_ui_adaptation_flags, 
    get_diagnostic_domain, get_system_mode
)
from ml_inference_V2 import get_engine

load_dotenv()  
Base.metadata.create_all(bind=engine)

logger = logging.getLogger(__name__)

app = FastAPI(  
    title="Cognify — ASD Adaptive Learning Platform",  
    version="3.0.0",  
)

origins = [  
    "http://localhost:5173",  
    "http://127.0.0.1:5173",
    "http://localhost:8080",
    "http://192.168.1.11:8080"  # <--- ADD THIS EXACT LINE
]  

app.add_middleware(  
    CORSMiddleware,  
    allow_origins=origins,  
    allow_credentials=True,  
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# ── DOMAIN CAPITALIZATION HELPERS ────────────────────────────────────────────  
def _capitalize_domain(domain: str) -> str:  
    if not domain:  
        return "Attention"  
    return " ".join(word.capitalize() for word in domain.split())

def _lowercase_domain(domain: str) -> str:  
    if not domain:  
        return "attention"  
    return domain.lower()

# ── DRY Marshaling Helper ────────────────────────────────────────────────────
def _marshal_attempts(attempts: List[TaskAttempt]) -> List[Dict]:
    """Converts SQLAlchemy TaskAttempt objects into dicts with all V3 fields."""
    return [
        {
            "student_id":                   a.student_id,
            "session_id":                   a.session_id,
            "task_id":                      int(a.task_id) if str(a.task_id).isdigit() else 0,
            "domain_id":                    _capitalize_domain(a.domain_id or "attention"),
            "correct":                      1 if a.correct is True else 0,
            "response_time_ms":             float(a.response_time_ms or 0),
            "difficulty":                   int(a.difficulty or 3),
            "hints_used":                   int(a.hints_used or 0),
            "frustration_flag":             1 if a.frustration_flag is True else 0,
            "attention_drop_flag":          1 if a.attention_drop_flag is True else 0,
            "engagement_score":             float(a.engagement_score or 0.75),
            "task_subtype":                 a.task_subtype,
            "session_fatigue_index":        float(a.session_fatigue_index or 0.0),
            "perceived_difficulty":         int(a.perceived_difficulty or a.difficulty or 3),
            "sensory_load":                 int(a.sensory_load or 2),
            "domain_switch_count":          int(a.domain_switch_count or 0),
            "session_position_normalized":  float(a.session_position_norm or 0.5),
            "cognitive_profile":            a.cognitive_profile or "unknown",
        }
        for a in attempts
    ]

# ── Health ───────────────────────────────────────────────────────────────────  
@app.get("/")  
def root():  
    return {"message": "Cognify API active", "status": "healthy", "engine": "transformer_v2_gated"}

# ── Analytics & Dashboard ───────────────────────────────────────────────────  
@app.get("/analytics/summary")  
def analytics_summary(db: Session = Depends(get_db)):  
    return {  
        "total_students": db.query(Student).count(),  
        "total_sessions": db.query(SessionModel).count(),  
        "total_attempts": db.query(TaskAttempt).count(),  
        "total_tasks":    71,  
    }

# ── Registration & Students ──────────────────────────────────────────────────
@app.get("/registration/quiz")
def get_registration_quiz():
    return {"questions": get_quiz_questions()}

@app.post("/registration/register", status_code=201)
def register_new_student(data: RegistrationRequest, db: Session = Depends(get_db)):
    is_valid, error = validate_registration_data(data.dict())
    if not is_valid:
        raise HTTPException(status_code=400, detail=error)
    result = register_student(data.dict(), db)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["error"])
    return result

@app.get("/students", response_model=List[StudentResponse])  
def get_all_students(db: Session = Depends(get_db)):  
    return db.query(Student).all()

@app.get("/students/{student_id}", response_model=StudentResponse)  
def get_student(student_id: str, db: Session = Depends(get_db)):  
    s = db.query(Student).filter(Student.student_id == student_id).first()  
    if not s:  
        raise HTTPException(status_code=404, detail="Student not found")  
    return s

@app.get("/students/{student_id}/profile")
def get_student_profile_endpoint(student_id: str, db: Session = Depends(get_db)):
    result = get_student_profile(student_id, db)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result

# ── Sessions ─────────────────────────────────────────────────────────────────  
@app.post("/sessions", response_model=SessionResponse, status_code=201)  
def create_session(session: SessionCreate, db: Session = Depends(get_db)):  
    if not db.query(Student).filter(Student.student_id == session.student_id).first():  
        raise HTTPException(status_code=404, detail="Student not found")  
        
    count = db.query(SessionModel).count()  
    candidate_id = f"SES{count + 1:04d}"
    
    # FIX: Fallback to UUID if ID already exists (prevents duplicate key crash)
    if db.query(SessionModel).filter(SessionModel.session_id == candidate_id).first():
        candidate_id = f"SES{uuid.uuid4().hex[:6].upper()}"
        
    db_session = SessionModel(  
        session_id=candidate_id,  
        student_id=session.student_id,  
        session_start=datetime.now(),  
    )  
    db.add(db_session)  
    db.commit()  
    db.refresh(db_session)  
    return db_session

@app.post("/sessions/{session_id}/end")
async def end_session(session_id: str, db: Session = Depends(get_db)):
    s = db.query(SessionModel).filter(SessionModel.session_id == session_id).first()
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    s.session_end = datetime.now()
    db.commit()
    db.refresh(s)
    
    try:
        update_knowledge_states(student_id=s.student_id, session_id=session_id, db=db)
    except Exception as e:
        logger.error(f"Failed to update knowledge states for {session_id}: {e}")
    
    # CraftX call is synchronous, so no 'await' here
    summary = generate_session_summary(session_id, db)
    
    return {
        "session": {
            "session_id": s.session_id, 
            "student_id": s.student_id, 
            "session_end": s.session_end
        }, 
        "summary": summary
    }

# ── Domains & Tasks ──────────────────────────────────────────────────────────  
@app.get("/domains", response_model=List[DomainResponse])  
def get_domains(db: Session = Depends(get_db)):  
    return db.query(TaskDomain).all()

@app.get("/tasks", response_model=List[TaskResponse])  
def get_tasks(db: Session = Depends(get_db)):  
    return db.query(Task).all()

# ── Attempts ─────────────────────────────────────────────────────────────────  
@app.post("/attempts", response_model=TaskAttemptResponse, status_code=201)  
def record_attempt(attempt: TaskAttemptCreate, db: Session = Depends(get_db)):  
    is_correct_bool = True if attempt.correct in [1, True] else False

    db_attempt = TaskAttempt(  
        student_id          = attempt.student_id,  
        session_id          = attempt.session_id,  
        task_id             = str(attempt.task_id),  
        sequence_step       = attempt.sequence_step,  
        correct             = is_correct_bool,  
        response_time_ms    = attempt.response_time_ms,  
        response_time_sec   = attempt.response_time_sec,  
        attempt_number      = attempt.attempt_number,  
        hints_used          = attempt.hints_used,  
        skips               = attempt.skips,  
        difficulty          = attempt.difficulty,  
        frustration_flag    = attempt.frustration_flag,  
        attention_drop_flag = attempt.attention_drop_flag,  
        idle_time_sec       = attempt.idle_time_sec,  
        engagement_score    = attempt.engagement_score,  
        domain_id           = attempt.domain_id,  
        learning_type       = attempt.learning_type,  
        task_subtype        = attempt.task_subtype,
        perceived_difficulty   = attempt.perceived_difficulty,
        session_fatigue_index  = attempt.session_fatigue_index,
        domain_switch_count    = attempt.domain_switch_count,
        session_position_norm  = attempt.session_position_norm,
        sensory_load           = attempt.sensory_load,
        task_source            = attempt.task_source,
        cognitive_profile      = attempt.cognitive_profile,
    )  
    db.add(db_attempt)  
    db.commit()  
    db.refresh(db_attempt)  
    return db_attempt

# ── Adaptive ML Endpoints ────────────────────────────────────────────────────  
@app.post("/adaptive/next-task")  
async def get_next_task(request: NextTaskRequest, db: Session = Depends(get_db)):
    attempts = (
        db.query(TaskAttempt)
        .filter(TaskAttempt.student_id == request.student_id)
        .order_by(TaskAttempt.sequence_step.asc())
        .all()
    )

    attempts_raw = _marshal_attempts(attempts)
    student = db.query(Student).filter(Student.student_id == request.student_id).first()
    
    engine = get_engine()
    student_context = engine.extract_student_context(attempts_raw) if attempts_raw else {}
    student_context["asd_severity"] = student.asd_severity if student else "moderate"
    student_context["cognitive_profile"] = student.cognitive_profile if student else "unknown"

    system_mode = get_system_mode(len(attempts))
    last_5 = attempts_raw[-5:] if len(attempts_raw) >= 5 else attempts_raw
    
    frustrated, recovery_domain = check_frustration_trigger_recovery(last_5)
    if frustrated:
        recommendation = {
            "predicted_domain": recovery_domain,
            "recommended_difficulty": 1,
            "recommended_task_id": 0,
            "domain_confidence": 1.0,
            "model_used": "FrustrationRule_Override"
        }
    elif system_mode in ["cold_start", "diagnostic"]:
        diag_domain = get_diagnostic_domain(len(attempts))
        recommendation = {
            "predicted_domain": diag_domain,
            "recommended_difficulty": 2,
            "recommended_task_id": 0,
            "domain_confidence": 1.0,
            "model_used": "DiagnosticProtocol"
        }
    else:
        recommendation = engine.get_next_task(attempts_raw, request.temperature)

    ui_adaptation = get_ui_adaptation_flags(student_context)

    task = await get_task_for_student(
        domain=recommendation["predicted_domain"],
        difficulty=recommendation["recommended_difficulty"],
        student_id=request.student_id,
        student_context=student_context,
        db=db
    )

    try:
        # FIX: Prevent ForeignKey violation by setting to None if task_id is 0
        rec_task_id_val = recommendation.get("recommended_task_id", 0)
        rec_task_id = str(rec_task_id_val) if rec_task_id_val and rec_task_id_val != 0 else None
        
        rec_log = Recommendation(
            id=str(uuid.uuid4()),
            student_id=request.student_id,
            session_id=request.session_id,
            recommended_domain=recommendation["predicted_domain"],
            recommended_difficulty=recommendation["recommended_difficulty"],
            recommended_task_id=rec_task_id, 
            task_source=recommendation.get("model_used"),
            domain_confidence=recommendation.get("domain_confidence"),
            student_context=student_context,
        )
        db.add(rec_log)
        db.commit()
    except Exception as e:
        logger.warning(f"Failed to log recommendation: {e}")

    return {
        "status": "success",
        "data": {
            **recommendation,
            "task": task,
            "ui_adaptation": ui_adaptation,
            "system_mode": system_mode,
            "student_context": student_context,
        }
    }

@app.post("/adaptive/session-report", response_model=SessionReportResponse)  
def session_report(request: SessionReportRequest, db: Session = Depends(get_db)):  
    attempts = (  
        db.query(TaskAttempt)  
        .filter(TaskAttempt.session_id == request.session_id)  
        .order_by(TaskAttempt.sequence_step.asc())  
        .all()  
    )  
    if not attempts:  
        raise HTTPException(status_code=404, detail="No attempts found for this session")

    student_id = attempts[0].student_id  
    total = len(attempts)  
    correct_count = 0  
    total_resp = 0.0  
    total_idle = 0.0  
    frustration_count = 0  
    attention_drops = 0  
    hints_count = 0  
    domain_stats = {}  

    for a in attempts:  
        if a.correct is True:  
            correct_count += 1  
        total_resp += float(getattr(a, "response_time_sec", 0.0))  
        total_idle += float(getattr(a, "idle_time_sec", 0.0))  
        if a.frustration_flag is True:  
            frustration_count += 1  
        if a.attention_drop_flag is True:  
            attention_drops += 1  
        hints_count += int(getattr(a, "hints_used", 0))

        d_name = getattr(a, "domain_id", "attention") or "attention"  
        if d_name not in domain_stats:  
            domain_stats[d_name] = {"attempts": 0, "correct": 0}  
        domain_stats[d_name]["attempts"] += 1  
        if a.correct is True:  
            domain_stats[d_name]["correct"] += 1

    accuracy = correct_count / total * 100  
    avg_rt = total_resp / total  
    total_window = total_resp + total_idle  
    idle_ratio = (total_idle / total_window * 100) if total_window > 0 else 0.0

    if frustration_count >= 2 or attention_drops >= 2:  
        insight = "Student showed signs of cognitive fatigue. Recommend shorter blocks next session."  
    elif accuracy >= 80 and avg_rt < 15:  
        insight = "High focus and fast responses. Ready for difficulty escalation."  
    else:  
        insight = "Stable engagement throughout the session."

    feedback = (  
        "Excellent session! Amazing focus today!"        if accuracy >= 80 else  
        "Great work! Keep practicing!"                   if accuracy >= 60 else  
        "Good effort! Every attempt helps the brain learn!" if accuracy >= 40 else  
        "Thank you for trying! Every step counts."  
    )

    fatigue_values = [float(getattr(a, "session_fatigue_index", 0.0) or 0.0) for a in attempts]  
    cognitive_profile = getattr(attempts[-1], "cognitive_profile", "unknown") if attempts else "unknown"

    return {  
        "status":     "success",  
        "session_id": request.session_id,  
        "student_id": student_id,  
        "metrics": MetricSummary(  
            total_attempts            = total,  
            overall_accuracy_pct      = round(accuracy, 1),  
            average_response_time_sec = round(avg_rt, 1),  
            total_frustration_flags   = frustration_count,  
            total_attention_drops     = attention_drops,  
            total_hints_used          = hints_count,  
            idle_time_ratio_pct       = round(idle_ratio, 1),
            avg_fatigue               = round(sum(fatigue_values) / len(fatigue_values), 3) if fatigue_values else 0.0,  
            max_fatigue               = round(max(fatigue_values), 3) if fatigue_values else 0.0,  
            cognitive_profile         = cognitive_profile,
        ),  
        "domain_breakdown": {  
            d: DomainMetric(  
                attempts     = s["attempts"],  
                accuracy_pct = round(s["correct"] / s["attempts"] * 100, 1),  
            ) for d, s in domain_stats.items()  
        },  
        "clinical_insight":  insight,  
        "clinical_feedback": feedback,  
    }

@app.get("/adaptive/student-progress/{student_id}")  
def student_progress(student_id: str, db: Session = Depends(get_db)):  
    attempts = (  
        db.query(TaskAttempt)  
        .filter(TaskAttempt.student_id == student_id)  
        .order_by(TaskAttempt.sequence_step.asc())  
        .all()  
    )  
    if not attempts:  
        raise HTTPException(status_code=404, detail="No data for this student")

    domain_stats = {}  
    for a in attempts:  
        d = getattr(a, "domain_id", "attention") or "attention"  
        if d not in domain_stats:  
            domain_stats[d] = {"attempts": 0, "correct": 0}  
        domain_stats[d]["attempts"] += 1  
        if a.correct is True:  
            domain_stats[d]["correct"] += 1

    return {  
        "student_id":   student_id,  
        "total_attempts": len(attempts),  
        "domain_mastery": {  
            d: round(s["correct"] / s["attempts"] * 100, 1)  
            for d, s in domain_stats.items()  
        },  
    }

# ── V3 ENDPOINTS ─────────────────────────────────────────────────────────────
@app.get("/adaptive/session-status/{session_id}", response_model=SessionStatusResponse)
def get_session_status(session_id: str, db: Session = Depends(get_db)):
    session = db.query(SessionModel).filter(SessionModel.session_id == session_id).first()
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    attempts = db.query(TaskAttempt).filter(TaskAttempt.session_id == session_id).all()
    attempts_raw = [{"correct": a.correct, "session_fatigue_index": float(a.session_fatigue_index or 0)} for a in attempts]
    
    student = db.query(Student).filter(Student.student_id == session.student_id).first()
    severity = student.asd_severity if student else "moderate"
    
    should_end, reason = should_end_session(attempts_raw, severity)
    
    fatigue_vals = [float(a.session_fatigue_index or 0) for a in attempts]
    avg_fatigue = sum(fatigue_vals) / len(fatigue_vals) if fatigue_vals else 0
    fatigue_label = "HIGH" if avg_fatigue > 0.65 else "MODERATE" if avg_fatigue > 0.35 else "LOW"
    
    return SessionStatusResponse(
        session_id=session_id,
        should_end=should_end,
        reason=reason,
        steps_completed=len(attempts),
        fatigue_level=fatigue_label
    )

@app.get("/adaptive/alerts/{student_id}", response_model=AlertResponse)
def get_student_alerts(student_id: str, db: Session = Depends(get_db)):
    attempts = db.query(TaskAttempt).filter(
        TaskAttempt.student_id == student_id
    ).order_by(TaskAttempt.sequence_step.desc()).limit(5).all()
    
    attempts_raw = [{
        "correct": a.correct,
        "frustration_flag": a.frustration_flag,
        "response_time_ms": float(a.response_time_ms or 0)
    } for a in reversed(attempts)]
    
    triggered, level, message = check_meltdown_alert(attempts_raw)
    
    return AlertResponse(
        student_id=student_id,
        alert_triggered=triggered,
        alert_level=level,
        message=message,
        checked_at=datetime.now().isoformat()
    )

@app.get("/adaptive/session-summary/{session_id}")
async def get_session_summary(session_id: str, db: Session = Depends(get_db)):
    # Removed 'await' because CraftX API call is synchronous
    summary = generate_session_summary(session_id, db)
    if "error" in summary:
        raise HTTPException(status_code=404, detail=summary["error"])
    return summary

@app.get("/adaptive/knowledge-state/{student_id}", response_model=KnowledgeStateResponse)  
def get_knowledge_state(student_id: str, db: Session = Depends(get_db)):  
    states = db.query(KnowledgeState).filter(KnowledgeState.student_id == student_id).all()  
    if not states:  
        return {  
            "student_id": student_id,  
            "knowledge_state": {d: 50.0 for d in [  
                "Attention", "Memory", "Language", "Executive Function",  
                "Social Cognition", "Sensory Processing", "Motor Skills", "Emotional Regulation"  
            ]}  
        }  
    return {  
        "student_id": student_id,  
        "knowledge_state": {  
            s.domain_name: round(float(s.mastery_score or 0.5) * 100, 1)  
            for s in states  
        }  
    }

@app.get("/adaptive/student-context/{student_id}")  
def get_student_context(student_id: str, db: Session = Depends(get_db)):  
    attempts = (  
        db.query(TaskAttempt)  
        .filter(TaskAttempt.student_id == student_id)  
        .order_by(TaskAttempt.sequence_step.asc())  
        .all()  
    )  
    if not attempts:  
        raise HTTPException(status_code=404, detail="No data for this student")

    attempts_raw = _marshal_attempts(attempts)

    engine = get_engine()  
    context = engine.extract_student_context(attempts_raw)  
    return {"student_id": student_id, "context": context}