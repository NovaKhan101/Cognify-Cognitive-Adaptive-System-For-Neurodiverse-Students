"""
registration.py
Cognify — Student Registration & Cold Start Initialization
Handles new student registration, cognitive profile quiz, and default knowledge state creation.
"""

import uuid
import logging
from typing import Dict, List, Tuple, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text

from models import Student, TaskAttempt, KnowledgeState
from adaptive_logic import get_default_mastery, get_system_mode, DOMAINS

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────
COGNITIVE_PROFILE_QUIZ = [
    {
        "question_id": "q1",
        "question": "Does your child prefer to do the same activities in the same order each day?",
        "yes_profile": "routine-seeking",
        "weight": 3
    },
    {
        "question_id": "q2", 
        "question": "Does your child get distracted easily and have trouble staying on one task?",
        "yes_profile": "attention-variable",
        "weight": 3
    },
    {
        "question_id": "q3",
        "question": "Does your child often forget instructions by the time they need to use them?",
        "yes_profile": "working-memory-limited",
        "weight": 3
    },
    {
        "question_id": "q4",
        "question": "Is your child sensitive to sounds, lights, textures, or other sensory experiences?",
        "yes_profile": "sensory-hypersensitive",
        "weight": 3
    },
    {
        "question_id": "q5",
        "question": "Does your child get very upset when their routine is changed unexpectedly?",
        "yes_profile": "routine-seeking",
        "weight": 2
    },
    {
        "question_id": "q6",
        "question": "Does your child need instructions repeated multiple times?",
        "yes_profile": "working-memory-limited",
        "weight": 2
    },
    {
        "question_id": "q7",
        "question": "Does your child seem overwhelmed in busy or loud environments?",
        "yes_profile": "sensory-hypersensitive",
        "weight": 2
    },
    {
        "question_id": "q8",
        "question": "Does your child have trouble focusing for more than a few minutes at a time?",
        "yes_profile": "attention-variable",
        "weight": 2
    }
]

SEVERITY_MAPPING = {
    "Level 1": "mild",
    "Level 2": "moderate", 
    "Level 3": "severe"
}

PROFILE_FRIENDLY_NAMES = {
    "attention-variable":     "Focus Learner",
    "working-memory-limited": "Step-by-Step Learner",
    "routine-seeking":        "Routine Learner",
    "sensory-hypersensitive": "Sensory Learner"
}

PROFILE_DESCRIPTIONS_FOR_PARENT = {
    "attention-variable": 
        "Your child learns best with short, engaging tasks and frequent encouragement.",
    "working-memory-limited": 
        "Your child learns best when instructions are simple and broken into small steps.",
    "routine-seeking": 
        "Your child learns best with consistent, predictable task formats.",
    "sensory-hypersensitive": 
        "Your child learns best in calm, low-stimulation learning environments."
}

# ─────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────
def get_quiz_questions() -> List[Dict]:
    """Returns the quiz questions for the frontend to display."""
    return [
        {
            "question_id": q["question_id"],
            "question": q["question"]
        }
        for q in COGNITIVE_PROFILE_QUIZ
    ]

def determine_cognitive_profile(answers: Dict[str, bool]) -> str:
    """Takes quiz answers and determines cognitive profile using weighted scoring."""
    scores = {
        "attention-variable": 0,
        "working-memory-limited": 0,
        "routine-seeking": 0,
        "sensory-hypersensitive": 0
    }

    for q in COGNITIVE_PROFILE_QUIZ:
        if answers.get(q["question_id"], False):
            scores[q["yes_profile"]] += q["weight"]

    max_score = max(scores.values())
    if max_score == 0:
        return "attention-variable"
        
    return max(scores, key=scores.get)

def generate_student_id(name: str, db: Session) -> str:
    """Generate a unique readable student ID with race-condition fallback."""
    try:
        count = db.query(Student).count()
        candidate_id = f"STU_{count + 1:04d}"
        
        # Check if ID already exists (fixes race condition)
        existing = db.query(Student).filter(
            Student.student_id == candidate_id
        ).first()
        
        if existing:
            # Fallback to UUID-based ID
            return f"STU_{str(uuid.uuid4())[:8].upper()}"
            
        return candidate_id
        
    except Exception as e:
        logger.warning(f"Could not query student count for ID generation: {e}")
        return f"STU_{str(uuid.uuid4())[:8].upper()}"

def initialize_knowledge_states(student_id: str, asd_severity: str, db: Session) -> None:
    """Creates default knowledge state rows for all 8 domains when student registers."""
    default_mastery = get_default_mastery(asd_severity)
    
    for domain in DOMAINS:
        domain_id = domain.lower().replace(" ", "_")
        
        db.execute(text("""
            INSERT INTO knowledge_states
            (student_id, session_id, domain_id, domain_name, 
             mastery_score, attempts_count, last_updated)
            VALUES
            (:student_id, NULL, :domain_id, :domain_name,
             :mastery_score, 0, NOW())
            ON CONFLICT (student_id, domain_id) DO NOTHING
        """), {
            "student_id": student_id,
            "domain_id": domain_id,
            "domain_name": domain,
            "mastery_score": default_mastery
        })
    db.commit()

def validate_registration_data(data: Dict) -> Tuple[bool, Optional[str]]:
    """Validates registration input before processing."""
    if not data.get("name") or len(str(data["name"]).strip()) < 2:
        return False, "Name must be at least 2 characters"
        
    try:
        age = int(data.get("age", 0))
        if not (3 <= age <= 25):
            return False, "Age must be between 3 and 25"
    except (ValueError, TypeError):
        return False, "Invalid age format"
        
    if data.get("diagnosis_level") not in ["Level 1", "Level 2", "Level 3"]:
        return False, "Invalid diagnosis level"
        
    if data.get("gender") not in ["Male", "Female", "Other", None]:
        return False, "Invalid gender value"

    answers = data.get("quiz_answers", {})
    if not answers:
        return False, "Quiz answers required"
        
    expected_questions = {q["question_id"] for q in COGNITIVE_PROFILE_QUIZ}
    provided_questions = set(answers.keys())
    
    if not expected_questions.issubset(provided_questions):
        missing = expected_questions - provided_questions
        return False, f"Missing quiz answers for: {missing}"

    return True, None

# ─────────────────────────────────────────────────────────
# MAIN REGISTRATION ORCHESTRATION
# ─────────────────────────────────────────────────────────
def register_student(registration_data: Dict, db: Session) -> Dict:
    """Main registration function — creates student record + initializes everything."""
    try:
        name = registration_data["name"].strip()
        age = int(registration_data["age"])
        gender = registration_data.get("gender")
        diagnosis_level = registration_data["diagnosis_level"]
        quiz_answers = registration_data["quiz_answers"]
        guardian_name = registration_data.get("guardian_name")
        guardian_contact = registration_data.get("guardian_contact")
        notes = registration_data.get("notes")
        
        # 1. Map severity and determine profile
        asd_severity = SEVERITY_MAPPING.get(diagnosis_level, "moderate")
        cognitive_profile = determine_cognitive_profile(quiz_answers)
        
        # 2. Generate Student ID
        student_id = generate_student_id(name, db)
        
        # 3. Create Student record
        new_student = Student(
            student_id=student_id,
            name=name,
            age=age,
            gender=gender,
            diagnosis_level=diagnosis_level,
            asd_severity=asd_severity,
            cognitive_profile=cognitive_profile,
            guardian_name=guardian_name,
            guardian_contact=guardian_contact,
            notes=notes
        )
        db.add(new_student)
        db.commit()
        db.refresh(new_student)
        
        # 4. Initialize Knowledge States
        initialize_knowledge_states(student_id, asd_severity, db)
        
        logger.info(f"✅ Registered new student: {student_id} ({name})")
        
        return {
            "success": True,
            "student_id": student_id,
            "name": name,
            "cognitive_profile": cognitive_profile,
            "profile_friendly_name": PROFILE_FRIENDLY_NAMES.get(cognitive_profile, "Learner"),
            "profile_description": PROFILE_DESCRIPTIONS_FOR_PARENT.get(cognitive_profile, ""),
            "asd_severity": asd_severity,
            "diagnosis_level": diagnosis_level,
            "age": age,
            "system_mode": "cold_start",
            "message": f"Welcome to Cognify! {name}'s learning profile has been set up.",
            "next_step": "Start your first session to begin the diagnostic assessment."
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Registration failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Registration failed. Please try again."
        }

def get_student_profile(student_id: str, db: Session) -> Dict:
    """Returns full student profile including current system mode and knowledge states."""
    student = db.query(Student).filter(Student.student_id == student_id).first()
    if not student:
        return {"error": "Student not found"}
        
    attempt_count = db.query(TaskAttempt).filter(
        TaskAttempt.student_id == student_id
    ).count()

    knowledge_states = db.query(KnowledgeState).filter(
        KnowledgeState.student_id == student_id
    ).all()

    system_mode = get_system_mode(attempt_count)

    return {
        "student_id": student_id,
        "name": student.name,
        "age": student.age,
        "cognitive_profile": student.cognitive_profile,
        "profile_friendly_name": PROFILE_FRIENDLY_NAMES.get(
            student.cognitive_profile, "Learner"
        ),
        "asd_severity": student.asd_severity,
        "diagnosis_level": student.diagnosis_level,
        "system_mode": system_mode,
        "total_attempts": attempt_count,
        "knowledge_states": {
            ks.domain_name: round(float(ks.mastery_score or 0.5) * 100, 1)
            for ks in knowledge_states
        },
        "registered_at": student.registration_date.isoformat() if student.registration_date else None
    }