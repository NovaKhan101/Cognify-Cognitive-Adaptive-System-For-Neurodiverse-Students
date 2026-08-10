"""
task_agent.py
Cognify — Agentic Task Generation & Validation Loop
Connects ML behavioral outputs to dynamic LLM task generation with validation.
Uses CraftX API (OpenAI-compatible) instead of Google Gemini SDK.
"""

import os  
import json  
import logging  
import re  
import random  
import uuid  
import requests
from typing import Dict, Optional, Tuple, List

# Import pure logic & DB models  
from adaptive_logic import SENSORY_LOAD, DOMAINS  
from models import TaskBank, StudentTaskHistory  
from database import SessionLocal  
from sqlalchemy.orm import Session  
from sqlalchemy import text, func

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────  
# CONSTANTS & CRAFTX API SETUP  
# ─────────────────────────────────────────────────────────  
CRAFTX_API_KEY = os.getenv("CRAFTX_API_KEY")
CRAFTX_ENDPOINT = "https://api.craftx.corecraftsolutions.com/api/v1/chat/completions"
GEMINI_MODEL = "Gemini 2.5 Flash"  # Exact name from CraftX dropdown
MAX_RETRIES = 2

def call_craftx(prompt: str) -> Optional[str]:
    """Helper function to call CraftX API."""
    if not CRAFTX_API_KEY:
        logger.warning("CRAFTX_API_KEY not found")
        return None
    try:
        response = requests.post(
            CRAFTX_ENDPOINT,
            headers={
                "Authorization": f"Bearer {CRAFTX_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": GEMINI_MODEL,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 1000,
                "temperature": 0.7,
            },
            timeout=30
        )
        data = response.json()
        
        # Check for warnings (low balance)
        if data.get("warning"):
            logger.warning(f"CraftX warning: {data['warning']}")
            
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"CraftX API call failed: {e}")
        return None

DIFFICULTY_DESCRIPTIONS = {  
    1: "Very simple, single concept, familiar objects, max 5 words instruction",  
    2: "Simple, one step, common scenarios, max 8 words instruction",  
    3: "Moderate, two concepts, requires basic reasoning, max 12 words",  
    4: "Challenging, multi-step thinking, abstract concepts, max 15 words",  
    5: "Advanced, complex reasoning, requires prior knowledge, max 20 words"  
}

PROFILE_RULES = {  
    "working-memory-limited": [  
        "Maximum 6 words in instruction",  
        "Never use multi-part questions",  
        "Use concrete familiar objects only",  
        "One concept per task only",  
        "Answer options must be clearly different from each other"  
    ],  
    "attention-variable": [  
        "Start instruction with an action verb",  
        "Keep task completable in under 30 seconds",  
        "Make correct answer visually distinct when described",  
        "Use engaging relatable scenarios"  
    ],  
    "routine-seeking": [  
        "Use familiar 'Choose the correct...' format",  
        "No surprise or trick elements",  
        "Predictable question structure",  
        "Use concrete real-world examples"  
    ],  
    "sensory-hypersensitive": [  
        "No loud, bright, or overwhelming descriptions",  
        "Calm neutral language throughout",  
        "Simple clean answer options",  
        "Avoid sensory-loaded words like 'loud', 'bright', 'crowded'"  
    ]  
}

DOMAIN_CONTEXTS = {  
    "Attention": "focused attention, sustained concentration, selective attention tasks",  
    "Memory": "working memory, recall, recognition, short-term memory tasks",  
    "Language": "vocabulary, comprehension, verbal reasoning, communication",  
    "Executive Function": "planning, sequencing, problem solving, decision making",  
    "Social Cognition": "emotions, social situations, perspective taking, facial expressions",  
    "Sensory Processing": "sensory awareness, calm sensory exploration, gentle sensory tasks",  
    "Motor Skills": "fine motor control, hand-eye coordination, movement sequencing",  
    "Emotional Regulation": "identifying emotions, calming strategies, emotional awareness"  
}

# ─────────────────────────────────────────────────────────  
# AGENT DECISION ENGINE  
# ─────────────────────────────────────────────────────────  
def decide_task_style(context: Dict) -> Dict:  
    """Applies heuristic rules based on student behavioral context before generating."""  
    style = {  
        "max_words": 12,  
        "num_steps": 1,  
        "options_count": 4,  
        "tone": "neutral",  
        "difficulty_nudge": 0,  
        "use_story_context": False,  
        "format_type": "standard"  
    }

    profile = context.get("cognitive_profile", "unknown")  
    fatigue = float(context.get("avg_fatigue", 0.0))  
    frustration = float(context.get("frustration_rate", 0.0))  
    consec_correct = int(context.get("consecutive_correct", 0))  
    session_pos = float(context.get("session_position", 0.5))

    # Cognitive profile adjustments  
    if profile == "working-memory-limited":  
        style["max_words"] = 6  
        style["options_count"] = 3  
    if profile == "routine-seeking":  
        style["format_type"] = "standard"  
        style["use_story_context"] = False  
    if profile == "attention-variable":  
        style["use_story_context"] = True  
        style["tone"] = "warm"  
    if profile == "sensory-hypersensitive":  
        style["tone"] = "calm"  
        style["use_story_context"] = False

    # Behavioral state adjustments  
    if fatigue > 0.7:  
        style["difficulty_nudge"] = -1  
        style["tone"] = "warm"  
        style["max_words"] = min(style["max_words"], 8)  
    if frustration > 0.4:  
        style["difficulty_nudge"] = -1  
        style["tone"] = "warm"  
    if consec_correct >= 3:  
        style["difficulty_nudge"] = 1  
        style["tone"] = "celebratory"  
    if session_pos > 0.8:  
        style["max_words"] = min(style["max_words"], 8)  
        style["tone"] = "celebratory"  
        style["difficulty_nudge"] = min(style["difficulty_nudge"], -1)

    return style

# ─────────────────────────────────────────────────────────  
# PROMPT BUILDER  
# ─────────────────────────────────────────────────────────  
def build_agent_prompt(domain: str, difficulty: int, style: Dict, context: Dict) -> str:  
    profile = context.get("cognitive_profile", "unknown")  
    severity = context.get("asd_severity", "moderate")  
    fatigue = float(context.get("avg_fatigue", 0.0))  
    accuracy = float(context.get("recent_accuracy", 0.5)) * 100  
    frustration = float(context.get("frustration_rate", 0.0))  
    consec_correct = int(context.get("consecutive_correct", 0))  
    session_pos = float(context.get("session_position", 0.5))

    fatigue_lvl = "HIGH" if fatigue > 0.7 else "MODERATE" if fatigue > 0.4 else "LOW"  
    frust_lvl = "HIGH" if frustration > 0.4 else "LOW"  
    near_end = "YES" if session_pos > 0.8 else "NO"

    rules = PROFILE_RULES.get(profile, ["Clear simple language", "One concept at a time"])  
    rules_str = "\n".join([f"- {r}" for r in rules])

    domain_ctx = DOMAIN_CONTEXTS.get(domain, "educational tasks")  
    diff_desc = DIFFICULTY_DESCRIPTIONS.get(difficulty, "Standard difficulty")

    story_ctx = ""  
    if style["use_story_context"]:  
        story_ctx = "[STORY CONTEXT]\nWrap the task in a simple relatable scenario involving a child named Alex.\n"

    prompt = f"""  
[ROLE]  
You are a specialist educational content creator for children with ASD.  
Never use complex language. Never be condescending. Always be encouraging.

[STUDENT PROFILE]  
Cognitive Profile: {profile}  
ASD Severity: {severity}  
Domain: {domain} — {domain_ctx}  
Target Difficulty: {difficulty}/5 — {diff_desc}

[STUDENT STATE RIGHT NOW]  
Session Fatigue: {fatigue_lvl}  
Recent Accuracy: {accuracy:.1f}%  
Frustration Level: {frust_lvl}  
Consecutive Correct: {consec_correct}  
Near End of Session: {near_end}

[TASK STYLE REQUIREMENTS — FOLLOW STRICTLY]  
- Maximum instruction length: {style['max_words']} words  
- Tone: {style['tone']}  
- Answer options: {style['options_count']}  
{rules_str}

[DOMAIN-SPECIFIC INSTRUCTIONS]  
{rules_str}

{story_ctx}  
[OUTPUT FORMAT]  
Return ONLY valid JSON. No markdown. No explanation. No backticks.  
{{  
  "title": "short task title",  
  "instruction": "the task instruction",  
  "options": ["option1", "option2", "option3", "option4"],  
  "correct_answer": "exact match to one option",  
  "hint": "gentle hint if student is stuck",  
  "encouragement": "positive message shown after correct answer"  
}}  
"""  
    return prompt

# ─────────────────────────────────────────────────────────  
# VALIDATION & PARSING  
# ─────────────────────────────────────────────────────────  
def parse_gemini_response(raw_text: str) -> Optional[Dict]:  
    """Safely parse the JSON response."""  
    try:  
        cleaned = re.sub(r'```json|```', '', raw_text).strip()  
        return json.loads(cleaned)  
    except json.JSONDecodeError:  
        match = re.search(r'\{.*\}', raw_text, re.DOTALL)  
        if match:  
            try:  
                return json.loads(match.group(0))  
            except json.JSONDecodeError as e:  
                logger.error(f"JSON Regex parse failed: {e}")  
        logger.error(f"Failed to parse response: {raw_text}")  
        return None

def validate_task(task: Dict, style: Dict, difficulty: int) -> Tuple[bool, list]:  
    """Validates the output against style constraints and logic."""  
    issues = []  
    required_keys = ["title", "instruction", "options", "correct_answer", "hint", "encouragement"]  
     
    for key in required_keys:  
        if key not in task or not task[key]:  
            issues.append(f"missing_or_empty_field_{key}")

    if issues:  
        return False, issues

    if task["correct_answer"] not in task["options"]:  
        issues.append("correct_answer_not_in_options")

    if len(task["options"]) != style["options_count"]:  
        issues.append(f"wrong_options_count_expected_{style['options_count']}")

    word_count = len(task["instruction"].split())  
    if word_count > style["max_words"] + 3:  
        issues.append("instruction_too_long")

    if any(not opt for opt in task["options"]):  
        issues.append("empty_option_string")

    if len(set(task["options"])) != len(task["options"]):  
        issues.append("duplicate_options")

    if not task["instruction"]:  
        issues.append("instruction_is_empty")

    return (len(issues) == 0), issues

# ─────────────────────────────────────────────────────────  
# MAIN AGENT LOOP  
# ─────────────────────────────────────────────────────────  
def agent_generate_task(context: Dict) -> Dict:  
    """Main entry point — orchestrates everything."""  
    style = decide_task_style(context)  
    domain = context.get("domain", "Attention")  
    difficulty = max(1, min(5, context.get("difficulty", 3) + style["difficulty_nudge"]))  

    prompt = build_agent_prompt(domain, difficulty, style, context)  

    for attempt in range(MAX_RETRIES):  
        raw = call_craftx(prompt)  

        if not raw:  
            break  

        task = parse_gemini_response(raw)  

        if not task:  
            continue  

        valid, issues = validate_task(task, style, difficulty)  

        if valid:  
            task["source"] = "gemini_agent"  
            task["domain"] = domain  
            task["difficulty"] = difficulty  
            task["cognitive_profile"] = context.get("cognitive_profile", "unknown")  
            task["style_applied"] = style  
            task["task_subtype"] = task.get("title", "adaptive").replace(" ", "_").lower()  
            return task  

        # Add issues to next retry  
        style["retry_note"] = f"Previous attempt failed validation: {issues}. Fix these."  

    return fallback_task(context)

# ─────────────────────────────────────────────────────────  
# FALLBACK TASKS  
# ─────────────────────────────────────────────────────────  
FALLBACK_TASKS = {  
    "Attention": [{  
        "title": "Find the Different One",  
        "instruction": "Which one is different?",  
        "options": ["Red Circle", "Red Circle", "Blue Circle", "Red Circle"],  
        "correct_answer": "Blue Circle",  
        "hint": "Look carefully at the colors",  
        "encouragement": "Great spotting!"  
    }],  
    "Memory": [{  
        "title": "Remember the Item",  
        "instruction": "What was in the box?",  
        "options": ["A Ball", "A Car", "A Tree", "A Book"],  
        "correct_answer": "A Ball",  
        "hint": "It is something you bounce",  
        "encouragement": "Great memory!"  
    }],  
    "Language": [{  
        "title": "Opposite Meaning",  
        "instruction": "What is the opposite of hot?",  
        "options": ["Warm", "Cold", "Boiling", "Spicy"],  
        "correct_answer": "Cold",  
        "hint": "It is how winter feels",  
        "encouragement": "Excellent work!"  
    }],  
    "Executive Function": [{  
        "title": "Complete the Pattern",  
        "instruction": "Which comes next? Red, Blue, Red, ___",  
        "options": ["Green", "Yellow", "Blue", "Red"],  
        "correct_answer": "Blue",  
        "hint": "Look at the repeating colors",  
        "encouragement": "You got it!"  
    }],  
    "Social Cognition": [{  
        "title": "How Do They Feel?",  
        "instruction": "Someone is crying. How do they feel?",  
        "options": ["Happy", "Sad", "Angry", "Silly"],  
        "correct_answer": "Sad",  
        "hint": "Tears usually mean this emotion",  
        "encouragement": "Good empathy!"  
    }],  
    "Sensory Processing": [{  
        "title": "Soft or Hard",  
        "instruction": "Touch a pillow. How does it feel?",  
        "options": ["Hard", "Soft", "Wet", "Hot"],  
        "correct_answer": "Soft",  
        "hint": "Pillows are squishy",  
        "encouragement": "Nice sensing!"  
    }],  
    "Motor Skills": [{  
        "title": "Hand Action",  
        "instruction": "What do you do with a pencil?",  
        "options": ["Eat", "Write", "Sleep", "Run"],  
        "correct_answer": "Write",  
        "hint": "You use it on paper",  
        "encouragement": "Good job!"  
    }],  
    "Emotional Regulation": [{  
        "title": "Calm Down Strategy",  
        "instruction": "When angry, what is a good choice?",  
        "options": ["Yell", "Take deep breaths", "Hit", "Run away"],  
        "correct_answer": "Take deep breaths",  
        "hint": "Inhale... exhale...",  
        "encouragement": "Wonderful choice!"  
    }]  
}

def fallback_task(context: Dict) -> Dict:  
    """Returns a safe hardcoded task when API fails."""  
    domain = context.get("domain", "Attention")  
    task_list = FALLBACK_TASKS.get(domain, FALLBACK_TASKS["Attention"])  
    task = task_list[0].copy()  
     
    task["source"] = "fallback"  
    task["domain"] = domain  
    task["difficulty"] = context.get("difficulty", 3)  
    task["cognitive_profile"] = context.get("cognitive_profile", "unknown")  
    task["task_subtype"] = task.get("title", "fallback").replace(" ", "_").lower()  
    return task

# ─────────────────────────────────────────────────────────  
# DATABASE INTEGRATION  
# ─────────────────────────────────────────────────────────  
def get_task_from_bank(domain: str, difficulty: int, cognitive_profile: str, student_id: str, db: Session) -> Optional[Dict]:  
    """Fetch pre-generated task from task_bank table."""  
    try:  
        seen_ids = db.execute(text(  
            "SELECT task_db_id::text FROM student_task_history WHERE student_id = :sid"  
        ), {"sid": student_id}).scalars().all()

        task = db.query(TaskBank).filter(  
            TaskBank.domain_name == domain,  
            TaskBank.difficulty == difficulty,  
            TaskBank.cognitive_profile == cognitive_profile,  
            ~TaskBank.id.in_(seen_ids)  
        ).order_by(TaskBank.usage_count.asc(), func.random()).first()

        if not task:  
            return None

        task.usage_count = (task.usage_count or 0) + 1  
         
        db.execute(text(  
            "INSERT INTO student_task_history (student_id, task_db_id) VALUES (:sid, :tid)"  
        ), {"sid": student_id, "tid": task.id})  
         
        db.commit()

        return {  
            "title": task.title,  
            "instruction": task.instruction,  
            "options": task.options if isinstance(task.options, list) else json.loads(task.options),  
            "correct_answer": task.correct_answer,  
            "hint": task.hint,  
            "encouragement": task.encouragement,  
            "task_subtype": task.cognitive_profile,  
            "source": "task_bank",  
            "domain": domain,  
            "difficulty": difficulty,  
            "cognitive_profile": cognitive_profile  
        }

    except Exception as e:  
        db.rollback()  
        logger.error(f"Error fetching from task bank: {e}")  
        return None

def save_task_to_bank(task: Dict, domain: str, difficulty: int, profile: str, db: Session) -> None:  
    """Cache generated task for future reuse by other students."""  
    try:  
        db.execute(text("""  
            INSERT INTO task_bank  
            (id, domain_name, difficulty, cognitive_profile,  
             title, instruction, options, correct_answer,  
             hint, encouragement, sensory_load, is_adaptive, usage_count)  
            VALUES  
            (gen_random_uuid(), :domain, :difficulty, :profile,  
             :title, :instruction, :options, :correct_answer,  
             :hint, :encouragement, :sensory_load, true, 0)  
        """), {  
            "domain": domain,  
            "difficulty": difficulty,  
            "profile": profile,  
            "title": task.get("title", ""),  
            "instruction": task.get("instruction", ""),  
            "options": json.dumps(task.get("options", [])),  
            "correct_answer": task.get("correct_answer", ""),  
            "hint": task.get("hint", ""),  
            "encouragement": task.get("encouragement", ""),  
            "sensory_load": SENSORY_LOAD.get(domain, 2),  
        })  
        db.commit()  
        logger.info(f"💾 Task cached to task_bank: {domain} D{difficulty} {profile}")  
    except Exception as e:  
        db.rollback()  
        logger.warning(f"Could not cache task to bank: {e}")

# ─────────────────────────────────────────────────────────  
# MAIN ORCHESTRATOR  
# ─────────────────────────────────────────────────────────  
async def get_task_for_student(domain: str, difficulty: int, student_id: str, student_context: Dict, db: Session) -> Dict:  
    """Main task delivery function called from main.py"""  
    student_context["domain"] = domain  
    student_context["difficulty"] = difficulty

    high_fatigue = float(student_context.get("avg_fatigue", 0.0)) > 0.7  
    high_frustration = float(student_context.get("frustration_rate", 0.0)) > 0.4

    if high_fatigue or high_frustration:  
        logger.info("🤖 High fatigue/frustration detected. Using Live Agent.")  
        # Removed await because agent_generate_task is now sync
        task = agent_generate_task(student_context)  
        task["source"] = "gemini_live"  
        return task

    profile = student_context.get("cognitive_profile", "unknown")  
    task = get_task_from_bank(domain, difficulty, profile, student_id, db)  
     
    if task:  
        logger.info("📚 Task retrieved from Task Bank.")  
        return task

    logger.info("🤖 Task Bank exhausted. Generating new task.")  
    # Removed await because agent_generate_task is now sync
    task = agent_generate_task(student_context)  
     
    if not high_fatigue and not high_frustration:  
        save_task_to_bank(task, domain, difficulty, profile, db)  
         
    task["source"] = "gemini_exhausted"  
    return task