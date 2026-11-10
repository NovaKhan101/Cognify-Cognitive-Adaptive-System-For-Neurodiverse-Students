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

# Keywords that trigger the emoji validation check
COUNTING_KEYWORDS = [
    "how many", "count the", "which one has", "which one is",
    "find the", "spot the", "point to", "select the", "identify the",
    "touch the", "circle the", "click the"
]

# Indicators that a sentence is a question/identification task, not a command
# (Currently not used for strict rejection, but kept for potential future soft-scoring)
QUESTION_INDICATORS = ["?", "which", "what", "who", "where", "when", "how", "choose", "select", "identify"]

# Regex for detecting emojis
EMOJI_PATTERN = re.compile(
    r'[\U0001F300-\U0001FAFF\U00002700-\U000027BF\U0001F600-\U0001F64F]'
)

def safe_float(val, default=0.0):
    try: return float(val)
    except (ValueError, TypeError): return default

def safe_int(val, default=0):
    try: return int(val)
    except (ValueError, TypeError): return default

def _has_counting_keyword(text: str) -> bool:
    words = set(re.findall(r'\b\w+\b', text.lower()))
    text_lower = text.lower()
    # Check multi-word phrases safely with word boundaries
    return any(re.search(r'\b' + re.escape(kw) + r'\b', text_lower) for kw in COUNTING_KEYWORDS)

def _is_visual_question(instruction: str) -> bool:
    keywords = [
        "which shape", "which color", "which one", "what emotion",
        "what color", "which sound", "which texture", "emotion is shown"
    ]
    return any(k in instruction.lower() for k in keywords)

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
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        
        if not data.get("choices"):
            logger.error("CraftX response missing choices: %s", data)
            return None
            
        return data["choices"][0].get("message", {}).get("content")
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
# FEW-SHOT EXAMPLES (Matches ai-demo.tsx style & Emoji Consistency)
# ─────────────────────────────────────────────────────────
FEW_SHOT_EXAMPLES = {
    "Attention": {
        1: {"instruction": "Find the red circle. 🔴", "options": ["🔵", "🔺", "🔴", "🟢"], "correct_answer": "🔴"},
        2: {"instruction": "Which shape is red? 🔴🔵🟩⭐", "options": ["⚪", "🟦", "🔺", "🔴"], "correct_answer": "🔴"},
        3: {"instruction": "Now find the blue objects, not red. 🔴🔵🟢🔴🔵", "options": ["🔴", "🔵", "🟢", "🔴"], "correct_answer": "🔵"},
    },
    "Memory": {
        1: {"instruction": "Which animal did you see: 🍎 🐶 ⭐?", "options": ["🍎", "🐶", "🚗"], "correct_answer": "🐶"},
        2: {"instruction": "Select all three things you saw: 🌳 🚌 🍎", "options": ["🌳", "🚌", "🍎", "🐱", "🚲"], "correct_answer": "🌳"},
        3: {"instruction": "What was the order: 🔴 → 🔵 → 🟢?", "options": ["🔵→🔴→🟢", "🟢→🔵→🔴", "🔴→🔵→🟢"], "correct_answer": "🔴→🔵→🟢"},
    },
    "Language": {
        1: {"instruction": "The sun is ___. ☀️", "options": ["hot", "cold", "wet"], "correct_answer": "hot"},
        2: {"instruction": "Put in order: eats / Sara / an apple", "options": ["Sara eats an apple", "An apple eats Sara"], "correct_answer": "Sara eats an apple"},
        3: {"instruction": "Where did Mina go? She went to the park and played with a ball.", "options": ["School", "Park", "Shop"], "correct_answer": "Park"},
    },
    "Executive Function": {
        1: {"instruction": "You want to brush your teeth. What do you do first? 🪥", "options": ["Put toothpaste on the brush", "Put on shoes", "Go outside"], "correct_answer": "Put toothpaste on the brush"},
        2: {"instruction": "Put these steps in order: 🪥 Brush teeth, 👕 Get dressed, 🎒 Pack bag, 👟 Shoes", "options": ["Brush → Dress → Pack → Shoes", "Shoes → Pack → Dress → Brush"], "correct_answer": "Brush → Dress → Pack → Shoes"},
        3: {"instruction": "You have 10 minutes before school. Which 3 things should you do?", "options": ["Brush teeth, Pack bag, Put on shoes", "Play video game, Sleep, Pack bag"], "correct_answer": "Brush teeth, Pack bag, Put on shoes"},
    },
    "Social Cognition": {
        1: {"instruction": "What emotion is shown? 😊", "options": ["😊", "😢", "😠", "😲"], "correct_answer": "😊"},
        2: {"instruction": "A friend says 'I didn't do well on my test.' 😢 What could you say?", "options": ["I'm sorry, do you want to talk?", "That's funny", "I don't care"], "correct_answer": "I'm sorry, do you want to talk?"},
        3: {"instruction": "Someone keeps looking at the clock while talking to you. ⏰ What might this mean?", "options": ["They might need to leave soon", "They don't like you", "They are hungry"], "correct_answer": "They might need to leave soon"},
    },
    "Sensory Processing": {
        1: {"instruction": "Which sound setting feels more comfortable right now?", "options": ["🔇 No sound", "🔊 Soft voice", "🎵 Music"], "correct_answer": "🔊 Soft voice"},
        2: {"instruction": "Which space feels easier to focus in?", "options": ["Bright, busy room", "Soft light, calm room"], "correct_answer": "Soft light, calm room"},
        3: {"instruction": "Which texture would feel calming to hold? 🧸🪨✨", "options": ["🧸 Soft toy", "🪨 Rough stone", "✨ Sparkly slime"], "correct_answer": "🧸 Soft toy"},
    },
    "Motor Skills": {
        1: {"instruction": "Tap the blue circle. 🔴🔵🟢", "options": ["🔴", "🔵", "🟢"], "correct_answer": "🔵"},
        2: {"instruction": "Tap the shapes in order: 🔵 → 🟢 → 🔴", "options": ["🔵🟢🔴", "🔴🟢🔵"], "correct_answer": "🔵🟢🔴"},
        3: {"instruction": "Which path moves the ball straight to the star? ⚽ ⭐", "options": ["Straight line ➡️", "Zigzag line 〰️", "Backward ⬅️"], "correct_answer": "Straight line ➡️"},
    },
    "Emotional Regulation": {
        1: {"instruction": "This activity feels hard. 😣 What could help you feel calmer?", "options": ["🫁 Take a slow breath", "😡 Give up", "🙄 Ignore it"], "correct_answer": "🫁 Take a slow breath"},
        2: {"instruction": "How are you feeling right now?", "options": ["😊 Good", "😟 Worried", "😡 Frustrated"], "correct_answer": "😊 Good"},
        3: {"instruction": "You're waiting for something you really want and feel frustrated. 😤 What might help?", "options": ["🫁 Breathing", "⏸️ Take a break", "💬 Ask for help"], "correct_answer": "🫁 Breathing"},
    },
}

def get_few_shot_example(domain: str, difficulty: int) -> Optional[Dict]:
    domain_examples = FEW_SHOT_EXAMPLES.get(domain)
    if not domain_examples:
        return None
    closest = min(domain_examples.keys(), key=lambda d: abs(d - difficulty))
    return domain_examples[closest]

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
    fatigue = safe_float(context.get("avg_fatigue", 0.0))
    frustration = safe_float(context.get("frustration_rate", 0.0))
    consec_correct = safe_int(context.get("consecutive_correct", 0))
    session_pos = safe_float(context.get("session_position", 0.5))

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
    fatigue = safe_float(context.get("avg_fatigue", 0.0))
    accuracy = safe_float(context.get("recent_accuracy", 0.5)) * 100
    frustration = safe_float(context.get("frustration_rate", 0.0))
    consec_correct = safe_int(context.get("consecutive_correct", 0))
    session_pos = safe_float(context.get("session_position", 0.5))

    fatigue_lvl = "HIGH" if fatigue > 0.7 else "MODERATE" if fatigue > 0.4 else "LOW"
    frust_lvl = "HIGH" if frustration > 0.4 else "LOW"
    near_end = "YES" if session_pos > 0.8 else "NO"

    rules = PROFILE_RULES.get(profile, ["Clear simple language", "One concept at a time"])
    rules_str = "\n".join([f"- {r}" for r in rules])

    domain_ctx = DOMAIN_CONTEXTS.get(domain, "educational tasks")
    diff_desc = DIFFICULTY_DESCRIPTIONS.get(difficulty, "Standard difficulty")

    story_ctx = ""
    if style["use_story_context"]:
        story_ctx = "[STORY CONTEXT]\nWrap the task in a simple relatable scenario involving a child named Alex, and include the scenario details directly inside the instruction text.\n"

    # Inject Few-Shot Example to match ai-demo.tsx style
    example = get_few_shot_example(domain, difficulty)
    example_section = ""
    if example:
        example_section = f"""
[STYLE REFERENCE ONLY — DO NOT COPY THIS CONTENT]
This shows the FORMAT/LENGTH/TONE to match, not the actual question to reuse:
Instruction: {example['instruction']}
Options: {example['options']}
Correct: {example['correct_answer']}
CRITICAL: Generate a COMPLETELY DIFFERENT scenario, object, or concept than the example above. Do not reuse the same colors, animals, or objects shown in the reference. The reference is ONLY for showing sentence length and emoji style — the actual content must be original.
"""

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

{story_ctx}
[VISUAL CONTENT RULE — CRITICAL]
If the task involves counting, identifying, or comparing objects, you MUST embed the actual emoji directly in the instruction text to represent those objects visually. 
NEVER reference a picture, image, or external visual. Instead of saying "Find the blue object in the picture", write "Which one is blue? 🔴 🚗 🐦 🌳".
Example: "How many red apples? 🍎🍎🍎" NOT "How many red apples?"
If the task is abstract (e.g. emotions, opposites, memory recall) and no suitable emoji exists, you may omit this — do not force emoji into non-visual tasks.

[VISUAL CONSISTENCY RULE — CRITICAL]
If this task involves colors, shapes, or emotions, you MUST:
1. Embed the actual emoji directly in the instruction (e.g., "What emotion is shown? 😊" — not "shown here" with no emoji).
2. If asking "Which one is X?", every option must BE an emoji from this exact set, not a text description:
   Shapes: ⚪ 🟦 🔺 ⭐
   Colors: 🔴 🔵 🟢 🟡 🟣
   Emotions: 😊 😢 😠 😲 😌
3. NEVER write descriptive text like "The red circle" or "Happy smiley face" as an option.
   Instead the option must literally be the emoji itself: 🔴 or 😊
4. The correct_answer must be the exact emoji string, matching one of the options exactly.

[ANSWERABILITY RULE — CRITICAL]
The instruction must be phrased as a question (e.g., "Which object is blue?", "What should you do?"). 
NEVER use imperative commands like "Find the blue object", "Put the spoon down", "Sort the blocks".
If the domain involves a physical/motor action, REPHRASE it as a recognition question instead:
- BAD: "Put the spoon down."
- GOOD: "What should you do after eating? 🥄" with options like "Put it down", "Throw it", "Keep eating", "Hide it"
Options must be genuine, distinct answer choices — not restatements or paraphrases of the instruction itself, and not variations of the same 2-3 words with different punctuation.
All options must use consistent grammar and punctuation (either all end with a period, or none do — never mixed).
Options must directly answer the question. If asking "Which one?", options must be the objects/names themselves (e.g., "The red car", "The blue ball"), NOT actions or commands.

[SELF-CONTAINED RULE — CRITICAL]
The instruction must be fully understandable on its own. Never reference a person, story, image, or prior event (e.g. "Alex", "that drawing", "the story we read") unless the full relevant detail is included in the instruction text itself. Do not assume prior context exists.

{example_section}
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

    # Inject retry note into prompt so LLM knows what went wrong
    retry_note = style.get("retry_note", "")
    if retry_note:
        prompt += f"\n[PREVIOUS ATTEMPT FEEDBACK]\n{retry_note}\n"

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
        # FIX: Correct regex to match JSON block without escaping braces
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

    # FIX: Use helper function for strict word boundary matching on counting keywords
    # Also check options for emoji presence, not just instruction
    has_emoji = bool(EMOJI_PATTERN.search(task["instruction"] + " " + " ".join(task.get("options", []))))
    if _has_counting_keyword(task["instruction"]) and not has_emoji:
        issues.append("missing_emoji_for_countable_task")

    # Check for inconsistent punctuation across options
    ends_with_period = [opt.strip().endswith('.') for opt in task["options"]]
    if len(set(ends_with_period)) > 1:
        issues.append("inconsistent_option_punctuation")

    # FIX: Removed the `looks_like_command` check entirely.
    # The few-shot examples intentionally use imperative commands ("Find the red circle."), 
    # so rejecting commands contradicts the desired design style.

    # FIX: Hard validation for visual consistency
    if _is_visual_question(task["instruction"]):
        # Every option must be pure emoji, not descriptive text
        for opt in task["options"]:
            if not EMOJI_PATTERN.search(opt) or len(opt.strip()) > 4:
                issues.append("visual_question_options_must_be_emoji")
                break
        
        # The instruction itself must contain at least one emoji if it references "shown"
        if "shown" in task["instruction"].lower() and not EMOJI_PATTERN.search(task["instruction"]):
            issues.append("visual_reference_with_no_visible_emoji")

    # Check options aren't near-duplicates of each other
    # FIX: Skip duplicate check for sequence-ordering tasks (where options are just re-ordered items)
    is_sequence_task = any("→" in opt for opt in task["options"])
    if not is_sequence_task:
        option_words = [set(opt.lower().replace(".", "").split()) for opt in task["options"]]
        for i in range(len(option_words)):
            for j in range(i + 1, len(option_words)):
                overlap = len(option_words[i] & option_words[j])
                smaller = min(len(option_words[i]), len(option_words[j]))
                if smaller > 0 and overlap / smaller > 0.6:
                    issues.append("options_too_similar")
                    break

    return (len(issues) == 0), issues

# ─────────────────────────────────────────────────────────
# MAIN AGENT LOOP
# ─────────────────────────────────────────────────────────
def agent_generate_task(context: Dict) -> Dict:
    """Main entry point — orchestrates everything."""
    style = decide_task_style(context)
    domain = context.get("domain", "Attention")
    difficulty = max(1, min(5, safe_int(context.get("difficulty", 3)) + style["difficulty_nudge"]))

    # FIX: Moved prompt builder inside the loop so retry_note is dynamically injected
    for attempt in range(MAX_RETRIES):
        prompt = build_agent_prompt(domain, difficulty, style, context)

        raw = call_craftx(prompt)

        # FIX: Changed break to continue to allow a retry if API hiccups
        if not raw:
            continue

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
# FALLBACK TASKS (Fixed to pass validation rules)
# ─────────────────────────────────────────────────────────
FALLBACK_TASKS = {
    "Attention": [
        {"title": "Find the Circle", "instruction": "Find the red circle. 🔴", "options": ["🔵", "🔴", "🟢", "🟡"], "correct_answer": "🔴", "hint": "Look for the red one", "encouragement": "Great job!"},
        {"title": "Find the Square", "instruction": "Find the blue square. 🟦", "options": ["🟦", "🔺", "⭐", "🟢"], "correct_answer": "🟦", "hint": "Look for the blue one", "encouragement": "Nice spotting!"},
        {"title": "Find the Star", "instruction": "Find the yellow star. ⭐", "options": ["🔴", "🟦", "⭐", "🟢"], "correct_answer": "⭐", "hint": "Look for the star shape", "encouragement": "Well done!"},
    ],
    "Memory": [
        {"title": "Remember the Fruit", "instruction": "Which fruit did you see: 🍎 🍌 🍇?", "options": ["🍎", "🍌", "🍒"], "correct_answer": "🍎", "hint": "It was the first one", "encouragement": "Great memory!"},
        {"title": "Remember the Animal", "instruction": "Which animal did you see: 🐶 🐱 🐦?", "options": ["🐶", "🐰", "🐦"], "correct_answer": "🐶", "hint": "Think of the first one", "encouragement": "Nice recall!"},
        {"title": "Remember the Order", "instruction": "What was the order: 🔴 → 🟢 → 🔵?", "options": ["🔴→🟢→🔵", "🔵→🟢→🔴", "🟢→🔴→🔵"], "correct_answer": "🔴→🟢→🔵", "hint": "Red came first", "encouragement": "Perfect sequence!"},
    ],
    "Language": [
        {"title": "Complete the Sentence", "instruction": "The sky is ___. ☁️", "options": ["blue", "loud", "square"], "correct_answer": "blue", "hint": "Look up on a clear day", "encouragement": "Great word choice!"},
        {"title": "Complete the Sentence", "instruction": "Ice is ___. 🧊", "options": ["cold", "soft", "loud"], "correct_answer": "cold", "hint": "Think about the freezer", "encouragement": "Nice job!"},
        {"title": "Complete the Sentence", "instruction": "Fire is ___. 🔥", "options": ["hot", "wet", "quiet"], "correct_answer": "hot", "hint": "You feel it when close to flames", "encouragement": "Well said!"},
    ],
    "Executive Function": [
        {"title": "Complete the Pattern", "instruction": "Which comes next? 🔴 🔵 🔴 ___", "options": ["🔵", "🟢", "🔴", "🟡"], "correct_answer": "🔵", "hint": "Look at the repeating pattern", "encouragement": "You got it!"},
        {"title": "What's Next?", "instruction": "You wake up. What do you do first? 🛏️", "options": ["Brush teeth", "Go to sleep", "Eat dinner"], "correct_answer": "Brush teeth", "hint": "Think about your morning routine", "encouragement": "Great planning!"},
        {"title": "Sort the Steps", "instruction": "What comes after 👕 getting dressed?", "options": ["👟 Put on shoes", "🛏️ Go to sleep", "🪥 Brush teeth"], "correct_answer": "👟 Put on shoes", "hint": "Think about what's next before leaving", "encouragement": "Nice sequencing!"},
    ],
    "Social Cognition": [
        {"title": "How Do They Feel?", "instruction": "What emotion is shown? 😢", "options": ["😊", "😢", "😠"], "correct_answer": "😢", "hint": "Tears usually mean this emotion", "encouragement": "Good empathy!"},
        {"title": "How Do They Feel?", "instruction": "What emotion is shown? 😠", "options": ["😊", "😲", "😠"], "correct_answer": "😠", "hint": "This face looks upset", "encouragement": "Nice reading!"},
        {"title": "What Could You Say?", "instruction": "A friend looks sad. 😢 What could you say?", "options": ["Are you okay?", "That's funny", "I don't care"], "correct_answer": "Are you okay?", "hint": "Think of a kind response", "encouragement": "Very thoughtful!"},
    ],
    "Sensory Processing": [
        {"title": "Soft or Hard", "instruction": "Which feels soft to hold? 🧸🪨", "options": ["🧸", "🪨"], "correct_answer": "🧸", "hint": "One is squishy", "encouragement": "Nice sensing!"},
        {"title": "Sound Choice", "instruction": "Which sound setting feels calmer?", "options": ["🔇 No sound", "🎵 Loud music"], "correct_answer": "🔇 No sound", "hint": "Think about quiet spaces", "encouragement": "Good choice!"},
        {"title": "Light Choice", "instruction": "Which space feels easier to relax in?", "options": ["Soft light 💡", "Bright flashing lights ⚡"], "correct_answer": "Soft light 💡", "hint": "Think about calm rooms", "encouragement": "Well chosen!"},
    ],
    "Motor Skills": [
        {"title": "Hand Action", "instruction": "What do you do with a pencil? ✏️", "options": ["Eat", "Write", "Sleep"], "correct_answer": "Write", "hint": "You use it on paper", "encouragement": "Good job!"},
        {"title": "Tap the Shape", "instruction": "Tap the blue circle. 🔴🔵🟢", "options": ["🔴", "🔵", "🟢"], "correct_answer": "🔵", "hint": "Look for the blue one", "encouragement": "Nice tap!"},
        {"title": "Follow the Order", "instruction": "Tap the shapes in order: 🔴 → 🟢", "options": ["🔴🟢", "🟢🔴"], "correct_answer": "🔴🟢", "hint": "Red comes first", "encouragement": "Great coordination!"},
    ],
    "Emotional Regulation": [
        {"title": "Calm Down Strategy", "instruction": "When angry, what is a good choice? 😠", "options": ["Yell", "Take deep breaths", "Hit"], "correct_answer": "Take deep breaths", "hint": "Inhale... exhale...", "encouragement": "Wonderful choice!"},
        {"title": "How Are You Feeling?", "instruction": "How are you feeling right now?", "options": ["😊 Good", "😟 Worried", "😡 Frustrated"], "correct_answer": "😊 Good", "hint": "Think about right now", "encouragement": "Thanks for sharing!"},
        {"title": "Take a Break", "instruction": "This feels hard. 😣 What could help?", "options": ["🫁 Take a breath", "😡 Give up", "🙄 Ignore it"], "correct_answer": "🫁 Take a breath", "hint": "Try something calming first", "encouragement": "Good self-care!"},
    ],
}

def fallback_task(context: Dict) -> Dict:
    """Returns a safe hardcoded task when API fails."""
    domain = context.get("domain", "Attention")
    task_list = FALLBACK_TASKS.get(domain, FALLBACK_TASKS["Attention"])
    
    # FIX: Pick randomly instead of always index [0]
    task = random.choice(task_list).copy()
    
    task["source"] = "fallback"
    task["domain"] = domain
    task["difficulty"] = safe_int(context.get("difficulty", 3))
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

        query = db.query(TaskBank).filter(
            TaskBank.domain_name == domain,
            TaskBank.difficulty == difficulty,
            TaskBank.cognitive_profile == cognitive_profile
        )
        
        # FIX: Only apply filter if seen_ids is not empty to avoid invalid SQL
        if seen_ids:
            query = query.filter(~TaskBank.id.in_(seen_ids))
            
        task = query.order_by(TaskBank.usage_count.asc(), func.random()).first()

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
# FIX: Removed `async` since there are no `await` calls inside.
def get_task_for_student(domain: str, difficulty: int, student_id: str, student_context: Dict, db: Session) -> Dict:
    """Main task delivery function called from main.py"""
    student_context["domain"] = domain
    student_context["difficulty"] = difficulty

    high_fatigue = safe_float(student_context.get("avg_fatigue", 0.0)) > 0.7
    high_frustration = safe_float(student_context.get("frustration_rate", 0.0)) > 0.4

    if high_fatigue or high_frustration:
        logger.info("🤖 High fatigue/frustration detected. Using Live Agent.")
        task = agent_generate_task(student_context)
        task["source"] = "gemini_live"
        return task

    profile = student_context.get("cognitive_profile", "unknown")
    task = get_task_from_bank(domain, difficulty, profile, student_id, db)
    
    if task:
        logger.info("📚 Task retrieved from Task Bank.")
        return task

    logger.info("🤖 Task Bank exhausted. Generating new task.")
    task = agent_generate_task(student_context)
    
    if not high_fatigue and not high_frustration:
        save_task_to_bank(task, domain, difficulty, profile, db)
        
    task["source"] = "gemini_exhausted"
    return task