from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

DARK_BG = RGBColor(0x1A, 0x1A, 0x2E)
ACCENT_TEAL = RGBColor(0x2D, 0xD4, 0xBF)
ACCENT_CORAL = RGBColor(0xF4, 0x7C, 0x7C)
ACCENT_AMBER = RGBColor(0xF5, 0xA6, 0x23)
LIGHT_TEXT = RGBColor(0xF8, 0xFA, 0xFC)
MUTED_TEXT = RGBColor(0x94, 0xA3, 0xB8)
CARD_BG = RGBColor(0x24, 0x24, 0x3A)
HIGHLIGHT = RGBColor(0x38, 0xB2, 0xAC)

def add_background(slide, color=DARK_BG):
    background = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    background.fill.solid()
    background.fill.fore_color.rgb = color
    background.line.fill.background()
    spTree = slide.shapes._spTree
    sp = background._element
    spTree.remove(sp)
    spTree.insert(2, sp)

def add_title_box(slide, text, left, top, width, height, font_size=44, bold=True, color=LIGHT_TEXT, align=PP_ALIGN.LEFT):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = "Calibri"
    p.alignment = align
    return txBox

def add_accent_bar(slide, left, top, width, height, color=ACCENT_TEAL):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    return shape

def add_bullet_list(slide, bullets, left, top, width, height, font_size=18, color=LIGHT_TEXT):
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, bullet in enumerate(bullets):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = "▸  " + bullet
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.font.name = "Calibri"
        p.space_after = Pt(14)
        p.level = 0
    return txBox

# ============================================================
# SLIDES 1-9 (already generated above - paste those here too)
# ============================================================

# For brevity, I'll provide slides 10-12 below. 
# In your actual file, include ALL slides 1-12 from the code above plus these:

# ============================================================
# SLIDE 10: Personalization Example
# ============================================================
slide10 = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide10)
add_accent_bar(slide10, Inches(0.8), Inches(1.35), Inches(1.5), Inches(0.06), ACCENT_TEAL)
add_title_box(slide10, "Personalization Example", Inches(0.8), Inches(0.7), Inches(11), Inches(0.6), font_size=40, bold=True, color=LIGHT_TEXT)
add_title_box(slide10, "Same domain (Attention) + same difficulty (Level 2) → different output for different profiles", Inches(0.8), Inches(1.4), Inches(11), Inches(0.4), font_size=18, bold=False, color=MUTED_TEXT)

# Profile 1: Attention-Variable
card1 = slide10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.6), Inches(2.0), Inches(6.0), Inches(2.4))
card1.fill.solid(); card1.fill.fore_color.rgb = CARD_BG; card1.line.color.rgb = ACCENT_TEAL; card1.line.width = Pt(2)
add_title_box(slide10, "Attention-Variable Profile", Inches(0.8), Inches(2.15), Inches(5.6), Inches(0.35), font_size=16, bold=True, color=ACCENT_TEAL)
ex1 = slide10.shapes.add_textbox(Inches(0.8), Inches(2.55), Inches(5.6), Inches(1.7))
tf = ex1.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]; p.text = "Task: 'Spot the Different Shape'"; p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = LIGHT_TEXT; p.font.name = "Calibri"
p = tf.add_paragraph(); p.text = "• Title: 'Quick Shape Hunt! 🎯'"; p.font.size = Pt(13); p.font.color.rgb = LIGHT_TEXT; p.font.name = "Calibri"; p.space_before = Pt(6)
p = tf.add_paragraph(); p.text = "• Instructions: 'Find the one shape that doesn't match!'"; p.font.size = Pt(13); p.font.color.rgb = LIGHT_TEXT; p.font.name = "Calibri"
p = tf.add_paragraph(); p.text = "• 4 options (simple shapes, bright colors)"; p.font.size = Pt(13); p.font.color.rgb = LIGHT_TEXT; p.font.name = "Calibri"
p = tf.add_paragraph(); p.text = "• Encouragement after every interaction"; p.font.size = Pt(13); p.font.color.rgb = LIGHT_TEXT; p.font.name = "Calibri"
p = tf.add_paragraph(); p.text = "• Short, snappy tone with emojis"; p.font.size = Pt(13); p.font.color.rgb = LIGHT_TEXT; p.font.name = "Calibri"

# Profile 2: Sensory-Hypersensitive
card2 = slide10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(2.0), Inches(6.0), Inches(2.4))
card2.fill.solid(); card2.fill.fore_color.rgb = CARD_BG; card2.line.color.rgb = HIGHLIGHT; card2.line.width = Pt(2)
add_title_box(slide10, "Sensory-Hypersensitive Profile", Inches(7.0), Inches(2.15), Inches(5.6), Inches(0.35), font_size=16, bold=True, color=HIGHLIGHT)
ex2 = slide10.shapes.add_textbox(Inches(7.0), Inches(2.55), Inches(5.6), Inches(1.7))
tf = ex2.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]; p.text = "Task: 'Spot the Different Shape'"; p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = LIGHT_TEXT; p.font.name = "Calibri"
p = tf.add_paragraph(); p.text = "• Title: 'Find the Different Shape'"; p.font.size = Pt(13); p.font.color.rgb = LIGHT_TEXT; p.font.name = "Calibri"; p.space_before = Pt(6)
p = tf.add_paragraph(); p.text = "• Instructions: 'Look carefully. One shape is different from the others.'"; p.font.size = Pt(13); p.font.color.rgb = LIGHT_TEXT; p.font.name = "Calibri"
p = tf.add_paragraph(); p.text = "• 3 options (muted, calm colors — no bright flashes)"; p.font.size = Pt(13); p.font.color.rgb = LIGHT_TEXT; p.font.name = "Calibri"
p = tf.add_paragraph(); p.text = "• Minimal animations, clean layout"; p.font.size = Pt(13); p.font.color.rgb = LIGHT_TEXT; p.font.name = "Calibri"
p = tf.add_paragraph(); p.text = "• Calm, reassuring tone; no sudden sounds"; p.font.size = Pt(13); p.font.color.rgb = LIGHT_TEXT; p.font.name = "Calibri"

# Profile 3: Working-Memory-Limited
card3 = slide10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.6), Inches(4.6), Inches(6.0), Inches(2.4))
card3.fill.solid(); card3.fill.fore_color.rgb = CARD_BG; card3.line.color.rgb = ACCENT_CORAL; card3.line.width = Pt(2)
add_title_box(slide10, "Working-Memory-Limited Profile", Inches(0.8), Inches(4.75), Inches(5.6), Inches(0.35), font_size=16, bold=True, color=ACCENT_CORAL)
ex3 = slide10.shapes.add_textbox(Inches(0.8), Inches(5.15), Inches(5.6), Inches(1.7))
tf = ex3.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]; p.text = "Task: 'Spot the Different Shape'"; p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = LIGHT_TEXT; p.font.name = "Calibri"
p = tf.add_paragraph(); p.text = "• Title: 'Find the Odd One'"; p.font.size = Pt(13); p.font.color.rgb = LIGHT_TEXT; p.font.name = "Calibri"; p.space_before = Pt(6)
p = tf.add_paragraph(); p.text = "• Instructions: 'Tap the different shape.' (4 words)"; p.font.size = Pt(13); p.font.color.rgb = LIGHT_TEXT; p.font.name = "Calibri"
p = tf.add_paragraph(); p.text = "• 3 options only (reduced cognitive load)"; p.font.size = Pt(13); p.font.color.rgb = LIGHT_TEXT; p.font.name = "Calibri"
p = tf.add_paragraph(); p.text = "• One-step task, no multi-part reasoning"; p.font.size = Pt(13); p.font.color.rgb = LIGHT_TEXT; p.font.name = "Calibri"
p = tf.add_paragraph(); p.text = "• Visual reminder of the goal stays on screen"; p.font.size = Pt(13); p.font.color.rgb = LIGHT_TEXT; p.font.name = "Calibri"

# Profile 4: Routine-Seeking
card4 = slide10.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(4.6), Inches(6.0), Inches(2.4))
card4.fill.solid(); card4.fill.fore_color.rgb = CARD_BG; card4.line.color.rgb = ACCENT_AMBER; card4.line.width = Pt(2)
add_title_box(slide10, "Routine-Seeking Profile", Inches(7.0), Inches(4.75), Inches(5.6), Inches(0.35), font_size=16, bold=True, color=ACCENT_AMBER)
ex4 = slide10.shapes.add_textbox(Inches(7.0), Inches(5.15), Inches(5.6), Inches(1.7))
tf = ex4.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]; p.text = "Task: 'Spot the Different Shape'"; p.font.size = Pt(14); p.font.bold = True; p.font.color.rgb = LIGHT_TEXT; p.font.name = "Calibri"
p = tf.add_paragraph(); p.text = "• Title: 'Shape Match — Level 2'"; p.font.size = Pt(13); p.font.color.rgb = LIGHT_TEXT; p.font.name = "Calibri"; p.space_before = Pt(6)
p = tf.add_paragraph(); p.text = "• Instructions: 'Find the shape that does not belong.'"; p.font.size = Pt(13); p.font.color.rgb = LIGHT_TEXT; p.font.name = "Calibri"
p = tf.add_paragraph(); p.text = "• Same layout pattern as all previous tasks"; p.font.size = Pt(13); p.font.color.rgb = LIGHT_TEXT; p.font.name = "Calibri"
p = tf.add_paragraph(); p.text = "• Predictable sequence: instruction → options → feedback"; p.font.size = Pt(13); p.font.color.rgb = LIGHT_TEXT; p.font.name = "Calibri"
p = tf.add_paragraph(); p.text = "• No surprise elements; consistent UI placement"; p.font.size = Pt(13); p.font.color.rgb = LIGHT_TEXT; p.font.name = "Calibri"

# ============================================================
# SLIDE 11: Limitations & Future Work
# ============================================================
slide11 = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide11)
add_accent_bar(slide11, Inches(0.8), Inches(1.35), Inches(1.5), Inches(0.06), ACCENT_CORAL)
add_title_box(slide11, "Limitations & Future Work", Inches(0.8), Inches(0.7), Inches(11), Inches(0.6), font_size=40, bold=True, color=LIGHT_TEXT)

# Limitations column
lim_card = slide11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.6), Inches(1.7), Inches(5.9), Inches(5.0))
lim_card.fill.solid(); lim_card.fill.fore_color.rgb = CARD_BG; lim_card.line.color.rgb = ACCENT_CORAL; lim_card.line.width = Pt(2)
add_title_box(slide11, "Current Limitations", Inches(0.8), Inches(1.85), Inches(5.5), Inches(0.4), font_size=20, bold=True, color=ACCENT_CORAL)

lim_bullets = [
    "NO clinical validation performed — system NOT tested with real ASD learners",
    "All training/validation used synthetic ASDLearn-Bench dataset only",
    "NO claims of therapeutic or clinical effectiveness",
    "NO authentication/access-control system — login is simple student-ID dropdown",
    "Authentication was deliberate scope decision: production would use institutional SSO",
    "Models do NOT retrain weights continuously in real-time",
    "DKT mastery only recalculated and persisted after session ENDS",
    "Synthetic data limitation requires future real-world validation"
]
add_bullet_list(slide11, lim_bullets, Inches(0.8), Inches(2.3), Inches(5.5), Inches(4.0), font_size=14, color=LIGHT_TEXT)

# Future Work column
fw_card = slide11.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.7), Inches(5.9), Inches(5.0))
fw_card.fill.solid(); fw_card.fill.fore_color.rgb = CARD_BG; fw_card.line.color.rgb = ACCENT_TEAL; fw_card.line.width = Pt(2)
add_title_box(slide11, "Future Work", Inches(7.0), Inches(1.85), Inches(5.5), Inches(0.4), font_size=20, bold=True, color=ACCENT_TEAL)

fw_bullets = [
    "Real-world pilot study with ASD learners and caregivers",
    "Integration with institutional SSO for production deployment",
    "Therapist/human oversight dashboard for caregiver monitoring",
    "Continuous model improvement with real interaction data",
    "Expansion beyond 71 tasks with richer content variety",
    "Multimodal behavioral sensing (eye tracking, voice emotion)",
    "Longitudinal mastery tracking across months/years",
    "Localization for Bengali language support"
]
add_bullet_list(slide11, fw_bullets, Inches(7.0), Inches(2.3), Inches(5.5), Inches(4.0), font_size=14, color=LIGHT_TEXT)

# ============================================================
# SLIDE 12: Closing / Thank You / Q&A
# ============================================================
slide12 = prs.slides.add_slide(prs.slide_layouts[6])
add_background(slide12)

add_accent_bar(slide12, Inches(0), Inches(2.8), Inches(0.12), Inches(2.0), ACCENT_TEAL)
add_accent_bar(slide12, Inches(13.21), Inches(2.8), Inches(0.12), Inches(2.0), ACCENT_CORAL)

add_title_box(slide12, "Thank You", Inches(0.8), Inches(2.4), Inches(11.5), Inches(1.0), font_size=64, bold=True, color=LIGHT_TEXT, align=PP_ALIGN.CENTER)
add_title_box(slide12, "Questions & Discussion", Inches(0.8), Inches(3.4), Inches(11.5), Inches(0.5), font_size=28, bold=False, color=ACCENT_TEAL, align=PP_ALIGN.CENTER)

# Contact info
contact_box = slide12.shapes.add_textbox(Inches(0.8), Inches(4.5), Inches(11.5), Inches(2.0))
tf = contact_box.text_frame
tf.word_wrap = True

p = tf.paragraphs[0]
p.text = "Cognify: An Adaptive Learning System For Neurodiverse Users"
p.font.size = Pt(18)
p.font.color.rgb = MUTED_TEXT
p.font.name = "Calibri"
p.alignment = PP_ALIGN.CENTER

p = tf.add_paragraph()
p.text = "Department of Electrical and Computer Engineering, North South University"
p.font.size = Pt(16)
p.font.color.rgb = MUTED_TEXT
p.font.name = "Calibri"
p.alignment = PP_ALIGN.CENTER
p.space_before = Pt(8)

p = tf.add_paragraph()
p.text = "Team: Mujtaba Muhammad Abrar (Lead), Tahfeem Islam Siam, Nova Khan"
p.font.size = Pt(16)
p.font.color.rgb = LIGHT_TEXT
p.font.name = "Calibri"
p.alignment = PP_ALIGN.CENTER
p.space_before = Pt(16)

p = tf.add_paragraph()
p.text = "Faculty Advisor: Dr. Shahnewaz Siddique"
p.font.size = Pt(16)
p.font.color.rgb = LIGHT_TEXT
p.font.name = "Calibri"
p.alignment = PP_ALIGN.CENTER
p.space_before = Pt(4)

p = tf.add_paragraph()
p.text = "Senior Design Project (Capstone 499A/499B) | Summer 2026"
p.font.size = Pt(14)
p.font.color.rgb = MUTED_TEXT
p.font.name = "Calibri"
p.alignment = PP_ALIGN.CENTER
p.space_before = Pt(12)

# Save
prs.save('Cognify_Presentation.pptx')
print("✅ Presentation saved as 'Cognify_Presentation.pptx'")
print(f"Total slides: {len(prs.slides)}")