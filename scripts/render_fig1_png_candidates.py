#!/usr/bin/env python3
"""Render reviewer-facing PNG candidates for paper Figure 1.

These are fast design previews. After choosing a direction, port the winning
layout to vector PDF/TikZ or a carefully verified Matplotlib PDF.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "icml2026_draft" / "figures" / "candidates_png"
W, H = 2200, 1000

BLUE = (34, 122, 184)
BLUE_DARK = (18, 65, 96)
RED = (184, 58, 58)
GRAY = (142, 142, 142)
DARK = (35, 35, 35)
MID = (90, 90, 90)
LIGHT = (247, 247, 247)
PALE_BLUE = (232, 244, 252)
PALE_GREEN = (232, 246, 232)
GREEN = (52, 128, 56)
BORDER = (190, 190, 190)

SERIF = "/System/Library/Fonts/Times.ttc"
SANS = "/System/Library/Fonts/Helvetica.ttc"
MONO = "/System/Library/Fonts/SFNSMono.ttf"


def font(size: int, kind: str = "serif") -> ImageFont.FreeTypeFont:
    path = {"serif": SERIF, "sans": SANS, "mono": MONO}[kind]
    return ImageFont.truetype(path, size=size)


def text(draw, xy, s, size=32, fill=DARK, kind="serif", anchor=None, **kwargs):
    draw.text(xy, s, font=font(size, kind), fill=fill, anchor=anchor, **kwargs)


def rounded(draw, box, radius=28, fill=(255, 255, 255), outline=BORDER, width=4):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow(draw, p1, p2, fill=MID, width=6):
    draw.line([p1, p2], fill=fill, width=width)
    x1, y1 = p1
    x2, y2 = p2
    # Simple arrowhead for mostly horizontal/vertical arrows.
    if abs(x2 - x1) > abs(y2 - y1):
        sign = 1 if x2 > x1 else -1
        pts = [(x2, y2), (x2 - sign * 22, y2 - 14), (x2 - sign * 22, y2 + 14)]
    else:
        sign = 1 if y2 > y1 else -1
        pts = [(x2, y2), (x2 - 14, y2 - sign * 22), (x2 + 14, y2 - sign * 22)]
    draw.polygon(pts, fill=fill)


def person(draw, x, y, scale=1.0, color=GRAY, width=8):
    """Human icon. x,y is foot baseline center."""
    head_r = int(15 * scale)
    head_y = y - int(92 * scale)
    shoulder_y = y - int(68 * scale)
    hip_y = y - int(32 * scale)
    arm = int(34 * scale)
    leg = int(28 * scale)
    draw.ellipse((x - head_r, head_y - head_r, x + head_r, head_y + head_r), fill=color, outline=(255, 255, 255), width=max(2, int(2 * scale)))
    draw.line((x, shoulder_y, x, hip_y), fill=color, width=max(4, int(width * scale)))
    draw.line((x - arm, shoulder_y + 6, x + arm, shoulder_y + 6), fill=color, width=max(3, int(width * 0.65 * scale)))
    draw.line((x, hip_y, x - leg, y), fill=color, width=max(3, int(width * 0.65 * scale)))
    draw.line((x, hip_y, x + leg, y), fill=color, width=max(3, int(width * 0.65 * scale)))


def base():
    return Image.new("RGB", (W, H), "white"), None


def save(img, name):
    OUT.mkdir(parents=True, exist_ok=True)
    img.save(OUT / f"{name}.png")


def candidate_1():
    img, _ = base()
    d = ImageDraw.Draw(img)
    text(d, (70, 50), "Candidate 1: experiment in one glance", 30, MID, "sans")
    rounded(d, (90, 145, 680, 870), fill=LIGHT)
    rounded(d, (790, 145, 1370, 870), fill=(255, 255, 255))
    rounded(d, (1480, 145, 2110, 870), fill=(255, 255, 255))
    arrow(d, (700, 510), (770, 510))
    arrow(d, (1390, 510), (1460, 510))

    text(d, (130, 195), "15-shot prompt", 54)
    text(d, (130, 260), "Context examples define the comparison class.", 30, MID, "sans")
    heights = [153, 157, 160, 162, 165, 166, 168, 170, 172, 174, 176, 179, 181, 184, 187]
    for i, h in enumerate(heights):
        col, row = i % 5, i // 5
        person(d, 175 + col * 90, 560 + row * 100, 0.58 + (h - 153) / 115, GRAY, 8)
    d.rectangle((160, 735, 610, 825), fill=PALE_BLUE, outline=BLUE, width=5)
    text(d, (185, 760), "Target: 185 cm", 34, BLUE_DARK)
    text(d, (185, 795), "This person is ___", 30, BLUE_DARK)

    text(d, (835, 195), "Normalize by context", 48)
    d.rectangle((890, 355, 1255, 625), fill=PALE_GREEN, outline=(185, 222, 185), width=4)
    d.line((890, 490, 1255, 490), fill=MID, width=4)
    text(d, (1268, 477), "mean = 170", 28, DARK)
    person(d, 1070, 670, 1.1, BLUE, 9)
    arrow(d, (1190, 490), (1190, 370), BLUE, 5)
    arrow(d, (1190, 370), (1190, 490), BLUE, 5)
    text(d, (1210, 420), "x - mean", 28, BLUE_DARK, "sans")
    text(d, (875, 760), "z = (185 - 170) / 10 = +1.5", 36, DARK)
    text(d, (875, 808), "positive z = above this context", 28, MID, "sans")

    text(d, (1530, 195), "Compare completions", 48)
    text(d, (1530, 265), "Next-token logits:", 30, MID, "sans")
    d.rectangle((1565, 390, 1735, 480), fill=(250, 235, 235), outline=RED, width=5)
    text(d, (1610, 415), "short", 34, RED)
    text(d, (1760, 420), "low", 30, MID, "sans")
    d.rectangle((1565, 555, 1900, 650), fill=PALE_BLUE, outline=BLUE, width=6)
    text(d, (1610, 580), "tall", 38, BLUE_DARK)
    text(d, (1800, 585), "high", 30, MID, "sans")
    text(d, (1530, 765), "Δlogit = logit(tall) - logit(short)", 34, DARK)
    text(d, (1530, 815), "Question: does Δlogit track z rather than raw x?", 28, MID, "sans")
    save(img, "candidate_1_experiment_glance")


def candidate_2():
    img, _ = base()
    d = ImageDraw.Draw(img)
    text(d, (70, 50), "Candidate 2: relativity contrast", 30, MID, "sans")
    text(d, (150, 130), "The same 185 cm person can be tall or short depending on context", 56)
    panels = [
        (130, 250, 1000, 820, 170, BLUE, "ordinary-height context", "tall favored", "+1.5"),
        (1200, 250, 2070, 820, 195, RED, "basketball-team context", "short favored", "-1.0"),
    ]
    for x1, y1, x2, y2, mu, color, title, verdict, zval in panels:
        rounded(d, (x1, y1, x2, y2), fill=(255, 255, 255))
        text(d, (x1 + 45, y1 + 40), title, 40)
        d.rectangle((x1 + 85, y1 + 195, x2 - 85, y1 + 345), fill=PALE_GREEN, outline=(185, 222, 185), width=4)
        d.line((x1 + 430, y1 + 130, x1 + 430, y1 + 420), fill=MID, width=5)
        text(d, (x1 + 390, y1 + 105), f"μ = {mu}", 34)
        for i in range(9):
            scale = 0.58 + i * 0.055 + (0.28 if mu == 195 else 0)
            person(d, x1 + 135 + i * 70, y1 + 450, scale, GRAY, 8)
        person(d, x2 - 145, y1 + 450, 1.05, color, 10)
        text(d, (x2 - 190, y1 + 475), "185", 30, color)
        text(d, (x1 + 105, y2 - 105), f"target x = 185, z = {zval}", 34, DARK)
        text(d, (x2 - 360, y1 + 150), verdict, 42, color)
    text(d, (365, 900), "Dense-grid experiment: choose x and z independently, then set μ = x - zσ.", 36, DARK)
    save(img, "candidate_2_relativity_contrast")


def candidate_3():
    img, _ = base()
    d = ImageDraw.Draw(img)
    text(d, (70, 50), "Candidate 3: dense-grid method", 30, MID, "sans")
    text(d, (155, 120), "How the experiment separates raw value from relative standing", 56)
    rounded(d, (90, 230, 660, 800), fill=LIGHT)
    rounded(d, (820, 230, 1340, 800), fill=(255, 255, 255))
    rounded(d, (1500, 230, 2110, 800), fill=(255, 255, 255))
    arrow(d, (680, 520), (795, 520))
    arrow(d, (1360, 520), (1475, 520))
    text(d, (145, 295), "1. Choose grid cell", 42)
    d.rectangle((160, 405, 575, 570), fill=PALE_BLUE, outline=BLUE, width=5)
    text(d, (190, 435), "raw value x = 185 cm", 34, BLUE_DARK)
    text(d, (190, 495), "relative standing z = +1.5", 34, BLUE_DARK)
    text(d, (870, 295), "2. Build context", 42)
    text(d, (885, 415), "μ = x - zσ", 42, DARK)
    text(d, (885, 480), "= 185 - 1.5·10", 36, DARK)
    text(d, (885, 540), "= 170", 42, DARK)
    text(d, (1545, 295), "3. Test completion", 42)
    for i, h in enumerate([153, 157, 160, 162, 165, 166, 168, 170, 172, 174, 176, 179, 181, 184, 187]):
        person(d, 1570 + (i % 5) * 86, 540 + (i // 5) * 75, 0.48 + (h - 153) / 140, GRAY, 7)
    person(d, 1980, 660, 1.0, BLUE, 9)
    text(d, (1545, 720), "Compare logits: tall vs short", 34, DARK)
    text(d, (1545, 765), "Δlogit should rise with z", 30, BLUE_DARK, "sans")
    save(img, "candidate_3_dense_grid_method")


def candidate_4():
    img, _ = base()
    d = ImageDraw.Draw(img)
    text(d, (70, 50), "Candidate 4: paper-clean single panel", 30, MID, "sans")
    text(d, (155, 120), "Context makes 185 cm count as tall", 60)
    rounded(d, (120, 215, 2080, 835), fill=(255, 255, 255))
    d.rectangle((305, 405, 1620, 610), fill=PALE_GREEN, outline=(185, 222, 185), width=4)
    d.line((950, 340, 950, 675), fill=MID, width=5)
    text(d, (900, 300), "mean = 170", 34)
    text(d, (1320, 365), "+1σ", 34, GREEN)
    for i, h in enumerate([153, 157, 160, 162, 165, 166, 168, 170, 172, 174, 176, 179, 181, 184, 187]):
        person(d, 350 + i * 78, 700, 0.55 + (h - 153) / 100, GRAY, 8)
    person(d, 1780, 700, 1.08, BLUE, 11)
    text(d, (1715, 335), "target x = 185", 42, BLUE_DARK)
    arrow(d, (1735, 390), (1772, 520), BLUE, 6)
    arrow(d, (1875, 515), (1875, 405), BLUE, 5)
    arrow(d, (1875, 405), (1875, 515), BLUE, 5)
    text(d, (1895, 450), "x - mean", 30, BLUE_DARK, "sans")
    d.rectangle((145, 730, 655, 805), fill=PALE_BLUE, outline=BLUE, width=5)
    text(d, (175, 750), "Prompt: 185 cm person is ___", 30, BLUE_DARK, "sans")
    d.rectangle((780, 730, 1135, 805), fill=(255, 255, 255), outline=BORDER, width=4)
    text(d, (805, 750), "z = (185-170)/10 = +1.5", 29, DARK)
    d.rectangle((1300, 730, 1540, 805), fill=PALE_BLUE, outline=BLUE, width=6)
    text(d, (1345, 750), "tall", 36, BLUE_DARK)
    text(d, (1580, 750), "favored over short", 30, DARK, "sans")
    save(img, "candidate_4_paper_clean")


def double_arrow(draw, p1, p2, fill=BLUE, width=5):
    arrow(draw, p1, p2, fill, width)
    arrow(draw, p2, p1, fill, width)


def context_panel(draw, box, title, mu, target_x, z_text, target_color, target_side, inequality):
    x1, y1, x2, y2 = box
    rounded(draw, box, fill=(255, 255, 255), radius=26, width=4)
    text(draw, (x1 + 38, y1 + 30), title, 34)
    band = (x1 + 72, y1 + 155, x2 - 72, y1 + 310)
    draw.rectangle(band, fill=PALE_GREEN, outline=(185, 222, 185), width=4)
    mean_x = x1 + (x2 - x1) // 2
    draw.line((mean_x, y1 + 95, mean_x, y1 + 410), fill=MID, width=5)
    text(draw, (mean_x - 42, y1 + 62), f"μ = {mu}", 30)

    people_base = y1 + 392
    target_scale = 1.02
    if target_side == "right":
        gray_xs = [x1 + 135 + i * 66 for i in range(9)]
        gray_scales = [0.58 + i * 0.055 for i in range(9)]
        tx = x2 - 140
        offset_y = y1 + 270
        offset_start = mean_x + 15
        offset_end = tx - 18
        offset_label = "x - μ > 0"
    else:
        # Same target value, but now it is below the comparison-class mean.
        tx = x1 + 170
        offset_y = y1 + 270
        offset_start = tx + 18
        offset_end = mean_x - 15
        offset_label = "x - μ < 0"
        gray_xs = [x1 + 260 + i * 64 for i in range(9)]
        gray_scales = [0.86 + i * 0.045 for i in range(9)]

    for gx, gs in zip(gray_xs, gray_scales):
        person(draw, gx, people_base, gs, GRAY, 8)
    person(draw, tx, people_base, target_scale, target_color, 10)
    text(draw, (tx - 34, people_base + 22), "185", 25, target_color)
    double_arrow(draw, (offset_start, offset_y), (offset_end, offset_y), target_color, 4)
    text(
        draw,
        ((offset_start + offset_end) // 2 - 45, offset_y - 42),
        offset_label,
        23,
        target_color,
        "sans",
    )

    # Keep explanatory labels off the bodies.
    label_y = y2 - 95
    draw.rectangle((x1 + 65, label_y - 8, x2 - 65, y2 - 28), fill=(255, 255, 255))
    text(draw, (x1 + 88, label_y), f"target x = 185, z = {z_text}", 27, DARK)
    text(draw, (x1 + 88, label_y + 40), inequality, 27, target_color, "sans")


def candidate_5_combined():
    """Combined candidate: task pipeline plus relativity contrast."""
    img = Image.new("RGB", (2200, 1500), "white")
    d = ImageDraw.Draw(img)

    text(d, (115, 70), "Context-relative adjective judgments: from prompt to logits", 58)

    # Section A: task pipeline.
    text(d, (115, 205), "A. Task pipeline", 34, DARK, "sans")
    rounded(d, (115, 255, 695, 690), fill=LIGHT, radius=26, width=4)
    rounded(d, (820, 255, 1380, 690), fill=(255, 255, 255), radius=26, width=4)
    rounded(d, (1505, 255, 2085, 690), fill=(255, 255, 255), radius=26, width=4)
    arrow(d, (720, 480), (790, 480), MID, 6)
    arrow(d, (1405, 480), (1475, 480), MID, 6)

    text(d, (155, 292), "15-shot prompt", 42)
    text(d, (155, 345), "Examples define the comparison class", 28, MID, "sans")
    heights = [153, 157, 160, 162, 165, 166, 168, 170, 172, 174, 176, 179, 181, 184, 187]
    for i, h in enumerate(heights):
        col, row = i % 5, i // 5
        person(d, 185 + col * 86, 482 + row * 52, 0.40 + (h - 153) / 175, GRAY, 7)
    d.rectangle((155, 595, 635, 655), fill=PALE_BLUE, outline=BLUE, width=5)
    text(d, (178, 609), "P16: 185 cm. This person is ___", 26, BLUE_DARK, "sans")

    text(d, (860, 292), "Normalize by context", 40)
    d.rectangle((900, 365, 1275, 550), fill=PALE_GREEN, outline=(185, 222, 185), width=4)
    d.line((900, 458, 1275, 458), fill=MID, width=4)
    text(d, (1288, 445), "mean = 170", 25, DARK)
    # Use a marker rather than a human here: this panel is about the variable,
    # not a second target person.
    d.ellipse((1056, 382, 1100, 426), fill=BLUE, outline=(255, 255, 255), width=3)
    text(d, (1018, 336), "target x = 185", 26, BLUE_DARK, "sans")
    double_arrow(d, (1208, 458), (1208, 392), BLUE, 5)
    text(d, (1228, 412), "x - mean", 25, BLUE_DARK, "sans")
    text(d, (900, 625), "z = (185 - 170) / 10 = +1.5", 30, DARK)

    text(d, (1545, 292), "Read out adjective logits", 40)
    d.rectangle((1575, 382, 1745, 458), fill=(250, 235, 235), outline=RED, width=5)
    text(d, (1620, 402), "short", 30, RED)
    d.rectangle((1575, 508, 1900, 588), fill=PALE_BLUE, outline=BLUE, width=6)
    text(d, (1620, 530), "tall", 34, BLUE_DARK)
    text(d, (1925, 528), "higher logit", 25, MID, "sans")
    text(d, (1575, 630), "Δlogit = logit(tall) - logit(short)", 28, DARK)

    # Section B: relativity contrast.
    text(d, (115, 780), "B. Relativity intuition", 34, DARK, "sans")
    text(d, (115, 825), "The same raw value can support different completions under different comparison classes.", 32, MID, "sans")
    context_panel(
        d,
        (135, 895, 1035, 1390),
        "ordinary-height context",
        170,
        185,
        "+1.5",
        BLUE,
        "right",
        "logit(tall) > logit(short)",
    )
    context_panel(
        d,
        (1165, 895, 2065, 1390),
        "tall-team context",
        195,
        185,
        "-1.0",
        RED,
        "left",
        "logit(short) > logit(tall)",
    )
    text(d, (365, 1440), "Dense-grid design: choose x and z independently, then set μ = x - zσ.", 32, DARK)
    save(img, "candidate_5_combined_pipeline_relcontrast")


def candidate_6_paper_scale():
    """Paper-scale version: fewer words, larger labels, same visual story."""
    img = Image.new("RGB", (2200, 1250), "white")
    d = ImageDraw.Draw(img)

    text(d, (95, 55), "Prompt -> relative standing -> adjective logits", 66)

    # A: compact pipeline, large labels.
    y_top = 160
    box_h = 360
    rounded(d, (105, y_top, 675, y_top + box_h), fill=LIGHT, radius=26, width=5)
    rounded(d, (820, y_top, 1380, y_top + box_h), fill=(255, 255, 255), radius=26, width=5)
    rounded(d, (1525, y_top, 2095, y_top + box_h), fill=(255, 255, 255), radius=26, width=5)
    arrow(d, (700, y_top + 190), (790, y_top + 190), MID, 8)
    arrow(d, (1405, y_top + 190), (1495, y_top + 190), MID, 8)

    text(d, (145, y_top + 35), "15-shot context", 46)
    for i, h in enumerate([153, 157, 160, 162, 165, 166, 168, 170, 172, 174, 176, 179]):
        col, row = i % 6, i // 6
        person(d, 170 + col * 75, y_top + 235 + row * 55, 0.44 + (h - 153) / 170, GRAY, 8)
    d.rectangle((145, y_top + 270, 625, y_top + 330), fill=PALE_BLUE, outline=BLUE, width=6)
    text(d, (170, y_top + 286), "Target: 185 cm -> ___", 31, BLUE_DARK, "sans")

    text(d, (860, y_top + 35), "Compute z", 46)
    d.rectangle((900, y_top + 105, 1285, y_top + 250), fill=PALE_GREEN, outline=(185, 222, 185), width=5)
    d.line((900, y_top + 180, 1285, y_top + 180), fill=MID, width=5)
    text(d, (1298, y_top + 166), "μ=170", 30, DARK)
    d.ellipse((1062, y_top + 88, 1112, y_top + 138), fill=BLUE, outline=(255, 255, 255), width=4)
    double_arrow(d, (1215, y_top + 180), (1215, y_top + 110), BLUE, 6)
    text(d, (1237, y_top + 132), "x-μ", 31, BLUE_DARK, "sans")
    text(d, (925, y_top + 290), "z = (185-170)/10 = +1.5", 34, DARK)

    text(d, (1565, y_top + 35), "Read logits", 46)
    d.rectangle((1595, y_top + 105, 1775, y_top + 180), fill=(250, 235, 235), outline=RED, width=6)
    text(d, (1640, y_top + 125), "short", 32, RED)
    d.rectangle((1595, y_top + 220, 1945, y_top + 300), fill=PALE_BLUE, outline=BLUE, width=7)
    text(d, (1640, y_top + 242), "tall", 38, BLUE_DARK)
    text(d, (1962, y_top + 242), "higher", 30, MID, "sans")

    # B: relativity contrast, large enough for column view.
    text(d, (105, 600), "Same x, different context", 52)
    context_panel_large(
        d,
        (125, 685, 1040, 1135),
        "ordinary context",
        170,
        BLUE,
        "right",
        "+1.5",
        "logit(tall) > logit(short)",
    )
    context_panel_large(
        d,
        (1160, 685, 2075, 1135),
        "tall-team context",
        195,
        RED,
        "left",
        "-1.0",
        "logit(short) > logit(tall)",
    )
    text(d, (460, 1195), "Dense grid: choose x and z independently, then set μ = x - zσ.", 36, DARK)
    save(img, "candidate_6_paper_scale")


def context_panel_large(draw, box, title, mu, target_color, target_side, z_text, inequality):
    x1, y1, x2, y2 = box
    rounded(draw, box, fill=(255, 255, 255), radius=26, width=5)
    text(draw, (x1 + 35, y1 + 24), title, 36)
    draw.rectangle((x1 + 80, y1 + 135, x2 - 80, y1 + 255), fill=PALE_GREEN, outline=(185, 222, 185), width=5)
    mean_x = x1 + (x2 - x1) // 2
    draw.line((mean_x, y1 + 90, mean_x, y1 + 345), fill=MID, width=6)
    text(draw, (mean_x - 40, y1 + 58), f"μ={mu}", 30)
    people_y = y1 + 340
    if target_side == "right":
        tx = x2 - 140
        gray_xs = [x1 + 140 + i * 64 for i in range(8)]
        gray_scales = [0.60 + i * 0.055 for i in range(8)]
        arrow_start, arrow_end = mean_x + 18, tx - 22
        offset_text = "x-μ>0"
    else:
        tx = x1 + 145
        gray_xs = [x1 + 255 + i * 62 for i in range(9)]
        gray_scales = [0.82 + i * 0.045 for i in range(9)]
        arrow_start, arrow_end = tx + 22, mean_x - 18
        offset_text = "x-μ<0"
    for gx, gs in zip(gray_xs, gray_scales):
        person(draw, gx, people_y, gs, GRAY, 8)
    person(draw, tx, people_y, 1.0, target_color, 11)
    double_arrow(draw, (arrow_start, y1 + 225), (arrow_end, y1 + 225), target_color, 6)
    text(draw, ((arrow_start + arrow_end) // 2 - 40, y1 + 180), offset_text, 28, target_color, "sans")
    draw.rectangle((x1 + 60, y2 - 82, x2 - 60, y2 - 18), fill=(255, 255, 255))
    text(draw, (x1 + 85, y2 - 72), f"z={z_text}", 30, DARK)
    text(draw, (x1 + 220, y2 - 72), inequality, 30, target_color, "sans")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    candidate_1()
    candidate_2()
    candidate_3()
    candidate_4()
    candidate_5_combined()
    candidate_6_paper_scale()


if __name__ == "__main__":
    main()
