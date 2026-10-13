"""Build docs/shipdas_guide.pdf, a slide-style walkthrough of the whole project.

Run after docs/make_figures.py. Code snippets are cut from the real source files,
so the guide always shows the code as it is.
"""

import ast
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "docs/figures"
OUT = ROOT / "docs/shipdas_guide.pdf"
W, H = 960, 540

BG, INK, MUTED, LINE = HexColor("#fbfaf7"), HexColor("#3d3b45"), HexColor("#8b8994"), HexColor("#e9e6ee")
BLUE, PINK, MINT = HexColor("#dcebf8"), HexColor("#fbe3e9"), HexColor("#e0f3ea")
PEACH, LAV, LEMON = HexColor("#fdebdc"), HexColor("#ece3f4"), HexColor("#fcf4d4")
DEEP = {"blue": HexColor("#6f9cc9"), "pink": HexColor("#d98aa0"), "mint": HexColor("#5fae8c"),
        "lav": HexColor("#9b83b8"), "peach": HexColor("#d99a63")}

BODY = ParagraphStyle("body", fontName="Helvetica", fontSize=13.5, leading=19, textColor=INK)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=11, leading=15)
TINY = ParagraphStyle("tiny", parent=BODY, fontSize=9.5, leading=12.5, textColor=MUTED)
NOTE = ParagraphStyle("note", parent=BODY, fontSize=10.5, leading=14, alignment=TA_LEFT)

c = canvas.Canvas(str(OUT), pagesize=(W, H))
c.setTitle("shipdas: listening for ships with a submarine cable")
c.setAuthor("sriramachellu")
page_no = 0


# ---------- drawing helpers ----------

def new_page(title, section, color=LAV):
    global page_no
    if page_no:
        c.showPage()
    page_no += 1
    c.setFillColor(BG)
    c.rect(0, 0, W, H, stroke=0, fill=1)
    c.setFillColor(color)
    c.roundRect(40, H - 58, c.stringWidth(section, "Helvetica-Bold", 10) + 20, 22, 11, stroke=0, fill=1)
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, H - 51, section)
    c.setFont("Helvetica-Bold", 26)
    c.drawString(40, H - 100, title)
    c.setFont("Helvetica", 9)
    c.setFillColor(MUTED)
    c.drawRightString(W - 40, 24, f"{page_no}")
    c.drawString(40, 24, "shipdas  |  vessel detection from submarine cable DAS")


def para(text, x, y_top, width, style=BODY):
    p = Paragraph(text, style)
    _, h = p.wrap(width, 1000)
    p.drawOn(c, x, y_top - h)
    return y_top - h


def bullets(items, x, y_top, width, style=BODY, gap=8, dot=DEEP["lav"]):
    for it in items:
        c.setFillColor(dot)
        c.circle(x + 4, y_top - style.leading / 2 - 1, 3, stroke=0, fill=1)
        y_top = para(it, x + 16, y_top, width - 16, style) - gap
    return y_top


def image(name, x, y, w=None, h=None):
    img = ImageReader(str(FIG / name))
    iw, ih = img.getSize()
    if w and not h:
        h = w * ih / iw
    elif h and not w:
        w = h * iw / ih
    c.drawImage(img, x, y, w, h, mask="auto")
    return w, h


def card(x, y, w, h, fill, title=None, text=None, style=SMALL):
    c.setFillColor(fill)
    c.roundRect(x, y, w, h, 12, stroke=0, fill=1)
    top = y + h - 14
    if title:
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold", 13)
        c.drawString(x + 14, top - 10, title)
        top -= 22
    if text:
        para(text, x + 14, top, w - 28, style)


def box(x, y, w, h, label, fill, sub=None):
    c.setFillColor(fill)
    c.roundRect(x, y, w, h, 10, stroke=0, fill=1)
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(x + w / 2, y + h / 2 + (4 if sub else -4), label)
    if sub:
        c.setFont("Helvetica", 9.5)
        c.setFillColor(MUTED)
        c.drawCentredString(x + w / 2, y + h / 2 - 12, sub)


def arrow(x1, y1, x2, y2, color=MUTED):
    import math
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(1.4)
    c.line(x1, y1, x2, y2)
    a = math.atan2(y2 - y1, x2 - x1)
    pts = [x2, y2, x2 - 9 * math.cos(a - 0.4), y2 - 9 * math.sin(a - 0.4),
           x2 - 9 * math.cos(a + 0.4), y2 - 9 * math.sin(a + 0.4)]
    p = c.beginPath()
    p.moveTo(pts[0], pts[1])
    p.lineTo(pts[2], pts[3])
    p.lineTo(pts[4], pts[5])
    p.close()
    c.drawPath(p, stroke=0, fill=1)


def code(lines, x, y_top, w, size=9.2, fill=LAV):
    lh = size * 1.42
    h = lh * len(lines) + 18
    c.setFillColor(fill)
    c.roundRect(x, y_top - h, w, h, 10, stroke=0, fill=1)
    for i, line in enumerate(lines):
        yy = y_top - 14 - i * lh
        c.setFont("Courier", size - 1)
        c.setFillColor(MUTED)
        c.drawRightString(x + 24, yy, str(i + 1))
        c.setFont("Courier", size)
        c.setFillColor(INK)
        c.drawString(x + 32, yy, line)
    return y_top - h


def explain(notes, x, y_top, w):
    """notes: list of (line label, text)."""
    for label, text in notes:
        c.setFillColor(DEEP["lav"])
        c.setFont("Helvetica-Bold", 10)
        c.drawString(x, y_top - 11, label)
        y_top = para(text, x + 46, y_top, w - 46, NOTE) - 6
    return y_top


def source(path, name, start=None, stop=None):
    """Lines of a top-level or nested function from a real source file."""
    text = (ROOT / path).read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            lines = text.splitlines()[node.lineno - 1:node.end_lineno]
            return lines[start:stop]
    raise KeyError(name)


def drop_docstring(lines):
    out, inside = [], False
    for ln in lines:
        s = ln.strip()
        if s.startswith('"""'):
            if not (s.endswith('"""') and len(s) > 3):
                inside = not inside
            continue
        if inside:
            if s.endswith('"""'):
                inside = False
            continue
        out.append(ln)
    return out


# ---------- slides ----------

# 1. title
page_no = 1
c.setFillColor(BG)
c.rect(0, 0, W, H, stroke=0, fill=1)
for i, col in enumerate([BLUE, PINK, MINT, PEACH, LAV]):
    c.setFillColor(col)
    c.circle(760 + (i % 3) * 70, 330 + (i // 3) * 90 - i * 8, 70 - i * 6, stroke=0, fill=1)
c.setFillColor(INK)
c.setFont("Helvetica-Bold", 40)
c.drawString(60, 330, "Listening for ships")
c.drawString(60, 284, "with an undersea cable")
para("An end-to-end guide to <b>shipdas</b>: vessel detection and distance estimation from "
     "Distributed Acoustic Sensing (DAS) on a submarine fibre-optic cable. What the data is, "
     "how every step works, why each model was chosen, how to run it, and what we found.",
     60, 250, 560, BODY)
c.setFont("Helvetica", 11)
c.setFillColor(MUTED)
c.drawString(60, 80, "Dataset: Marlinks-NS (Zenodo, CC BY 4.0)   |   Code: github.com/sriramachellu/shipdas")
c.drawString(60, 62, "Hardware: laptop RTX 3060 (6 GB), 16 GB RAM")

# 2. contents
new_page("How to read this guide", "Overview")
parts = [("1  The problem", "Why cables need protecting and what DAS is", BLUE),
         ("2  The dataset", "Marlinks-NS: what one sample looks like and what the labels mean", PINK),
         ("3  Evaluation", "Day-wise cross-validation and the metrics", MINT),
         ("4  Pipeline and code", "Each script, with snippets explained line by line", PEACH),
         ("5  Models", "Why XGBoost, why a CNN, why a cascade of both", LAV),
         ("6  Results", "Headline numbers, what each step added, where it fails", LEMON),
         ("7  Run it yourself", "Commands, timings, hardware notes and lessons", BLUE)]
for k, (t, s, col) in enumerate(parts):
    x, y = 40 + (k % 2) * 450, 380 - (k // 2) * 82
    card(x, y, 430, 68, col, t, s)

# 3. problem
new_page("The problem: cables break, and we rarely see it coming", "1  The problem", BLUE)
bullets([
    "About <b>99% of intercontinental internet traffic</b> runs through submarine fibre-optic cables.",
    "Most faults are caused by people, not nature: <b>ship anchors dragged across the seabed and "
    "fishing gear</b> such as trawls.",
    "A repair needs a specialised ship and can take weeks. Knowing that a vessel is close to the "
    "cable, and how close, lets an operator warn it before damage happens.",
    "Ships report their position through <b>AIS</b>, but AIS can be switched off. We want a "
    "sensor that does not depend on the ship cooperating.",
    "<b>Goal of this project:</b> from the cable's own vibration data, (1) decide whether a vessel "
    "is within 1 km of the cable and (2) estimate its distance.",
], 40, 410, 520)
card(600, 150, 320, 250, LAV, "Two tasks, one input",
     "<b>Detection</b> (classification): is the closest vessel within 1,000 m?<br/><br/>"
     "<b>Localisation</b> (regression): how far is it, in metres?<br/><br/>"
     "Input for both: 10 seconds of vibration energy measured along 2.5 km of cable.")

# 4. DAS
new_page("How a fibre becomes 2,500 microphones", "1  The problem", BLUE)
box(50, 300, 150, 70, "Interrogator", PEACH, "laser + detector, on shore")
c.setStrokeColor(DEEP["blue"])
c.setLineWidth(5)
c.line(200, 335, 900, 335)
for k in range(12):
    c.setFillColor(PINK)
    c.circle(250 + k * 55, 335, 6, stroke=0, fill=1)
c.setFillColor(MUTED)
c.setFont("Helvetica", 10)
c.drawString(250, 350, "each dot = one sensing channel, every 10.21 m")
arrow(880, 318, 220, 318, DEEP["pink"])
c.drawString(480, 300, "backscattered light comes back")
bullets([
    "A laser pulse travels down the fibre. Tiny imperfections in the glass reflect a little light "
    "back (Rayleigh backscatter).",
    "When sound or vibration stretches the fibre, the phase of that reflected light changes. "
    "Timing tells us <b>where</b> along the fibre it happened.",
    "So one ordinary telecom fibre becomes thousands of virtual microphones. This is "
    "<b>Distributed Acoustic Sensing</b>.",
    "A ship's engine and propeller make low-frequency noise that reaches the seabed cable.",
], 40, 260, 880, SMALL, gap=5)

# 5. dataset facts
new_page("The dataset: Marlinks-NS", "2  The dataset", PINK)
rows = [("Where", "28 km cable off Zeebrugge, Belgium (North Sea)"),
        ("When", "16 to 25 June 2023, continuous, 10 days"),
        ("Instrument", "Alcatel OptoDAS, sampling 3,125 Hz"),
        ("Segment used", "2,553 m of cable, 250 channels (247 after removing 3 noisy ones)"),
        ("One sample", "10 seconds, summarised as energy in 100 frequency bands, 4 to 98 Hz"),
        ("Samples", "74,771 windows, X has shape (74771, 250, 100)"),
        ("Label y", "distance in metres to the closest vessel reported by AIS"),
        ("Extra", "ship type, length and beam of that vessel; daily weather"),
        ("Size", "14.1 GB HDF5, float64; licence CC BY 4.0")]
y0 = 410
for k, (a, b) in enumerate(rows):
    c.setFillColor(PINK if k % 2 == 0 else BG)
    c.rect(40, y0 - 30 * k - 24, 560, 30, stroke=0, fill=1)
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 11.5)
    c.drawString(52, y0 - 30 * k - 14, a)
    c.setFont("Helvetica", 11.5)
    c.drawString(170, y0 - 30 * k - 14, b)
card(630, 150, 290, 260, MINT, "Why this dataset",
     "It is the only public DAS dataset with labelled vessel distances. The authors also "
     "published a baseline (XGBoost) and an evaluation protocol, so we can compare our "
     "numbers to theirs fairly.<br/><br/>Labels come from AIS, so vessels without AIS are "
     "invisible to the labels.")

# 6. one sample
new_page("What one sample looks like", "2  The dataset", PINK)
image("sample_window.png", 40, 110, w=560)
bullets([
    "Each window is a small image: <b>247 channels</b> (x axis, position along the cable) by "
    "<b>100 frequency bands</b> (y axis).",
    "Colour is the log of the vibration energy in that band at that position.",
    "Horizontal stripes are the ship's tonal noise: engine and propeller harmonics heard "
    "along the whole segment.",
    "This picture-like structure is exactly why a <b>CNN</b> (an image model) makes sense later.",
], 630, 405, 300, SMALL)

# 7. labels
new_page("What we predict", "2  The dataset", PINK)
image("labels.png", 40, 120, w=540)
bullets([
    "Distance to the closest vessel ranges from 13 m to 9.1 km; the median is 1.7 km.",
    "Following the paper, a window is <b>near</b> when that distance is at most <b>1,000 m</b>. "
    "That gives 32% near and 68% far.",
    "The classes are imbalanced, so accuracy alone can mislead. We also report F1 per class, "
    "macro F1 (plain average) and weighted F1 (what the paper calls global F1).",
    "The noisy channels 59, 60 and 61 are already zeros in the release; we drop them.",
], 610, 405, 320, SMALL)

# 8. near minus far
new_page("Is there a signal? Near minus far", "2  The dataset", PINK)
image("near_minus_far.png", 40, 120, w=560)
bullets([
    "We averaged 2,000 near windows and 2,000 far windows (more than 3 km) and subtracted.",
    "Everything is warmer: a close ship raises energy almost everywhere.",
    "The strongest differences are the <b>lowest bands</b> (propeller and engine) and a band "
    "near <b>50 to 70 Hz</b>.",
    "It is also uneven along the cable, so <b>where</b> on the cable it is loud carries "
    "information. That motivates the spatial features.",
], 630, 405, 300, SMALL)

# 9. evaluation
new_page("Evaluation: hold out whole days", "3  Evaluation", MINT)
image("cv_scheme.png", 40, 130, w=520)
bullets([
    "Windows 10 s apart are nearly identical. A random train/test split would put almost the "
    "same window in both, and the score would be far too optimistic. This is <b>leakage</b>.",
    "So each fold tests on one <b>whole day</b> and trains on the other nine. Ten folds, one per "
    "day, scores averaged. This is the protocol the dataset authors recommend.",
    "Context features only look <b>backwards in time</b>, never across a recording gap, so they "
    "would also work live.",
], 590, 405, 340, SMALL)

# 10. metrics
new_page("Metrics in plain words", "3  Evaluation", MINT)
mets = [("Precision (near)", "Of the windows we flagged as near, how many really were.", BLUE),
        ("Recall (near)", "Of the windows that really were near, how many we caught.", PINK),
        ("F1", "Harmonic mean of precision and recall, one number per class.", MINT),
        ("Macro F1", "Plain average of the near F1 and the far F1. Treats both classes equally.", PEACH),
        ("Weighted F1", "Average weighted by class size. The paper's 'global F1'.", LAV),
        ("AUC", "Chance a random near window scores higher than a random far one.", LEMON),
        ("MAE", "Mean absolute error of the distance, in metres. Lower is better.", BLUE),
        ("Paper protocol MAE", "MAE of a model trained and tested only on windows within 1 km.", PINK)]
for k, (t, s, col) in enumerate(mets):
    card(40 + (k % 2) * 450, 360 - (k // 2) * 78, 430, 66, col, t, s)

# 11. pipeline
new_page("The pipeline at a glance", "4  Pipeline and code", PEACH)
steps = [("HDF5", "14.1 GB float64", BLUE), ("Cache", "3.7 GB float16", PINK),
         ("Features", "565 per window", MINT), ("XGBoost", "detect + distance", PEACH),
         ("Cascade", "detect, then measure", LAV), ("Demo", "FastAPI replay", LEMON)]
for k, (a, b, col) in enumerate(steps):
    box(40 + k * 150, 330, 128, 70, a, col, b)
    if k < len(steps) - 1:
        arrow(40 + k * 150 + 130, 365, 40 + (k + 1) * 150 - 4, 365)
box(190, 215, 128, 70, "CNN", LAV, "reads the cache directly")
arrow(254, 330, 254, 287)
arrow(318, 250, 700, 250)
arrow(700, 250, 700, 326)
scripts = [("01_build_cache.py", "stream HDF5 into a small cache", "5 min"),
           ("02_eda.py", "label stats and spectra plots", "1 min"),
           ("03_train_gbm.py", "XGBoost, 10 folds, 3 models per fold", "4 to 15 min on GPU"),
           ("04_train_cnn.py", "CNN, 10 folds", "about 95 min"),
           ("05_report.py", "comparison table, recall by ship type", "1 min"),
           ("06_cascade.py", "end-to-end evaluation", "seconds")]
for k, (a, b, t) in enumerate(scripts):
    y = 170 - k * 22
    c.setFont("Courier-Bold", 10.5)
    c.setFillColor(INK)
    c.drawString(60, y, a)
    c.setFont("Helvetica", 10.5)
    c.drawString(230, y, b)
    c.setFillColor(MUTED)
    c.drawString(560, y, t)

# 12. code: cache
new_page("Step 1: shrink 15 GB into 3.7 GB", "4  Pipeline and code", PEACH)
snip = drop_docstring(source("src/shipdas/data.py", "build_cache"))[:12]
code(snip, 40, 405, 560, size=9.0)
explain([
    ("1", "A function with a default path; chunk = how many windows we read at a time."),
    ("3", "Open the HDF5 file read-only. Nothing is loaded yet, h5py reads lazily."),
    ("6", "One global offset (a median) so the stored numbers are small. Small numbers keep "
          "float16 precise."),
    ("7", "open_memmap creates a .npy file on disk that we can fill like an array."),
    ("9", "Loop over the file 512 windows at a time, about 100 MB each. RAM never fills."),
    ("10", "Drop the noisy channels, take log10 of the energy, subtract the offset."),
    ("8, 11", "Track the smallest and largest value; the uint8 GPU store needs that range."),
    ("12", "Write the block into the cache. float64 to float16 is 4x smaller."),
], 620, 405, 310)

# 13. code: folds
new_page("Step 2: one fold per day", "4  Pipeline and code", PEACH)
snip = drop_docstring(source("src/shipdas/data.py", "day_folds"))
code(snip, 40, 405, 520, size=11)
explain([
    ("1", "days is an array with the date of every window, for example '2023-06-16'."),
    ("2", "For each distinct day, in date order..."),
    ("3", "...a boolean mask: True where the window belongs to the test day."),
    ("4", "yield hands back the day name, the training indices (every other day) and the test "
          "indices. Used by every training script, so they all share the same 10 folds."),
], 600, 405, 330)
para("<b>Why a generator?</b> The caller writes <font face='Courier'>for day, tr, te in "
     "day_folds(ds.days):</font> and gets one fold per loop, without building all ten up front.",
     40, 300, 520, SMALL)

# 14. code: features
new_page("Step 3: features for XGBoost", "4  Pipeline and code", PEACH)
snip = source("src/shipdas/features.py", "block_features")
code(snip, 40, 405, 600, size=9.6)
explain([
    ("2", "Work in float32; the cache is float16."),
    ("3", "Loudest channel for each band. Used twice: as a feature and for a stable average."),
    ("4", "The paper's feature: average the linear energy over channels, then log. Doing it "
          "as top + log(mean(10^(L - top))) avoids overflow."),
    ("5-8", "Cut the cable into 13 segments and the bands into 5 groups, and average each "
            "block: 65 numbers describing where along the cable it is loud."),
    ("9", "Return mean, spread (std), max and segments: 100+100+100+65 = 365 features."),
], 660, 405, 270)

# 15. why xgboost
new_page("Why XGBoost first", "5  Models", LAV)
bullets([
    "<b>It is the paper's baseline.</b> Replicating it first proves our data handling is right "
    "before trying anything new. We got 89.35% accuracy against their 89.58%.",
    "Gradient-boosted trees are the strongest standard model for <b>tabular features</b>: fast, "
    "little tuning, robust to features on different scales.",
    "It trains on the GPU: 10 folds in about 4 minutes for 100 features.",
    "Trees can use any feature, so it tells us cheaply whether the new spatial features help "
    "(they did: +1.4 points macro F1).",
    "Hyper-parameters copied from the paper: learning rate 0.05, depth 10, 500 trees.",
], 40, 410, 520, SMALL)
snip = source("scripts/03_train_gbm.py", "run")[4:10]
code([ln[4:] for ln in snip], 590, 410, 340, size=8.4)
para("Three models per fold: <b>clf</b> for near/far, <b>reg</b> for distance on all windows, "
     "<b>reg_n</b> for distance trained only on windows within 1 km (the paper's protocol, and "
     "the distance model in our final cascade).", 590, 300, 340, NOTE)

# 16. why cnn
new_page("Why a CNN next", "5  Models", LAV)
bullets([
    "Hand-made features throw information away. Each window is really a 247 x 100 "
    "<b>image</b>, and convolutional networks are built to read images.",
    "Convolutions find local patterns (a harmonic stripe, a loud stretch of cable) wherever "
    "they appear. Max pooling lets the model react to a ship at any position.",
    "Stacking the <b>5 previous windows</b> (50 s) as extra input planes gives the model "
    "motion: a ship approaching gets louder over time.",
    "One network, two outputs (multitask): a near/far logit and a distance. Sharing the "
    "features helps both.",
], 40, 410, 470, SMALL)
xs = [530, 610, 690, 770, 850]
labels = [("input", "6 x 247 x 100"), ("block 1", "32 ch"), ("block 2", "64 ch"),
          ("block 3+4", "128 ch"), ("head", "2 outputs")]
for k, (a, b) in enumerate(labels):
    hh = 150 - k * 22
    box(xs[k], 300 - hh / 2, 68, hh, a, [BLUE, PINK, MINT, PEACH, LAV][k])
    c.setFont("Helvetica", 9)
    c.setFillColor(MUTED)
    c.drawCentredString(xs[k] + 34, 300 - hh / 2 - 14, b)
    if k < 4:
        arrow(xs[k] + 68, 300, xs[k + 1] - 2, 300)
para("Each block: conv 3x3, batch norm, ReLU, twice, then 2x2 max pool. The head pools the "
     "map with both average and max, then two linear layers.", 530, 170, 400, NOTE)

# 17. code: context
new_page("Code: looking back 50 seconds, safely", "4  Pipeline and code", PEACH)
snip = drop_docstring(source("src/shipdas/cnn.py", "context_index"))
code(snip, 40, 405, 560, size=10.5)
explain([
    ("1", "contiguous_prev[t] is True when window t starts exactly 10 s after t-1."),
    ("3", "A table with k+1 columns: indices of the past windows, then the current one."),
    ("4", "The last column is the current window itself."),
    ("5", "Fill the columns from right to left."),
    ("6-7", "Step one window back only if the recording was continuous; otherwise repeat the "
            "same window. So we never jump across a gap and never look into the future."),
], 630, 405, 300)

# 18. code: model
new_page("Code: the CNN", "4  Pipeline and code", PEACH)
snip = source("src/shipdas/cnn.py", "DasNet")
code(snip, 40, 405, 560, size=9.0)
explain([
    ("2", "in_planes = 1 + context windows; width sets the number of filters (32)."),
    ("5-6", "Four Blocks double the filters while each halves the image size."),
    ("7-8", "The head: 256 pooled numbers to 128 hidden units, dropout 0.3, then 2 outputs."),
    ("12", "Global average and global max pooling, side by side. Max catches a ship that is "
           "loud on only part of the cable."),
    ("14", "Output 0 is the near/far logit, output 1 the standardised distance."),
], 630, 405, 300)

# 19. code: training
new_page("Code: one training step", "4  Pipeline and code", PEACH)
snip = source("scripts/04_train_cnn.py", "train_fold")
start = next(i for i, ln in enumerate(snip) if "for b in batches(tr" in ln)
code([ln[8:] for ln in snip[start:start + 13]], 40, 405, 560, size=9.6)
explain([
    ("1", "Shuffle the training windows and cut them into batches of 128."),
    ("3", "Mixed precision (float16) makes the 6 GB laptop GPU faster and fit more."),
    ("4", "inputs() gathers the 6 windows, normalises each band and adds a small random gain."),
    ("5", "Binary cross-entropy for the near/far output."),
    ("6-7", "Distance loss, masked by w so it can be limited to near windows if we want."),
    ("8-12", "Standard step: clear gradients, backpropagate (scaled for float16), update, and "
             "move the one-cycle learning-rate schedule."),
], 630, 405, 300)
image("loss.png", 40, 70, h=150)
para("8 epochs per fold, AdamW, peak learning rate 0.002. About 10 minutes per fold.", 330, 150, 280, NOTE)

# 20. cascade
new_page("Why a cascade of both", "5  Models", LAV)
box(60, 270, 170, 80, "10 s window", BLUE, "plus 50 s of context")
box(310, 270, 170, 80, "CNN", LAV, "P(vessel within 1 km)")
box(560, 320, 170, 70, "XGBoost reg_n", MINT, "distance in metres")
box(560, 215, 170, 70, "no alert", PEACH, "far, nothing to measure")
arrow(230, 310, 306, 310)
arrow(480, 320, 556, 352)
arrow(480, 300, 556, 252)
c.setFont("Helvetica", 10)
c.setFillColor(MUTED)
c.drawString(498, 352, "P >= 0.5")
c.drawString(498, 252, "P < 0.5")
bullets([
    "The CNN was the best <b>detector</b> (90.7% macro F1) but its distance head under the "
    "paper protocol was worse than XGBoost (152 to 206 m against 129 m).",
    "XGBoost on spatial features was the best <b>distance</b> model within 1 km.",
    "So we chain them, the way a deployed system would work: detect first, then measure.",
    "This is also a fairer test than the paper's protocol, which assumes we already know "
    "which windows are within 1 km.",
], 40, 190, 880, SMALL, gap=4)

# 21. headline
new_page("Results: better than the paper on both tasks", "6  Results", LEMON)
image("headline.png", 40, 160, w=600)
bullets([
    "<b>Detection:</b> weighted F1 91.94% against the paper's 89.43%; macro F1 90.72%, AUC 0.963.",
    "<b>Distance, fair comparison:</b> 128.5 m against 171 m, same protocol.",
    "<b>End to end:</b> 117 m on vessels the CNN detected (median 85 m). Detected vessels are "
    "somewhat easier, so compare the 128.5 m with the paper.",
], 660, 405, 270, SMALL)
para("All numbers: mean over 10 held-out days.", 40, 150, 600, TINY)

# 22. ablation
new_page("What each step bought", "6  Results", LEMON)
image("ablation.png", 40, 140, w=600)
bullets([
    "Spatial features: +1.4 points macro F1 and 62 m less error. Where on the cable it is "
    "loud matters.",
    "Hand-made context features did <b>not</b> help XGBoost.",
    "Learned context did help the CNN: +0.9 points and 38 m less error on the screening days.",
    "Bars: distance error of models trained on all distances, measured within 1 km.",
], 660, 405, 270, SMALL)

# 23. timeline + scatter
new_page("What the predictions look like", "6  Results", LEMON)
image("timeline.png", 40, 130, w=560)
image("scatter.png", 620, 140, w=300)
para("Left: two hours of a held-out day. The probability jumps up when a vessel enters the 1 km "
     "band. Right: predicted against true distance for every detected vessel.", 40, 120, 880, NOTE)

# 24. failures
new_page("Where it fails", "6  Results", LEMON)
image("ship_types.png", 40, 110, w=480)
image("per_day.png", 540, 230, w=390)
para("Fishing vessels (36%) and dredgers (71%) are missed most, and they are real cable threats. "
     "The roughest day was the hardest, but 10 days are too few to prove a weather effect.",
     540, 210, 390, NOTE)

# 25. run it
new_page("Run it yourself", "7  Run it yourself", BLUE)
cmds = ["uv venv .venv --python 3.12",
        "uv pip install --python .venv/Scripts/python.exe torch \\",
        "    --index-url https://download.pytorch.org/whl/cu126",
        "uv pip install --python .venv/Scripts/python.exe -e \".[dev]\"",
        "",
        "# put the Zenodo files in data/raw/, then:",
        "python scripts/01_build_cache.py",
        "python scripts/02_eda.py",
        "python scripts/03_train_gbm.py --sets paper spatial --tag paperhp",
        "python scripts/03_train_gbm.py --sets spatial --tag cascade",
        "python scripts/04_train_cnn.py --epochs 8 --context 5 --gain-aug 0.1 \\",
        "    --save-model --tag cnn_c5",
        "python scripts/06_cascade.py",
        "python scripts/05_report.py",
        "python -m shipdas.serve --day 2023-06-20    # open http://127.0.0.1:8765",
        "python -m pytest -q"]
code(cmds, 40, 405, 600, size=9)
card(670, 230, 260, 210, MINT, "Before you start",
     "Download from Zenodo record 15611778: the 14.1 GB .h5 file and misc.zip. Check the MD5: "
     "407ad62bb8eb02f472cda329d0f31750.<br/><br/>Python 3.12, an NVIDIA GPU with CUDA 12, "
     "about 25 GB free disk.")

# 26. engineering lessons
new_page("Engineering notes from a 16 GB laptop", "7  Run it yourself", BLUE)
notes = [("RAM, not GPU, was the limit", "The float64 file is bigger than RAM, so the cache is "
          "built in chunks. Two runs at once crashed with out-of-memory.", PINK),
         ("Data on the GPU as uint8", "The --gpu-data flag stores the dataset on the GPU in "
          "8 bits (1.85 GB). Same accuracy, frees 3.7 GB of RAM.", MINT),
         ("Parallel download", "Zenodo gave 2 MB/s per connection. Eight range requests in "
          "parallel reached about 20 MB/s; the MD5 matched.", BLUE),
         ("Leave GPU headroom", "One CNN run crashed with a cuDNN error when a browser took GPU "
          "memory; the rerun was fine.", PEACH)]
for k, (t, s, col) in enumerate(notes):
    card(40 + (k % 2) * 450, 300 - (k // 2) * 140, 430, 120, col, t, s)

# 27. limits
new_page("Limitations and next steps", "7  Run it yourself", BLUE)
bullets([
    "<b>One site:</b> 2.5 km of one cable, one sea, ten summer days. Another cable or season "
    "will need re-validation.",
    "<b>AIS labels:</b> vessels without AIS are missing, and positions are interpolated.",
    "<b>Mild selection bias:</b> CNN settings were picked on two of the ten days, which are also "
    "in the average. The gain held on the other eight.",
    "<b>Next:</b> class weighting or more data for small vessels; a single model that is good at "
    "both tasks; test on the Pacific OOI cable data; stream from a live interrogator.",
], 40, 410, 880)

# 28. references
new_page("References", "Sources", LAV)
bullets([
    "Ramirez-Torres et al., Marlinks-NS DAS dataset, Zenodo, doi:10.5281/zenodo.15611778 (CC BY 4.0).",
    "Ramirez-Torres et al., Vessel detection and localization using DAS in submarine optical "
    "fiber cables, IEEE JSTARS 2026, doi:10.1109/JSTARS.2026.3716768.",
    "Ramirez-Torres et al., A DAS dataset for vessel detection and localization in submarine "
    "cable protection, arXiv:2607.28306.",
    "Companion code: github.com/UAH-PSI/das-vessel-detection (GPL-3.0).",
    "XGBoost: Chen and Guestrin, KDD 2016. PyTorch: Paszke et al., NeurIPS 2019.",
], 40, 410, 880, SMALL)

c.save()
print(f"wrote {OUT} ({page_no} pages)")
