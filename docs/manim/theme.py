"""Shared visual direction for the mechlens step videos.

The tokens here mirror `frontend/src/index.css`, which is the source of truth.
If the two disagree, the CSS wins and this file is stale.

Direction, in the three rules that matter on a moving image:
  - dark is the design: #0A0B0D, text #E6E8EB, never pure black or pure white
  - the seven syntax colours are the entire accent system, `fn` leading
  - no gradient, no glow, no shadow, and nothing loops
"""

from manim import BLACK, DOWN, LEFT, RIGHT, UP, Line, Rectangle, Text, VGroup

# --- surfaces -----------------------------------------------------------
BG = "#0A0B0D"
SURFACE = "#101216"
ELEVATED = "#16181D"
BORDER = "#1E2127"
BORDER_STRONG = "#2A2E37"

# --- text ---------------------------------------------------------------
FG = "#E6E8EB"
SECONDARY = "#9BA1AC"
TERTIARY = "#7D848F"
COMMENT = "#7A828F"
RULE = "#6E7681"

# --- the syntax palette; `fn` leads, the rest stay in code and legends ---
FN = "#82AAFF"
NUM = "#F78C6C"
STR = "#C3E88D"
CONST = "#FFCB6B"
KW = "#C792EA"
ERR = "#F07178"

# Two radii and no more. At manim's scale 2px is a sharp corner.
RADIUS = 0.0


def _first_available(*candidates: str) -> str:
    """Prefer the real brand faces; fall back so a render never silently
    substitutes something arbitrary without us knowing which."""
    try:
        import manimpango
        installed = set(manimpango.list_fonts())
    except Exception:
        return candidates[-1]
    for name in candidates:
        if name in installed:
            return name
    return candidates[-1]


# Inter for anything a human wrote, JetBrains Mono for anything the machine
# produced. Neither is installed system-wide here, so these degrade on purpose.
SANS = _first_available("Inter", "Inter Variable", "DejaVu Sans")
MONO = _first_available("JetBrains Mono", "JetBrainsMono Nerd Font", "DejaVu Sans Mono")


def human(text, size=28, color=FG, weight="NORMAL"):
    """Prose, titles, captions — anything written by a person."""
    return Text(text, font=SANS, font_size=size, color=color, weight=weight)


def machine(text, size=22, color=SECONDARY):
    """Tokens, indices, hooks, numbers — anything the system emitted."""
    return Text(text, font=MONO, font_size=size, color=color)


def hairline(length, color=BORDER_STRONG, vertical=False):
    d = UP * length / 2 if vertical else RIGHT * length / 2
    return Line(-d, d, stroke_color=color, stroke_width=1)


def panel(width, height, fill=SURFACE, stroke=BORDER):
    """Depth is a layered surface and a 1px border. Never a shadow."""
    return Rectangle(width=width, height=height, fill_color=fill, fill_opacity=1,
                     stroke_color=stroke, stroke_width=1)


def caption(text, size=18, color=TERTIARY):
    return human(text, size=size, color=color)


def value_track(values, width, height=1.1, color=RULE, accent_at=None,
                accent=FN, floor=0.02):
    """A quantity drawn to scale as thin rules in a fixed track.

    Widths come from real values with a small floor, so a near-zero entry still
    prints a mark and nothing is invented to make the picture look better.
    """
    top = max(abs(v) for v in values) or 1.0
    n = len(values)
    pitch = width / n
    bars = VGroup()
    for i, v in enumerate(values):
        h = max(abs(v) / top, floor) * height
        bar = Rectangle(width=pitch * 0.62, height=h, stroke_width=0,
                        fill_color=accent if i == accent_at else color, fill_opacity=1)
        bar.move_to(LEFT * (width / 2) + RIGHT * (i + 0.5) * pitch, aligned_edge=DOWN)
        bars.add(bar)
    bars.move_to(LEFT * width / 2, aligned_edge=LEFT + DOWN)
    return bars


def sparse_track(indices, activations, domain, width, height=1.1,
                 color=FN, baseline=BORDER_STRONG):
    """The sparse code: a full-width track for the whole dictionary, with a
    mark at each firing feature's true index. Position is the real index, so
    the emptiness between marks is the measurement, not a styling choice."""
    top = max(activations) or 1.0
    track = Line(LEFT * width / 2, RIGHT * width / 2, stroke_color=baseline, stroke_width=1)
    marks = VGroup()
    for index, act in zip(indices, activations):
        x = -width / 2 + width * (index / domain)
        h = max(act / top, 0.06) * height
        mark = Line(RIGHT * x, RIGHT * x + UP * h, stroke_color=color, stroke_width=2.5)
        mark.shift(track.get_center())
        marks.add(mark)
    return VGroup(track, marks)
