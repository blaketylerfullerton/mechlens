"""Step 2 of 5: sparsify with an SAE.

Step 1 got the activations. This is what an SAE does to one of them — and,
because that is the step where a wrong answer still looks plausible, the two
diagnostics that catch it.

Every number on screen is read from `data/step2.json`, which `extract.py`
writes out of a real saved trace. Nothing here is drawn to a shape that was
chosen to look convincing.

    ./venv/bin/manim -pql docs/manim/step2_sparse_features.py SparseFeatures
    ./venv/bin/manim -pqh docs/manim/step2_sparse_features.py SparseFeatures
"""

import json
import sys
from pathlib import Path

from manim import (DOWN, LEFT, RIGHT, UP, Create, FadeIn, FadeOut, GrowFromEdge,
                   LaggedStart, LaggedStartMap, Scene, Transform, VGroup, Write)

sys.path.insert(0, str(Path(__file__).parent))
from theme import (BG, BORDER, BORDER_STRONG, COMMENT, CONST, FG, FN, NUM, RULE,
                   SECONDARY, STR, TERTIARY, caption, hairline, human, machine,
                   panel, sparse_track, value_track)

DATA = json.loads((Path(__file__).parent / "data" / "step2.json").read_text())
SRC, CELL, RESID = DATA["source"], DATA["cell"], DATA["residual"]
FEATURES, DIAG = DATA["features"], DATA["diagnostics"]


class SparseFeatures(Scene):
    def construct(self):
        self.camera.background_color = BG
        self.chapter()
        self.dense_vector()
        self.the_dictionary()
        self.the_gate()
        self.what_fired()
        self.what_it_cost()

    # -- a numbered section, Swiss habit: say where you are ----------------
    def chapter(self):
        number = machine("02 / 05", size=20, color=FN)
        title = human("sparsify with an SAE", size=46, weight="BOLD")
        sub = caption("2,304 entangled numbers become a handful of separable ones", size=22)
        head = VGroup(number, title, sub).arrange(DOWN, buff=0.34)
        self.play(FadeIn(number), Write(title), run_time=1.4)
        self.play(FadeIn(sub, shift=UP * 0.2))
        self.wait(1.2)
        self.play(FadeOut(head, shift=UP * 0.3))

    def rule_header(self, text):
        label = machine(text, size=19, color=TERTIARY).to_edge(UP, buff=0.55).to_edge(LEFT, buff=0.9)
        line = hairline(12.4, color=BORDER).next_to(label, DOWN, buff=0.22).align_to(label, LEFT)
        header = VGroup(label, line)
        self.play(FadeIn(label), Create(line), run_time=0.7)
        return header

    # -- 1. the thing we start from ---------------------------------------
    def dense_vector(self):
        header = self.rule_header(f"{SRC['hook']}  ·  layer {CELL['layer']}  ·  token {CELL['token']!r}")

        head = RESID["head"]
        track = value_track(head, width=11.2, height=1.5)
        track.move_to(UP * 0.55)

        lead = human("one residual vector", size=30, weight="BOLD")
        lead.next_to(track, UP, buff=0.75).align_to(track, LEFT)

        shown = caption(f"first {len(head)} of {CELL['d_model']:,} values, drawn to scale", size=18)
        shown.next_to(track, DOWN, buff=0.4).align_to(track, LEFT)
        span = machine(f"norm {RESID['norm']}   min {RESID['min']}   max {RESID['max']}",
                       size=18, color=COMMENT)
        span.next_to(shown, DOWN, buff=0.18).align_to(track, LEFT)

        point = human("no single entry means anything on its own —\nevery concept is smeared across all of them",
                      size=24, color=SECONDARY)
        point.next_to(span, DOWN, buff=0.6).align_to(track, LEFT)

        # Compose the whole section, then settle it under the header once, so
        # nothing drifts as it is revealed.
        VGroup(lead, track, shown, span, point).move_to(DOWN * 0.45)

        self.play(FadeIn(lead, shift=RIGHT * 0.2))
        self.play(LaggedStartMap(GrowFromEdge, track, edge=DOWN, lag_ratio=0.02, run_time=1.6))
        self.play(FadeIn(shown), FadeIn(span))
        self.wait(0.7)
        self.play(Write(point), run_time=1.6)
        self.wait(1.6)
        self.play(FadeOut(VGroup(header, lead, shown, span, point)), FadeOut(track))

    # -- 2. what we run it through ----------------------------------------
    def the_dictionary(self):
        header = self.rule_header("the dictionary")

        box = panel(7.6, 2.5)
        name = machine(SRC["release"], size=21, color=FN)
        detail = VGroup(
            machine(f"width      {SRC['width']}   ({CELL['d_sae']:,} features)", size=19, color=SECONDARY),
            machine(f"site       blocks.{CELL['layer']}.{SRC['hook']}", size=19, color=SECONDARY),
            machine(f"layers     {SRC['n_layers']}, one SAE each", size=19, color=SECONDARY),
        ).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        divider = hairline(6.6, color=BORDER)
        body = VGroup(name, divider, detail).arrange(DOWN, buff=0.34)
        body.move_to(box.get_center())
        card = VGroup(box, body).move_to(UP * 0.7)

        self.play(FadeIn(card, shift=UP * 0.2), run_time=0.9)

        pretrained = human("pretrained by Google, not by us — this is the half the\ntraining page replaces with your own",
                           size=22, color=SECONDARY)
        pretrained.next_to(card, DOWN, buff=0.7)
        self.play(FadeIn(pretrained))
        self.wait(1.8)

        job = human("it is an autoencoder: widen, then reconstruct", size=28, weight="BOLD")
        job.move_to(card.get_center())
        self.play(FadeOut(body), FadeOut(pretrained), Transform(box, panel(11.0, 1.5).move_to(card.get_center())),
                  FadeIn(job))

        flow = VGroup(
            machine(f"x  [{CELL['d_model']:,}]", size=20, color=SECONDARY),
            machine("encode →", size=20, color=TERTIARY),
            machine(f"a  [{CELL['d_sae']:,}]", size=20, color=FN),
            machine("decode →", size=20, color=TERTIARY),
            machine(f"x̂  [{CELL['d_model']:,}]", size=20, color=SECONDARY),
        ).arrange(RIGHT, buff=0.55)
        flow.next_to(box, DOWN, buff=0.9)
        self.play(LaggedStart(*[FadeIn(m, shift=RIGHT * 0.15) for m in flow], lag_ratio=0.25, run_time=1.8))
        self.wait(0.6)

        why = caption("widening only helps if almost all of it stays zero", size=21, color=SECONDARY)
        why.next_to(flow, DOWN, buff=0.6)
        self.play(Write(why), run_time=1.2)
        self.wait(1.6)
        self.play(FadeOut(VGroup(header, box, job, flow, why)))

    # -- 3. the sparsity, at true positions --------------------------------
    def the_gate(self):
        header = self.rule_header(f"encode  ·  JumpReLU gate  ·  token {CELL['token']!r}")

        indices = [f["index"] for f in FEATURES]
        acts = [f["activation"] for f in FEATURES]
        track = sparse_track(indices, acts, CELL["d_sae"], width=12.0, height=1.7)
        track.move_to(UP * 0.35)

        ends = VGroup(machine("0", size=17, color=COMMENT),
                      machine(f"{CELL['d_sae']:,}", size=17, color=COMMENT))
        ends[0].next_to(track, DOWN, buff=0.28).align_to(track, LEFT)
        ends[1].next_to(track, DOWN, buff=0.28).align_to(track, RIGHT)

        lead = human("almost all of it is zero", size=30, weight="BOLD")
        lead.next_to(track, UP, buff=0.85).align_to(track, LEFT)

        count = machine(f"l0 = {CELL['l0']}", size=30, color=FN)
        of = human(f"of {CELL['d_sae']:,} — {CELL['l0'] / CELL['d_sae'] * 100:.1f}% fired",
                   size=24, color=SECONDARY)
        tally = VGroup(count, of).arrange(RIGHT, buff=0.45)
        tally.next_to(ends, DOWN, buff=0.75).align_to(track, LEFT)

        note = caption("each mark sits at its real feature index, so the gaps are the measurement", size=19)
        note.next_to(tally, DOWN, buff=0.45).align_to(track, LEFT)

        VGroup(lead, track, ends, tally, note).move_to(DOWN * 0.45)

        self.play(FadeIn(lead, shift=RIGHT * 0.2))
        self.play(Create(track[0]), run_time=0.8)
        self.play(LaggedStartMap(GrowFromEdge, track[1], edge=DOWN, lag_ratio=0.08, run_time=1.6))
        self.play(FadeIn(ends))
        self.wait(0.5)
        self.play(FadeIn(tally, shift=UP * 0.2))
        self.wait(0.6)
        self.play(FadeIn(note))
        self.wait(2.0)
        self.play(FadeOut(VGroup(header, lead, ends, tally, note)), FadeOut(track))

    # -- 4. the payoff: the features are readable --------------------------
    def what_fired(self):
        header = self.rule_header(f"layer {CELL['layer']}  ·  token {CELL['token']!r}  ·  strongest of {CELL['l0']}")

        # Fixed columns rather than arranged runs, and the column edges are
        # measured off the widest cell rather than guessed: a numeric column
        # that does not align is the figure admitting it was drawn, not read.
        shown = FEATURES[:6]
        top = max(f["activation"] for f in shown)
        gutter = 0.34
        index_cells = [machine(f"#{f['index']}", size=21, color=TERTIARY) for f in shown]
        value_cells = [machine(f"{f['activation']:.2f}", size=21,
                               color=FN if i == 0 else SECONDARY) for i, f in enumerate(shown)]
        index_w = max(c.width for c in index_cells)
        value_w = max(c.width for c in value_cells)
        bar_span = 2.6

        x_index = -6.5
        x_value = x_index + index_w + gutter + value_w   # right edge: numbers align right
        x_bar = x_value + gutter
        x_text = x_bar + bar_span + gutter

        rows = VGroup()
        for i, f in enumerate(shown):
            lit = i == 0
            y = DOWN * i * 0.66
            index_cells[i].move_to(y + RIGHT * x_index, aligned_edge=LEFT)
            value_cells[i].move_to(y + RIGHT * x_value, aligned_edge=RIGHT)
            bar = hairline(max(f["activation"] / top, 0.04) * bar_span,
                           color=FN if lit else RULE)
            bar.move_to(y + RIGHT * x_bar, aligned_edge=LEFT)
            text = human(f["label"] or "unlabelled", size=21, color=FG if lit else SECONDARY)
            text.move_to(y + RIGHT * x_text, aligned_edge=LEFT)
            rows.add(VGroup(index_cells[i], value_cells[i], bar, text))
        rows.move_to(DOWN * 0.15)

        lead = human("and now they are readable", size=30, weight="BOLD")
        lead.next_to(rows, UP, buff=0.75).align_to(rows, LEFT)
        prompt = machine(f"prompt: {SRC['prompt']!r}", size=18, color=COMMENT)
        prompt.next_to(rows, DOWN, buff=0.65).align_to(rows, LEFT)
        got = human("the model is holding \"bridge\" and \"the Bay Area\" apart,\nin the same vector, at the same instant",
                    size=23, color=SECONDARY)
        got.next_to(prompt, DOWN, buff=0.4).align_to(rows, LEFT)

        VGroup(lead, rows, prompt, got).move_to(DOWN * 0.45)

        self.play(FadeIn(lead, shift=RIGHT * 0.2))
        self.play(LaggedStartMap(FadeIn, rows, shift=RIGHT * 0.25, lag_ratio=0.22, run_time=2.4))
        self.wait(1.0)
        self.play(FadeIn(prompt))
        self.wait(1.6)
        self.play(Write(got), run_time=1.8)
        self.wait(2.0)
        self.play(FadeOut(VGroup(header, lead, rows, prompt, got)))

    # -- 5. the caveat, which never hides ----------------------------------
    def what_it_cost(self):
        header = self.rule_header("what that cost, and how we know")

        entries = [
            (f"{SRC['top_k']} of {CELL['l0']}", "features kept per cell — the rest fired and were not saved", CONST),
            (f"{DIAG['explained_variance_mean']:.3f}", "explained variance, mean over 26 layers "
                                                       f"(worst layer {DIAG['explained_variance_min']:.3f})", STR),
            (f"{DIAG['l0_mean']:.1f}", "features firing per token, averaged — 16,384 would mean the gate was skipped", NUM),
        ]
        rows = VGroup()
        for value, text, colour in entries:
            v = machine(value, size=30, color=colour)
            t = human(text, size=20, color=SECONDARY)
            row = VGroup(v, t).arrange(RIGHT, buff=0.55, aligned_edge=DOWN)
            rows.add(row)
        rows.arrange(DOWN, buff=0.6, aligned_edge=LEFT)

        line = hairline(11.0, color=BORDER).next_to(rows, DOWN, buff=0.7).align_to(rows, LEFT)
        closing = human("a sparse code that reconstructs badly is a confident\npicture of nothing — so the number is on screen, not in a log",
                        size=23, color=SECONDARY)
        closing.next_to(line, DOWN, buff=0.45).align_to(rows, LEFT)

        VGroup(rows, line, closing).move_to(DOWN * 0.45)

        self.play(LaggedStartMap(FadeIn, rows, shift=RIGHT * 0.2, lag_ratio=0.3, run_time=2.0))
        self.wait(1.4)
        self.play(Create(line), run_time=0.6)
        self.play(Write(closing), run_time=2.0)
        self.wait(2.4)

        self.play(FadeOut(VGroup(header, rows, line, closing)))
        end = human("next — step 3: check it didn't lie to you", size=30, weight="BOLD")
        tag = machine("03 / 05", size=20, color=FN).next_to(end, UP, buff=0.4)
        self.play(FadeIn(tag), Write(end), run_time=1.5)
        self.wait(2.0)
        self.play(FadeOut(VGroup(tag, end)))
