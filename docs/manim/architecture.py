"""
A conceptual explainer: how sparse autoencoders (SAEs) turn a model's tangled
internal activations into individually meaningful, human-labeled features.

This is about the IDEA, not the codebase: dense activation -> sparse features
-> trained by reconstruction -> labeled by auto-interp on activating examples.

Render with:
    ./venv/bin/manim -pql docs/manim/architecture.py SAEConcept   # quick draft
    ./venv/bin/manim -pqh docs/manim/architecture.py SAEConcept   # high quality
"""

import random

from manim import *

BG = "#0b0d12"
BOX = "#1f2733"
BORDER = "#7dd3fc"
ACCENT = "#facc15"
DIM = "#475569"
TEXT = WHITE
MUTED = "#9ca3af"

random.seed(7)


def vector_bars(n, height_fn, color=DIM, active_color=ACCENT, active_set=None, width=0.28, gap=0.08):
    active_set = active_set or set()
    bars = VGroup()
    for i in range(n):
        h = height_fn(i)
        c = active_color if i in active_set else color
        bar = Rectangle(width=width, height=max(h, 0.04), fill_color=c, fill_opacity=1, stroke_width=0)
        bar.move_to(RIGHT * i * (width + gap), aligned_edge=DOWN)
        bars.add(bar)
    bars.move_to(ORIGIN, aligned_edge=DOWN)
    return bars


def labeled_box(text, sub=None, width=3.2, height=1.0, color=BOX):
    box = RoundedRectangle(corner_radius=0.12, width=width, height=height,
                            fill_color=color, fill_opacity=1, stroke_color=BORDER, stroke_width=2)
    t = Text(text, font_size=24, color=TEXT, weight=BOLD)
    group = VGroup(box, t)
    if sub:
        t.shift(UP * 0.15)
        s = Text(sub, font_size=14, color=MUTED)
        s.next_to(t, DOWN, buff=0.1)
        group.add(s)
    t.move_to(box.get_center() if not sub else t.get_center())
    return group


class SAEConcept(Scene):
    def construct(self):
        self.camera.background_color = BG

        # ---------- Part 1: a dense, tangled activation ----------
        title = Text("what's inside one number in the network?", font_size=36, color=TEXT, weight=BOLD)
        title.to_edge(UP, buff=0.5)
        self.play(Write(title))

        dense = vector_bars(14, lambda i: 0.4 + 1.6 * random.random())
        dense.move_to(ORIGIN + DOWN * 0.3)
        dense_label = Text("one activation vector, mid-layer", font_size=20, color=MUTED)
        dense_label.next_to(dense, DOWN, buff=0.4)

        self.play(LaggedStartMap(GrowFromEdge, dense, edge=DOWN, lag_ratio=0.06))
        self.play(FadeIn(dense_label))
        self.wait(0.4)

        tangle_note = Text(
            "every number here blends many ideas at once — not readable on its own",
            font_size=22, color=MUTED,
        )
        tangle_note.next_to(dense_label, DOWN, buff=0.35)
        self.play(Write(tangle_note))
        self.wait(1.2)

        self.play(FadeOut(tangle_note))

        # ---------- Part 2: the sparse autoencoder unmixes it ----------
        self.play(dense.animate.scale(0.7).to_edge(LEFT, buff=1.0), FadeOut(dense_label))

        encoder = labeled_box("encoder", width=2.2, height=1.4)
        encoder.next_to(dense, RIGHT, buff=0.8)
        arrow_in = Arrow(dense.get_right(), encoder.get_left(), color=MUTED, buff=0.1)

        self.play(GrowArrow(arrow_in), FadeIn(encoder, shift=RIGHT * 0.2))
        self.wait(0.3)

        n_sparse = 24
        active_idx = {3, 9, 16, 21}
        sparse = vector_bars(
            n_sparse,
            lambda i: (0.3 + 1.5 * random.random()) if i in active_idx else 0.05,
            active_set=active_idx, width=0.16, gap=0.05,
        )
        sparse.scale(0.7)
        sparse.next_to(encoder, RIGHT, buff=0.9)
        arrow_mid = Arrow(encoder.get_right(), sparse.get_left(), color=MUTED, buff=0.1)

        self.play(GrowArrow(arrow_mid), LaggedStartMap(GrowFromEdge, sparse, edge=DOWN, lag_ratio=0.03))
        sparse_label = Text("a much bigger, mostly-zero code", font_size=18, color=MUTED)
        sparse_label.next_to(sparse, DOWN, buff=0.35)
        few_label = Text("only a handful of \"features\" turn on", font_size=18, color=ACCENT)
        few_label.next_to(sparse_label, DOWN, buff=0.2)
        self.play(FadeIn(sparse_label), FadeIn(few_label))
        self.wait(1.0)

        decoder = labeled_box("decoder", width=2.2, height=1.4)
        decoder.next_to(sparse, RIGHT, buff=0.9)
        arrow_out = Arrow(sparse.get_right(), decoder.get_left(), color=MUTED, buff=0.1)
        self.play(GrowArrow(arrow_out), FadeIn(decoder, shift=RIGHT * 0.2))

        recon = dense.copy().scale(1.0)
        recon.next_to(decoder, RIGHT, buff=0.8)
        arrow_recon = Arrow(decoder.get_right(), recon.get_left(), color=MUTED, buff=0.1)
        self.play(GrowArrow(arrow_recon), FadeIn(recon, shift=RIGHT * 0.2))

        recon_label = Text("rebuilt from almost nothing — and it matches", font_size=18, color=MUTED)
        recon_label.next_to(recon, DOWN, buff=0.35)
        self.play(FadeIn(recon_label))
        self.wait(1.3)

        self.play(*[FadeOut(m) for m in [
            title, dense, encoder, sparse, decoder, recon, arrow_in, arrow_mid,
            arrow_out, arrow_recon, sparse_label, few_label, recon_label,
        ]])

        # ---------- Part 3: training the autoencoder ----------
        title2 = Text("how it learns that trick", font_size=36, color=TEXT, weight=BOLD)
        title2.to_edge(UP, buff=0.5)
        self.play(Write(title2))

        corpus = labeled_box("lots of text", sub="millions of tokens", width=2.6)
        corpus.move_to(LEFT * 5 + UP * 0.5)
        frozen = labeled_box("frozen model", sub="weights untouched", width=2.8)
        frozen.next_to(corpus, RIGHT, buff=0.9)
        sae_box = labeled_box("sparse autoencoder", sub="the part being trained", width=3.2)
        sae_box.next_to(frozen, RIGHT, buff=0.9)

        a1 = Arrow(corpus.get_right(), frozen.get_left(), color=MUTED, buff=0.1)
        a2 = Arrow(frozen.get_right(), sae_box.get_left(), color=MUTED, buff=0.1)

        self.play(FadeIn(corpus, shift=RIGHT * 0.2))
        self.play(GrowArrow(a1), FadeIn(frozen, shift=RIGHT * 0.2))
        self.play(GrowArrow(a2), FadeIn(sae_box, shift=RIGHT * 0.2))
        self.wait(0.4)

        loop = CurvedArrow(sae_box.get_bottom() + DOWN * 0.1, sae_box.get_bottom() + LEFT * 1.2 + DOWN * 0.1,
                            color=ACCENT, angle=-TAU / 3)
        loss_text = Text("adjust weights to: rebuild well + use as few features as possible",
                          font_size=20, color=ACCENT)
        loss_text.next_to(VGroup(corpus, frozen, sae_box), DOWN, buff=1.1)
        self.play(Create(loop))
        self.play(Write(loss_text))
        self.wait(1.4)

        self.play(*[FadeOut(m) for m in [title2, corpus, frozen, sae_box, a1, a2, loop, loss_text]])

        # ---------- Part 4: giving a feature meaning ----------
        title3 = Text("a feature is just a number, until you name it", font_size=32, color=TEXT, weight=BOLD)
        title3.to_edge(UP, buff=0.5)
        self.play(Write(title3))

        feature_tag = labeled_box("feature 4192", width=2.6, height=0.9, color="#1f2733")
        feature_tag.move_to(UP * 1.6)
        self.play(FadeIn(feature_tag, scale=0.8))
        self.wait(0.3)

        examples = VGroup(*[
            Text(s, font_size=20, color=MUTED, t2c={hl: ACCENT})
            for s, hl in [
                ("\"...she felt utterly  betrayed  by...\"", "betrayed"),
                ("\"...a  betrayal  no one saw coming...\"", "betrayal"),
                ("\"...he  turned on  his own crew...\"", "turned on"),
            ]
        ])
        examples.arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        examples.next_to(feature_tag, DOWN, buff=0.6)

        self.play(LaggedStartMap(FadeIn, examples, shift=UP * 0.2, lag_ratio=0.3))
        fires_note = Text("examples where this feature fires strongly", font_size=18, color=MUTED)
        fires_note.next_to(examples, DOWN, buff=0.3)
        self.play(FadeIn(fires_note))
        self.wait(1.0)

        interp_box = labeled_box("auto-interp", sub="reads the examples, proposes a name", width=3.4)
        interp_box.next_to(fires_note, DOWN, buff=0.6)
        arrow_to_interp = Arrow(examples.get_bottom(), interp_box.get_top(), color=MUTED, buff=0.15)
        self.play(GrowArrow(arrow_to_interp), FadeIn(interp_box, shift=DOWN * 0.2))
        self.wait(0.5)

        label_bubble = RoundedRectangle(corner_radius=0.2, width=4.4, height=0.9,
                                         fill_color="#14532d", fill_opacity=1,
                                         stroke_color="#4ade80", stroke_width=2)
        label_text = Text("\"betrayal / turning against someone\"", font_size=20, color="#bbf7d0")
        label_text.move_to(label_bubble.get_center())
        label_group = VGroup(label_bubble, label_text)
        label_group.next_to(interp_box, DOWN, buff=0.5)
        arrow_to_label = Arrow(interp_box.get_bottom(), label_group.get_top(), color=MUTED, buff=0.1)

        self.play(GrowArrow(arrow_to_label), FadeIn(label_group, shift=DOWN * 0.2))
        self.wait(1.5)

        checked_note = Text("verified against held-out examples before it's trusted",
                             font_size=16, color=MUTED)
        checked_note.next_to(label_group, DOWN, buff=0.3)
        self.play(FadeIn(checked_note))
        self.wait(1.5)

        final = Text("a number becomes something you can read", font_size=30, color=TEXT, weight=BOLD)
        self.play(*[FadeOut(m) for m in [
            title3, feature_tag, examples, fires_note, interp_box, label_group,
            arrow_to_interp, arrow_to_label, checked_note,
        ]])
        self.play(Write(final))
        self.wait(2)
