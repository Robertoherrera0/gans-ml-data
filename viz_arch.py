from manim import *
import numpy as np

class ReflectometryTransformerScene(Scene):
    def construct(self):
        title = Text(
            "Bounded Hybrid FNO–Transformer for Multi-Contrast Neutron Reflectometry",
            font_size=30
        ).to_edge(UP)

        # Reflectivity mini-plot
        axes = Axes(
            x_range=[0, 0.25, 0.05],
            y_range=[-8, 0, 2],
            x_length=3.0,
            y_length=2.0,
            tips=False,
            axis_config={"font_size": 18},
        )

        axes_labels = axes.get_axis_labels(
            x_label=Text("q"),
            y_label=Text("log10 R(q)")
        )

        q = np.linspace(0.005, 0.25, 250)

        def curve_func(phase, offset):
            return -1.0 - 22*q + 0.25*np.sin(70*q + phase) + offset

        curves = VGroup()
        colors = [BLUE, GREEN, ORANGE]
        names = ["MIX", "D_2O", "H_2O"]

        for phase, offset, color in zip([0, 0.8, 1.6], [0, -0.5, -1.0], colors):
            graph = axes.plot_line_graph(
                x_values=q,
                y_values=curve_func(phase, offset),
                add_vertex_dots=False,
                line_color=color,
                stroke_width=3,
            )
            curves.add(graph)

        plot_group = VGroup(axes, axes_labels, curves).scale(0.75).to_edge(LEFT).shift(DOWN*0.2)

        input_label = Text("3 reflectivity curves\n+ q channel", font_size=22).next_to(plot_group, DOWN)

        # Architecture blocks
        input_proj = RoundedRectangle(width=2.0, height=1.0, corner_radius=0.15)
        input_proj_text = Text("Input projection\n1×1 Conv", font_size=22)
        input_proj_group = VGroup(input_proj, input_proj_text).next_to(plot_group, RIGHT, buff=0.8)

        hybrid = RoundedRectangle(width=3.0, height=1.8, corner_radius=0.15)
        hybrid_text = Text(
            "Hybrid block × 4\n\nSpectral Conv\nSelf-Attention\nFeed-Forward",
            font_size=21
        )
        hybrid_group = VGroup(hybrid, hybrid_text).next_to(input_proj_group, RIGHT, buff=0.8)

        pooling = RoundedRectangle(width=1.8, height=1.0, corner_radius=0.15)
        pooling_text = Text("Mean\npooling", font_size=22)
        pooling_group = VGroup(pooling, pooling_text).next_to(hybrid_group, RIGHT, buff=0.8)

        head = RoundedRectangle(width=2.0, height=1.0, corner_radius=0.15)
        head_text = Text("Bounded\nhead", font_size=22)
        head_group = VGroup(head, head_text).next_to(pooling_group, RIGHT, buff=0.8)

        output = Text("16 physical parameters\nthickness, SLD, roughness", font_size=22)
        output.next_to(head_group, RIGHT, buff=0.6)

        # Prior branch
        prior = RoundedRectangle(width=2.5, height=0.9, corner_radius=0.15)
        prior_text = Text("Prior bounds\n32 values", font_size=21)
        prior_group = VGroup(prior, prior_text).next_to(hybrid_group, DOWN, buff=0.9)

        prior_encoder = RoundedRectangle(width=2.0, height=0.9, corner_radius=0.15)
        prior_encoder_text = Text("Prior\nencoder", font_size=21)
        prior_encoder_group = VGroup(prior_encoder, prior_encoder_text).next_to(prior_group, RIGHT, buff=0.8)

        # Arrows
        arrows = VGroup(
            Arrow(plot_group.get_right(), input_proj_group.get_left(), buff=0.15),
            Arrow(input_proj_group.get_right(), hybrid_group.get_left(), buff=0.15),
            Arrow(hybrid_group.get_right(), pooling_group.get_left(), buff=0.15),
            Arrow(pooling_group.get_right(), head_group.get_left(), buff=0.15),
            Arrow(head_group.get_right(), output.get_left(), buff=0.15),
            Arrow(prior_group.get_right(), prior_encoder_group.get_left(), buff=0.15),
            Arrow(prior_encoder_group.get_top(), pooling_group.get_bottom(), buff=0.15),
        )

        # Animation
        self.play(Write(title))
        self.play(Create(axes), Write(axes_labels))
        self.play(Create(curves), Write(input_label))
        self.wait(0.5)

        for obj in [input_proj_group, hybrid_group, pooling_group, head_group, output]:
            self.play(Create(obj), run_time=0.5)

        self.play(Create(prior_group), Create(prior_encoder_group))
        self.play(Create(arrows))
        self.wait(2)