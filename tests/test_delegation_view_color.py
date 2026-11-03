"""The delegation view in colour: boxes by their node's state, the tree's edge gradient, and the depth they are painted at.

Colour is only ever SGR parameters around the same text, so every test that
paints compares the painted row, with its escapes taken out, against the
plain drawing. The background rule is measured on what reaches the terminal:
at 256 and 16 colours that is the index the painter chose, read back as the
colour a terminal draws it, so a quantized edge is held to the same rule as a
24-bit one (lighter than a dark background, darker than a light one).
"""

from __future__ import annotations

import colorsys
from io import StringIO
import os
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

from _local_package import load_local_package

load_local_package()

from _cli_harness import run_cli  # noqa: E402
from _delegation_graphs import dag_export_repair_run, dag_export_run, nested, owner_mock, parallel  # noqa: E402
from _delegation_state import PARENT, dispatched, homes, route, start  # noqa: E402
from _tui_lines import line_text  # noqa: E402
from omh.cli import main  # noqa: E402
from omh.plugin_bundle.omh.delegation_snapshot import messenger_snapshot  # noqa: E402
from omh.plugin_bundle.omh.delegation_view import ASCII, UNICODE, Focus, cells, render_delegation_view  # noqa: E402
from omh.skin_pack import SKIN_THEMES, default_theme, hex_to_rgb, skin_colors  # noqa: E402
from omh.tui import style as style_module  # noqa: E402
from omh.tui.app import TICK_SECONDS, App  # noqa: E402
from omh.tui.aura import GLINT_CELLS, aura_text  # noqa: E402
from omh.tui.escapes import strip_escapes  # noqa: E402
from omh.tui.panels import dag_line, dag_style, node_style, panel_lines  # noqa: E402
from omh.tui.screen import paint_line  # noqa: E402
from omh.tui.style import (  # noqa: E402
    COLOR_16,
    COLOR_256,
    COLOR_DEPTHS,
    NO_COLOR,
    TRUECOLOR,
    Style,
    color_depth,
    nearest_16,
    nearest_256,
)
from omh.tui.theme import (  # noqa: E402
    BLACK,
    TEXT_CONTRAST,
    DAG_EDGE_STEPS,
    DAG_EDGE_STOPS,
    EDGE_CONTRAST,
    build_theme,
    contrast_ratio,
    is_light,
    surface_for,
)
from omh.tui.width import line_width  # noqa: E402
from test_tui_app import FakePoller, ScriptedTerminal  # noqa: E402

# A navy and a pale indigo are where the default stops fall short of 3:1 and have to be lifted.
DARK = ("#000000", "#1E1E1E", "#282A36", "#002B36", "#1E3A8A")
LIGHT = ("#FFFFFF", "#FAF7F0", "#FDF6E3", "#EEEEEE", "#C7D2FE")
STATES = ("root", "waiting", "dispatched", "running", "stuck", "done", "failed", "interrupted", "unknown")


def drawings():
    """(name, view) for every shape: boxes in every state and selected, a wrapped frontier, the tree, ASCII,
    boxes under a line budget."""
    yield "export", render_delegation_view(dag_export_run(), width=100)
    yield "export-dag", render_delegation_view(dag_export_run(), width=100, focus=Focus(node="d:r1:verify"))
    yield "repair", render_delegation_view(dag_export_repair_run(), width=80)
    yield "mock", render_delegation_view(owner_mock(), width=73)
    yield "nested", render_delegation_view(nested(), width=120)
    yield "six", render_delegation_view(parallel(6), width=80)
    yield "tree", render_delegation_view(nested(), width=34)
    yield "wide-tree", render_delegation_view(nested(), width=90, layout="tree")
    yield "ascii", render_delegation_view(owner_mock(), width=73, charset="ascii")
    yield "budgeted", render_delegation_view(owner_mock(), width=98, max_lines=24)
    yield "wrapped", render_delegation_view(parallel(8), width=98)


def painted(view, theme, depth: str) -> list[str]:
    return [paint_line(line, line_width(line)) for line in (dag_line(row, view.levels, theme, depth) for row in view.lines)]


def drawn_rgb(style: Style) -> tuple[int, int, int]:
    """The colour a terminal draws for `style`'s foreground at its depth (xterm's defaults for an index)."""
    rgb = hex_to_rgb(style.fg or "")
    assert rgb is not None
    if style.depth == COLOR_16:
        return style_module._ANSI_16[nearest_16(rgb)]
    if style.depth == COLOR_256:
        index = nearest_256(rgb)
        if index >= 232:
            return (8 + 10 * (index - 232),) * 3
        cube = index - 16
        levels = style_module._CUBE_LEVELS
        return levels[cube // 36], levels[cube // 6 % 6], levels[cube % 6]
    return rgb


def graph_spans(lines) -> list:
    """The spans of the delegation block's rows (indented under the agents header), its indent left out."""
    return [span for line in lines if line_text(line).startswith("  ") for span in line[1:] if span.text.strip()]


def graph_text(lines) -> str:
    """The text of the delegation block's rows, every cell kept: `graph_spans` leaves out a span that is only
    spaces, and the aura can draw a title's space as a span of its own."""
    return "\n".join(line_text(line) for line in lines if line_text(line).startswith("  "))


RUNNING_TITLES = ("Refresh HUD docs", "Update HUD tests")  # `owner_mock`'s running nodes
LONE_SPACE_PHASE = 8  # the band covers "h " of "Refresh HUD docs": its space a span apart from both neighbours


def aura_app(phase: int) -> App:
    """A truecolor app whose clock stands at aura `phase` (`App._spinner_frame`), the owner mock in its HUD."""
    app = App(ScriptedTerminal(), None, {"COLORTERM": "truecolor", "TERM": "xterm-256color"},  # type: ignore[arg-type]
              poller=FakePoller(), clock=lambda: (phase + 0.5) * TICK_SECONDS)
    app.hud = type("Hud", (), {"payload": {"privacy": "metadata_only", "delegation_graph": owner_mock()},
                               "received_at": None})()
    return app


def luminance(rgb: tuple[int, int, int]) -> float:
    return contrast_ratio("#" + "".join(f"{channel:02X}" for channel in rgb), BLACK)


class DepthTests(unittest.TestCase):
    def test_the_depth_an_output_takes(self) -> None:
        cases = (
            ({"COLORTERM": "truecolor", "TERM": "xterm-256color"}, True, TRUECOLOR),
            ({"COLORTERM": "24bit"}, True, TRUECOLOR),
            ({"WT_SESSION": "uuid"}, True, TRUECOLOR),
            ({"GHOSTTY_RESOURCES_DIR": "/path"}, True, TRUECOLOR),
            ({"TERM_PROGRAM": "ghostty"}, True, TRUECOLOR),
            ({"TERM": "xterm-ghostty"}, True, TRUECOLOR),
            ({"VSCODE_PID": "1234"}, True, TRUECOLOR),
            ({"TERM": "xterm-256color"}, True, COLOR_256),
            ({"TERM": "xterm"}, True, COLOR_16),
            ({}, True, COLOR_16),
            ({"TERM": "dumb", "COLORTERM": "truecolor"}, True, NO_COLOR),
            ({"NO_COLOR": "1", "COLORTERM": "truecolor"}, True, NO_COLOR),
            ({"NO_COLOR": "", "COLORTERM": "truecolor"}, True, TRUECOLOR),
            ({"COLORTERM": "truecolor"}, False, NO_COLOR),
        )
        for environ, tty, expected in cases:
            with self.subTest(environ=environ, tty=tty):
                self.assertEqual(color_depth(environ, tty), expected)

    def test_each_depth_paints_its_own_parameters_and_none_keeps_the_attributes(self) -> None:
        red = Style(fg="#FF0000", bg="#000000", bold=True)
        self.assertEqual(red.sgr(), "\x1b[1;38;2;255;0;0;48;2;0;0;0m")
        self.assertEqual(Style(fg="#FF0000", bg="#000000", bold=True, depth=COLOR_256).sgr(), "\x1b[1;38;5;196;48;5;16m")
        self.assertEqual(Style(fg="#FF0000", bg="#000000", bold=True, depth=COLOR_16).sgr(), "\x1b[1;91;40m")
        self.assertEqual(Style(fg="#FFFFFF", bg="#5C5CFF", depth=COLOR_16).sgr(), "\x1b[97;104m")
        self.assertEqual(Style(fg="#808080", depth=COLOR_256).sgr(), "\x1b[38;5;244m", "a grey takes the grey ramp")
        self.assertEqual(Style(fg="#FF0000", bold=True, depth=NO_COLOR).sgr(), "\x1b[1m")
        self.assertEqual(Style(fg="#FF0000", depth=NO_COLOR).sgr(), "")
        self.assertEqual(Style(fg="not-a-colour", depth=COLOR_256).sgr(), "")


class GradientTests(unittest.TestCase):
    def test_the_edges_step_from_violet_through_indigo_to_blue(self) -> None:
        for background in (None, *DARK, *LIGHT):
            theme = build_theme(default_theme(), background)
            hues = []
            for step in theme.dag_edges:
                red, green, blue = (channel / 255 for channel in hex_to_rgb(step.fg))
                hues.append(colorsys.rgb_to_hsv(red, green, blue)[0] * 360)
            with self.subTest(background=background):
                self.assertEqual(len(theme.dag_edges), DAG_EDGE_STEPS)
                self.assertTrue(250 <= hues[0] <= 290, f"the top is violet: {hues[0]:.0f}")
                self.assertTrue(230 <= hues[DAG_EDGE_STEPS // 2] <= 245, f"the middle is indigo: {hues[4]:.0f}")
                self.assertTrue(205 <= hues[-1] <= 225, f"the bottom is blue: {hues[-1]:.0f}")
                self.assertEqual(hues, sorted(hues, reverse=True), "each step is bluer than the one above")

    def test_a_level_takes_its_place_on_the_gradient(self) -> None:
        theme = build_theme(default_theme(), "#1E1E1E")
        steps = theme.dag_edges
        self.assertEqual([dag_style("edge", theme, level, 3) for level in range(3)], [steps[0], steps[4], steps[8]])
        self.assertEqual([dag_style("edge", theme, level, 2) for level in range(2)], [steps[0], steps[8]])
        self.assertEqual(dag_style("edge", theme, 0, 1), steps[0], "one band is the top of the gradient")
        self.assertEqual(dag_style("edge", theme, 9, 3), steps[8], "a level past the last is the last")

    def test_the_stops_are_theme_tokens_a_palette_can_name(self) -> None:
        self.assertEqual(build_theme(default_theme(), "#1E1E1E").dag_edges[0].fg, DAG_EDGE_STOPS["dag_edge_start"])
        own = {"dag_edge_start": "#FF00FF", "dag_edge_end": "#00FFFF"}
        with patch("omh.tui.theme.skin_colors", side_effect=lambda name: {**skin_colors(name), **own}):
            theme = build_theme(default_theme(), "#000000")
        self.assertEqual((theme.dag_edges[0].fg, theme.dag_edges[-1].fg), ("#FF00FF", "#00FFFF"))
        self.assertEqual(theme.dag_edges[4].fg, DAG_EDGE_STOPS["dag_edge_mid"], "a stop it does not name stays")

    def test_a_box_takes_its_nodes_state_colour_from_the_theme(self) -> None:
        for background in (None, "#1E1E1E", "#FFFFFF"):
            theme = build_theme(default_theme(), background)
            middle = DAG_EDGE_STEPS // 2
            self.assertEqual(theme.dag_edges[middle].fg, DAG_EDGE_STOPS["dag_edge_mid"], "handed out is the middle stop")
            marks = {
                "selected": theme.dag_edges[0], "running": theme.dag_edges[-1], "dispatched": theme.dag_edges[middle],
                "failed": theme.error, "interrupted": theme.error, "stuck": theme.warn, "done": theme.text,
                "waiting": theme.dim, "unknown": theme.dim,
            }
            words = {
                **marks, "selected": theme.dag_words[0], "running": theme.dag_words[-1],
                "dispatched": theme.dag_words[middle], "waiting": theme.muted, "unknown": theme.muted,
            }
            for state in marks:
                with self.subTest(background=background, state=state):
                    self.assertEqual(dag_style(f"node:{state}", theme), marks[state], "a border or a glyph")
                    self.assertEqual(node_style(state, theme), marks[state])
                    self.assertEqual(dag_style(f"node-word:{state}", theme), words[state], "a status word")
                    self.assertEqual(dag_style(f"node-title:{state}", theme), Style(**{**vars(words[state]), "bold": True}))
            self.assertIsNone(dag_style("node:done", theme).fg, "done is the terminal's own foreground, light or dark")
            self.assertEqual((dag_style("connector", theme), dag_style("connector:running", theme)),
                             (theme.dim, theme.dag_edges[-1]))
        self.assertTrue(dag_style("title", theme).bold)
        self.assertEqual(dag_style("unknown", theme), theme.dim)
        self.assertEqual((dag_style("category", theme), dag_style("model", theme)), (theme.label, theme.hint))

    def test_a_boxs_words_read_at_text_contrast_and_its_lines_at_mark_contrast(self) -> None:
        """The owner's coloured lines stay at the 3:1 a mark needs; the title and status words in the same hue
        reach the 4.5:1 every word in the TUI does (`TEXT_ROLES`), on every palette, light or dark."""
        states = ("selected", "running", "dispatched", "failed", "interrupted", "stuck", "waiting", "unknown")
        for skin in SKIN_THEMES:
            for background in (*DARK, *LIGHT, None):
                theme = build_theme(skin, background)
                surface = surface_for(skin, background)
                for state in states:
                    with self.subTest(skin=skin.short_name, background=background, state=state):
                        for role in (f"node-word:{state}", f"node-title:{state}"):
                            self.assertGreaterEqual(contrast_ratio(dag_style(role, theme).fg, surface), TEXT_CONTRAST)
                        self.assertGreaterEqual(contrast_ratio(dag_style(f"node:{state}", theme).fg, surface),
                                                EDGE_CONTRAST - 0.01)
        # The default palette, measured: (word, mark) against its dark stand-in and against white. Waiting is the
        # TUI's neutral grey since the rail and kind colours (2026-10-22), no longer a tint of the skin's accent.
        expected = {
            None: {"running": (4.78, 4.48), "dispatched": (4.67, 3.69), "selected": (4.51, 3.89), "waiting": (5.66, 3.15)},
            "#FFFFFF": {"running": (4.9, 3.68), "dispatched": (4.86, 4.47), "selected": (4.63, 4.23),
                        "waiting": (6.11, 3.04)},
        }
        for background, ratios in expected.items():
            theme = build_theme(default_theme(), background)
            surface = surface_for(default_theme(), background)
            for state, (word, mark) in ratios.items():
                with self.subTest(background=background, state=state):
                    self.assertEqual((round(contrast_ratio(dag_style(f"node-word:{state}", theme).fg, surface), 2),
                                      round(contrast_ratio(dag_style(f"node:{state}", theme).fg, surface), 2)),
                                     (word, mark))

    def test_each_depth_downgrades_a_state_colour_to_its_own_parameters(self) -> None:
        theme = build_theme(default_theme(), "#1E1E1E")
        expected = {
            "running": ("\x1b[38;2;59;130;246m", "\x1b[38;5;69m", "\x1b[94m", ""),
            "selected": ("\x1b[38;2;139;92;246m", "\x1b[38;5;99m", "\x1b[94m", ""),
            "dispatched": ("\x1b[38;2;99;102;241m", "\x1b[38;5;63m", "\x1b[94m", ""),
            "failed": ("\x1b[38;2;239;83;80m", "\x1b[38;5;203m", "\x1b[91m", ""),
            "waiting": ("\x1b[38;2;101;109;119m", "\x1b[38;5;242m", "\x1b[90m", ""),  # the neutral grey
            "done": ("", "", "", ""),
        }
        for state, sequences in expected.items():
            with self.subTest(state=state):
                self.assertEqual(tuple(dag_style(f"node:{state}", theme, depth=depth).sgr()
                                       for depth in (TRUECOLOR, COLOR_256, COLOR_16, NO_COLOR)), sequences)
        # At 16 colours selected and running both reach bright blue: the `>` before the title still says which.
        selected = render_delegation_view(dag_export_run(), width=100, focus=Focus(node="d:r1:button"))
        self.assertIn("[-] > Export button", selected.text())

    def test_at_16_colours_running_selected_and_dispatched_share_bright_blue_and_differ_by_glyph(self) -> None:
        # #2223 review (the lead's call): the 16-colour table keeps one bright blue for the three, and the glyph
        # carries the difference: running `●` (`*`), dispatched `◌` (`o`), and the selection's `>` before its title.
        theme = build_theme(default_theme(), "#1E1E1E")
        self.assertEqual({dag_style(f"node:{state}", theme, depth=COLOR_16).sgr()
                          for state in ("running", "selected", "dispatched")}, {"\x1b[94m"})
        for charset in (UNICODE, ASCII):
            with self.subTest(charset=charset.name):
                self.assertEqual(len({charset.glyphs["running"], charset.glyphs["dispatched"]}), 2)
        selected = render_delegation_view(dag_export_run(), width=100, focus=Focus(node="d:r1:button"))
        self.assertIn("> Export button", selected.text())

    def test_no_line_is_ever_dark_on_a_dark_background_or_light_on_a_light_one(self) -> None:
        # Done is the terminal's own foreground, which no palette can set too dark or too light: it is not measured.
        roles = ["edge", "connector", "connector:running", "node:selected",
                 *(f"node:{state}" for state in STATES if state != "done")]
        for skin in SKIN_THEMES:
            for background in (*DARK, *LIGHT, None):
                theme = build_theme(skin, background)
                surface = surface_for(skin, background)
                light = is_light(surface)
                for role in roles:
                    for level in range(3):
                        base = dag_style(role, theme, level, 3)
                        self.assertGreaterEqual(contrast_ratio(base.fg, surface), EDGE_CONTRAST - 0.01,
                                                (skin.short_name, background, role, level))
                        for depth in (TRUECOLOR, COLOR_256, COLOR_16):
                            drawn = luminance(drawn_rgb(dag_style(role, theme, level, 3, depth)))
                            ground = luminance(hex_to_rgb(surface))
                            with self.subTest(skin=skin.short_name, background=background, role=role, depth=depth):
                                if light:
                                    self.assertLess(drawn, ground, "a line on a light background is darker than it")
                                else:
                                    self.assertGreater(drawn, ground, "a line on a dark background is lighter than it")


class PaintTests(unittest.TestCase):
    def test_colour_never_changes_the_text_or_its_width(self) -> None:
        for name, view in drawings():
            plain = view.text_lines()
            for skin in SKIN_THEMES:
                for background in (None, "#1E1E1E", "#FFFFFF"):
                    theme = build_theme(skin, background)
                    for depth in COLOR_DEPTHS:
                        rows = painted(view, theme, depth)
                        with self.subTest(view=name, skin=skin.short_name, background=background, depth=depth):
                            self.assertEqual([strip_escapes(row) for row in rows], plain)
                            lines = [dag_line(row, view.levels, theme, depth) for row in view.lines]
                            self.assertEqual([line_width(line) for line in lines],
                                             [line_width(dag_line(row, view.levels, theme)) for row in view.lines])
                            if depth == NO_COLOR:
                                self.assertFalse(any("38;" in row or "[3" in row or "[9" in row for row in rows), rows)

    def test_the_panel_rows_are_the_same_text_at_every_depth(self) -> None:
        payload = {"privacy": "metadata_only", "delegation_graph": owner_mock()}
        theme = build_theme(default_theme(), "#1E1E1E")
        plain = [line_text(line) for line in panel_lines(payload, 100, 40, theme, 0.0)]
        self.assertTrue(any("Refresh HUD docs" in line for line in plain), plain)
        for depth in COLOR_DEPTHS:
            lines = panel_lines(payload, 100, 40, theme, 0.0, color_depth=depth)
            with self.subTest(depth=depth):
                self.assertEqual([line_text(line) for line in lines], plain)
                self.assertEqual([line_width(line) for line in lines], [cells(line) for line in plain])
                self.assertTrue(all(span.style.depth == depth for span in graph_spans(lines)))
        colourless = graph_spans(panel_lines(payload, 100, 40, theme, 0.0, color_depth=NO_COLOR))
        self.assertTrue(colourless)
        self.assertFalse(any(";" in span.style.sgr() for span in colourless))

    def test_the_tui_paints_the_graph_at_the_terminals_depth(self) -> None:
        for environ, expected in (({"COLORTERM": "truecolor", "TERM": "xterm-256color"}, TRUECOLOR),
                                  ({"TERM": "xterm-256color"}, COLOR_256), ({"NO_COLOR": "1"}, NO_COLOR)):
            app = App(ScriptedTerminal(), None, environ, poller=FakePoller(),  # type: ignore[arg-type]
                      clock=lambda: (LONE_SPACE_PHASE + 0.5) * TICK_SECONDS)
            app.hud = type("Hud", (), {"payload": {"privacy": "metadata_only", "delegation_graph": owner_mock()},
                                       "received_at": None})()
            with self.subTest(environ=environ):
                self.assertEqual(app.color_depth, expected)
                lines = app._agent_view(100, 40)  # the agent view under the info line draws it
                # A running node's title is a span a character in the aura where it moves (truecolor, motion on).
                self.assertIn("Refresh HUD docs", graph_text(lines))
                self.assertTrue(all(span.style.depth == expected for span in graph_spans(lines)))

    def test_every_aura_phase_keeps_a_running_title_exactly(self) -> None:
        theme = aura_app(0).theme
        cycle = max(len(title) for title in RUNNING_TITLES) + 2 * GLINT_CELLS  # the band enters, crosses and leaves
        lone_space = []
        for phase in range(2 * cycle):
            app = aura_app(phase)
            self.assertEqual(app._aura_phase(), phase)
            lines = app._agent_view(100, 40)
            rows = graph_text(lines)
            painted_rows = "\n".join(strip_escapes(paint_line(line, line_width(line))) for line in lines)
            for title in RUNNING_TITLES:
                with self.subTest(phase=phase, title=title):
                    spans = aura_text(title, theme.aura, theme.aura_glint, phase)
                    self.assertEqual("".join(span.text for span in spans), title)
                    self.assertIn(title, rows)
                    self.assertIn(title, painted_rows, "the terminal is sent every cell, the space included")
            if "Refresh HUD docs" not in "".join(span.text for span in graph_spans(lines)):
                lone_space.append(phase)  # a title's space drawn as a span of its own, which `graph_spans` leaves out
        self.assertIn(LONE_SPACE_PHASE, lone_space, "the sweep reaches the phases that dropped the space before")

    def test_a_chat_post_is_plain_text(self) -> None:
        with homes() as home:
            now = time.time()
            home.session(PARENT, "claude-opus-5-5", effort="high")
            route(home, "deep", "claude-opus-5-5", "high", written_at=now - 3)
            start(home, "child-0", "sa-0", "Port the auth module", now=now - 2)
            home.session("child-0", "claude-opus-5-5", parent=PARENT, effort="high")
            dispatched(home, ["sa-0"], now=now)
            with patch.dict(os.environ, {"COLORTERM": "truecolor", "TERM": "xterm-256color", "COLORFGBG": "15;0"}):
                block = messenger_snapshot(session_id=PARENT, platform="discord", model="claude-opus-5-5")
        self.assertIn("Port the auth module", block)
        self.assertNotIn("\x1b", block)


class _Terminal(StringIO):
    encoding = "utf-8"

    def isatty(self) -> bool:
        return True


class HudCommandTests(unittest.TestCase):
    def run_hud(self, home, environ: dict[str, str], *, tty: bool = True, extra: tuple[str, ...] = ()) -> str:
        argv = ["--omh-home", str(home.omh), "--hermes-home", str(home.hermes), "hud", "--graph", *extra]
        with patch.dict(os.environ, environ):
            for name in (
                "NO_COLOR", "COLORFGBG", "COLORTERM", "OMH_OUTPUT",
                "WT_SESSION", "WT_PROFILE_ID", "GHOSTTY_RESOURCES_DIR",
                "KITTY_WINDOW_ID", "WEZTERM_PANE", "VSCODE_PID", "TERM_PROGRAM",
            ):
                if name not in environ:
                    os.environ.pop(name, None)
            if not tty:
                return run_cli(argv, output_json=False)[1]
            stdout = _Terminal()
            with redirect_stdout(stdout), redirect_stderr(StringIO()):
                self.assertEqual(main(argv), 0)
            return stdout.getvalue()

    def test_omh_hud_graph_paints_only_where_the_background_is_named(self) -> None:
        with homes() as home:
            now = time.time()
            home.session(PARENT, "claude-opus-5-5", effort="high")
            for index, title in enumerate(("Refresh HUD docs", "Polish HUD CLI")):
                route(home, "quick", "glm-5.3-flash", "low", written_at=now - 5 + index)
                start(home, f"child-{index}", f"sa-{index}", title, now=now - 4 + index)
                home.session(f"child-{index}", "glm-5.3-flash", parent=PARENT, effort="low")
            dispatched(home, ["sa-0", "sa-1"], now=now)
            truecolor = {"COLORTERM": "truecolor", "TERM": "xterm-256color"}
            plain = self.run_hud(home, {**truecolor, "NO_COLOR": "1", "COLORFGBG": "15;0"})
            self.assertIn("Refresh HUD docs", plain)
            self.assertNotIn("\x1b", plain)
            for colorfgbg in ("15;0", "0;15"):
                coloured = self.run_hud(home, {**truecolor, "COLORFGBG": colorfgbg})
                with self.subTest(colorfgbg=colorfgbg):
                    self.assertIn("\x1b[38;2;", coloured)
                    self.assertEqual(strip_escapes(coloured), plain)
            self.assertIn("\x1b[38;5;", self.run_hud(home, {"TERM": "xterm-256color", "COLORFGBG": "15;0"}))
            unnamed = self.run_hud(home, truecolor)
            self.assertEqual(unnamed, plain, "no background named, so no colour measured against a guess")
            piped = self.run_hud(home, {**truecolor, "COLORFGBG": "15;0"}, tty=False)
            self.assertNotIn("\x1b", piped)
            as_json = self.run_hud(home, {**truecolor, "COLORFGBG": "15;0"}, extra=("--json",))
            self.assertNotIn("\x1b", as_json)


if __name__ == "__main__":
    unittest.main()
