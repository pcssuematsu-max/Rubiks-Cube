"""Numeric state viewer for symmetric and linear group puzzles."""

import tkinter as Tk

import numpy as np


class GroupStateViewer(Tk.Canvas):
    """Display one-line permutations or matrices in a cube-viewer-style panel."""

    def __init__(self, master, puzzle, mini_mode=False):
        self.puzzle = puzzle
        self.mini_mode = bool(mini_mode)
        self.cell_size = 36 if mini_mode else 62
        self.margin = 10 if mini_mode else 18
        self.header = 26 if mini_mode else 38
        self.rows, self.columns = self._grid_shape()
        width = self.margin * 2 + self.columns * self.cell_size
        height = self.margin * 2 + self.header + self.rows * self.cell_size
        Tk.Canvas.__init__(
            self,
            master,
            width=width,
            height=height,
            bg="#303030",
            highlightthickness=0,
        )
        self.set_color(self.puzzle.state)

    def _grid_shape(self):
        if self.puzzle.group_kind == "linear":
            return self.puzzle.dimension, self.puzzle.dimension
        return 1, self.puzzle.degree

    def set_color(self, state):
        self.delete("state")
        values = np.asarray(state, dtype=object).reshape(-1)
        title = self._title_text()
        self.create_text(
            self.margin,
            self.margin + self.header / 2,
            text=title,
            anchor="w",
            fill="#F0F0F0",
            font=("Menlo", 9 if self.mini_mode else 13, "bold"),
            tags="state",
        )
        for index in range(self.rows * self.columns):
            self._draw_cell(index, values[index] if index < values.size else "")

    def _title_text(self):
        if self.puzzle.display_name:
            return self.puzzle.display_name
        if self.puzzle.group_kind == "linear":
            return f"{self.puzzle.family}({self.puzzle.dimension}, F_{self.puzzle.modulus})"
        return f"S_{self.puzzle.degree}  (one-line)"

    def _draw_cell(self, index, value):
        row, column = divmod(index, self.columns)
        x0 = self.margin + column * self.cell_size
        y0 = self.margin + self.header + row * self.cell_size
        x1 = x0 + self.cell_size - 2
        y1 = y0 + self.cell_size - 2
        solved_value = int(self.puzzle.state_0[index])
        value_text = "" if value is None else str(value)
        is_blank = value_text in ("", "0.0") and self.mini_mode
        try:
            is_solved = int(value) == solved_value
        except (TypeError, ValueError):
            is_solved = False
        fill = "#315A3A" if is_solved else "#6A3B3B"
        if is_blank:
            fill = "#454545"
        self.create_rectangle(x0, y0, x1, y1, fill=fill, outline="#101010", width=2, tags="state")
        label = self.puzzle.coordinate_label(index)
        self.create_text(
            (x0 + x1) / 2,
            y0 + (9 if self.mini_mode else 13),
            text=label,
            fill="#C8C8C8",
            font=("Menlo", 6 if self.mini_mode else 9),
            tags="state",
        )
        self.create_text(
            (x0 + x1) / 2,
            y0 + self.cell_size * 0.61,
            text=value_text,
            fill="#FFFFFF",
            font=("Menlo", 12 if self.mini_mode else 22, "bold"),
            tags="state",
        )


StateViewer = GroupStateViewer
