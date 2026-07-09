"""Small dialog button widgets."""

import tkinter as Tk
from tkinter.scrolledtext import ScrolledText

import numpy as np

from cube.rubiks_cube import Rubiks_3
from cto.cube import CtoCube
from fto.cube import FtoCube
from megaminx.cube import MegaminxCube
from pyraminx.cube import MasterPyraminxCube, PyraminxCube
from skewb.cube import SkewbCube
from ui.cto.state_viewer import CtoStateViewer
from ui.fto.state_viewer import FtoStateViewer
from ui.megaminx.state_viewer import MegaminxStateViewer
from ui.pyraminx.state_viewer import PyraminxStateViewer
from ui.skewb.state_viewer import SkewbStateViewer
from ui.viewers import StateViewer

class MakeMypermOkButton(Tk.Button):
    def __init__(self,master,frame,entry):
        Tk.Button.__init__(self,master,text = 'OK',command = self.make_myperm)
        self.frame = frame
        self.entry = entry
        self.master = master

    def make_myperm(self):
        text = self.entry.get()
        self.frame.myperm_manager.apply_named_myperm(text)
        self.master.destroy()


class LpShowKeyButton(Tk.Button):
    def __init__(self,master,frame,entry_key,entry_length):
        Tk.Button.__init__(self,master,text = 'Show',command = self.show_key)
        self.frame = frame
        self.entry_key = entry_key
        self.entry_length = entry_length
        self.master = master

    def show_key(self):
        text_key = self.entry_key.get().strip()
        text_length = self.entry_length.get().strip()
        if text_length == '':
            length = None
        else:
            length = int(text_length)
        self.frame.lp_show(text_key,length)
        self.master.destroy()


# Backward-compatible aliases during dialog rename migration.
make_myperm_OK = MakeMypermOkButton
lp_show_key = LpShowKeyButton


class ParamEditorDialog(Tk.Toplevel):
    """AI parameter viewer/editor dialog."""

    def __init__(self, frame):
        Tk.Toplevel.__init__(self, frame)
        self.frame = frame
        self.title('param editor')
        self.font = ('Century Gothic', 12, 'bold')
        self.ai_index_var = Tk.StringVar(value = self._default_ai_text())
        self.param_key_var = Tk.StringVar()
        self.param_index_var = Tk.StringVar(value = '')
        self.param_value_var = Tk.StringVar(value = '')
        self.summary_var = Tk.StringVar(value = '')
        self.status_var = Tk.StringVar(value = '')
        self._build_widgets()
        self._refresh_param_keys()

    def _default_ai_text(self):
        text = self.frame.param_index_var.get().strip()
        if text != '':
            return text.split(',')[0].strip()
        return str(self.frame.AI_idx)

    def _build_widgets(self):
        Tk.Label(self, text = 'AI idx', font = self.font).grid(row = 0, column = 0, sticky = 'e')
        Tk.Entry(self, textvariable = self.ai_index_var, font = self.font, width = 8).grid(row = 0, column = 1, sticky = 'ew')
        Tk.Button(self, text = 'refresh', font = self.font, command = self._refresh_param_keys).grid(row = 0, column = 2, sticky = 'ew')

        Tk.Label(self, text = 'param', font = self.font).grid(row = 1, column = 0, sticky = 'e')
        self.param_menu = Tk.OptionMenu(self, self.param_key_var, '')
        self.param_menu.configure(font = self.font)
        self.param_menu.grid(row = 1, column = 1, columnspan = 2, sticky = 'ew')

        Tk.Label(self, text = 'index', font = self.font).grid(row = 2, column = 0, sticky = 'e')
        Tk.Entry(self, textvariable = self.param_index_var, font = self.font).grid(row = 2, column = 1, sticky = 'ew')
        Tk.Button(self, text = 'load value', font = self.font, command = self._load_value).grid(row = 2, column = 2, sticky = 'ew')

        Tk.Label(self, text = 'value', font = self.font).grid(row = 3, column = 0, sticky = 'e')
        Tk.Entry(self, textvariable = self.param_value_var, font = self.font).grid(row = 3, column = 1, sticky = 'ew')
        Tk.Button(self, text = 'apply', font = self.font, command = self._apply_value).grid(row = 3, column = 2, sticky = 'ew')

        Tk.Label(self, textvariable = self.summary_var, justify = Tk.LEFT, anchor = 'w').grid(row = 4, column = 0, columnspan = 3, sticky = 'ew')

        self.preview = ScrolledText(self, width = 72, height = 12, font = ('Courier', 11))
        self.preview.grid(row = 5, column = 0, columnspan = 3, sticky = 'nsew')
        self.preview.configure(state = Tk.DISABLED)

        Tk.Label(self, textvariable = self.status_var, justify = Tk.LEFT, anchor = 'w', fg = '#7A1F1F').grid(row = 6, column = 0, columnspan = 3, sticky = 'ew')

        for column in range(3):
            self.grid_columnconfigure(column, weight = 1)
        self.grid_rowconfigure(5, weight = 1)

        self.param_key_var.trace_add('write', lambda *_: self._refresh_preview())

    def _selected_ai_index(self):
        return self.frame.param_manager.selected_index(self.ai_index_var.get(), default = self.frame.AI_idx)

    def _refresh_param_keys(self):
        ai_index = self._selected_ai_index()
        keys = self.frame.param_manager.param_keys(ai_index)
        menu = self.param_menu['menu']
        menu.delete(0, 'end')
        for key in keys:
            menu.add_command(label = key, command = lambda value = key: self.param_key_var.set(value))
        if keys:
            current = self.param_key_var.get()
            self.param_key_var.set(current if current in keys else keys[0])
        self.status_var.set('')
        self._refresh_preview()

    def _refresh_preview(self):
        key = self.param_key_var.get()
        if key == '':
            return
        ai_index = self._selected_ai_index()
        summary = self.frame.param_manager.param_summary(ai_index, key)
        self.summary_var.set(
            f"shape={summary['shape']} size={summary['size']} dtype={summary['dtype']} "
            f"min={summary['min']:.6g} max={summary['max']:.6g} "
            f"mean={summary['mean']:.6g} std={summary['std']:.6g}"
        )
        preview_text = self.frame.param_manager.param_preview(ai_index, key)
        self.preview.configure(state = Tk.NORMAL)
        self.preview.delete('1.0', Tk.END)
        self.preview.insert(Tk.END, preview_text)
        self.preview.configure(state = Tk.DISABLED)

    def _load_value(self):
        key = self.param_key_var.get()
        ai_index = self._selected_ai_index()
        try:
            value = self.frame.param_manager.param_value(ai_index, key, self.param_index_var.get())
        except ValueError as exc:
            self.status_var.set(str(exc))
            return
        self.param_value_var.set(f'{value:.8g}')
        self.status_var.set('')

    def _apply_value(self):
        key = self.param_key_var.get()
        ai_index = self._selected_ai_index()
        try:
            value = self.frame.param_manager.set_param_value(
                ai_index,
                key,
                self.param_index_var.get(),
                self.param_value_var.get(),
            )
        except ValueError as exc:
            self.status_var.set(str(exc))
            return
        self.status_var.set(f'updated {key} = {value:.8g}')
        self._refresh_preview()
        self.frame.append_log(
            f'param updated: ai={ai_index} key={key} index={self.param_index_var.get() or "()"} value={value:.8g}'
        )


class ToolsDialog(Tk.Toplevel):
    """画面外へ追い出した補助操作をまとめる小さいツールパネル。"""

    def __init__(self, frame):
        Tk.Toplevel.__init__(self, frame)
        self.frame = frame
        self.title('tools')
        self.font = ('Century Gothic', 12, 'bold')
        self._build_widgets()

    def _build_widgets(self):
        buttons = [
            ('make myperm', self.frame.make_myperm),
            ('edit params', self.frame.open_param_editor),
            ('dataset inspector', self.frame.open_dataset_inspector),
            ('attention analysis', self.frame.analyze_transformer_attention),
            ('embedding analysis', self.frame.analyze_transformer_embedding),
            ('embedding map', self.frame.open_w1_embedding_map),
            ('lpk', self.frame.lpk),
            ('lp show', self.frame.lp_show_by_button),
            ('manual moves', self.frame.toggle_move_pad),
        ]
        for row_index, (text, command) in enumerate(buttons):
            Tk.Button(self, text = text, font = self.font, command = command).grid(
                row = row_index,
                column = 0,
                sticky = 'ew',
                padx = 4,
                pady = 2,
            )
        self.grid_columnconfigure(0, weight = 1)


class AnalysisScoresDialog(Tk.Toplevel):
    """Occlusion などのスコア一覧を表示するダイアログ。"""

    def __init__(self, frame):
        Tk.Toplevel.__init__(self, frame)
        self.title('analysis scores')
        self.geometry('420x320')
        self.text = ScrolledText(self, width = 60, height = 18, font = ('Menlo', 11))
        self.text.pack(fill = 'both', expand = True)
        self.text.configure(state = Tk.DISABLED)

    def set_content(self, title, rows):
        self.title(title)
        self.text.configure(state = Tk.NORMAL)
        self.text.delete('1.0', Tk.END)
        self.text.insert(Tk.END, title + '\n')
        self.text.insert(Tk.END, '-' * len(title) + '\n')
        for label, score in rows:
            self.text.insert(Tk.END, f'{label:<24} {score: .6f}\n')
        self.text.configure(state = Tk.DISABLED)

    def set_text_content(self, title, content):
        self.title(title)
        self.text.configure(state = Tk.NORMAL)
        self.text.delete('1.0', Tk.END)
        self.text.insert(Tk.END, content)
        self.text.configure(state = Tk.DISABLED)


class W1EmbeddingDialog(Tk.Toplevel):
    """PCA/t-SNE scatter viewer for W1 and its attention projections."""

    TYPE_COLORS = {
        'corner': '#d1495b',
        'midedge': '#2a9d8f',
        'edge': '#00798c',
        'center': '#edae49',
        'tip': '#7a5195',
        'diagonal': '#d1495b',
        'offdiagonal': '#4c78a8',
        'position': '#2a9d8f',
    }
    FALLBACK_COLORS = (
        '#7F0000', '#FF7F7F', '#FF7F00', '#FFBF7F', '#BFBF00', '#FFFF7F',
        '#007F00', '#7FFF7F', '#0000BF', '#7F7FFF', '#3F007F', '#BF7FFF',
    )
    FEATURE_COLOR_FILLS = {
        'R': '#d73027',
        'O': '#f28e2b',
        'Y': '#edc948',
        'W': '#f4f4f4',
        'G': '#2ca02c',
        'B': '#2878b5',
        'X': '#8b9098',
    }
    FEATURE_COLOR_NAMES = {
        'R': 'Red',
        'O': 'Orange',
        'Y': 'Yellow',
        'W': 'White',
        'G': 'Green',
        'B': 'Blue',
        'X': 'Masked',
    }

    def __init__(self, frame):
        Tk.Toplevel.__init__(self, frame)
        self.frame = frame
        self.title('embedding map')
        self.geometry('1020x760')
        self.minsize(760, 560)
        self.font = ('Century Gothic', 11, 'bold')
        self.ai_index_var = Tk.StringVar(value = self._default_ai_text())
        self.method_var = Tk.StringVar(value = 'PCA')
        self.point_mode_var = Tk.StringVar(value = 'Feature columns')
        self.embedding_source_var = Tk.StringVar(value = 'W1')
        self.pca_axes_var = Tk.StringVar(value = 'PC1 / PC2')
        self.color_mode_var = Tk.StringVar(value = 'Piece type')
        self.highlight_correct_var = Tk.BooleanVar(value = True)
        self.show_feature_colors_var = Tk.BooleanVar(value = False)
        self.focus_piece_type_var = Tk.StringVar(value = 'Any type')
        self.focus_color_vars = {
            color: Tk.BooleanVar(value = False)
            for color in ('R', 'O', 'Y', 'W', 'G', 'B')
        }
        self.max_points_var = Tk.StringVar(value = '600')
        self.perplexity_var = Tk.StringVar(value = '20')
        self.iterations_var = Tk.StringVar(value = '300')
        self.summary_var = Tk.StringVar(value = '')
        self.hover_var = Tk.StringVar(value = '')
        self.result = None
        self._build_widgets()
        self.after_idle(self.refresh)

    def _default_ai_text(self):
        if hasattr(self.frame, '_debug_viewer_ai_index'):
            return str(self.frame._debug_viewer_ai_index())
        return str(self.frame.AI_idx)

    def _build_widgets(self):
        controls = Tk.Frame(self)
        controls.pack(fill = 'x', padx = 8, pady = 6)
        primary_controls = Tk.Frame(controls)
        primary_controls.pack(fill = 'x')
        Tk.Label(primary_controls, text = 'AI', font = self.font).pack(side = 'left')
        Tk.Entry(primary_controls, textvariable = self.ai_index_var, font = self.font, width = 5).pack(side = 'left', padx = (3, 10))
        Tk.OptionMenu(primary_controls, self.method_var, 'PCA', 't-SNE').pack(side = 'left')
        Tk.OptionMenu(primary_controls, self.point_mode_var, 'Feature columns', 'Piece centroids').pack(side = 'left', padx = 5)
        Tk.OptionMenu(
            primary_controls,
            self.color_mode_var,
            'Piece type',
            'Solve group',
            command = lambda value: self._draw_result(),
        ).pack(side = 'left')
        Tk.Button(primary_controls, text = 'Run', font = self.font, command = self.refresh).pack(side = 'left', padx = 8)

        detail_controls = Tk.Frame(controls)
        detail_controls.pack(fill = 'x', pady = (4, 0))
        Tk.Label(detail_controls, text = 'source', font = self.font).pack(side = 'left', padx = (0, 2))
        Tk.OptionMenu(
            detail_controls,
            self.embedding_source_var,
            'W1',
            'WQ1 @ W1',
            'WK1 @ W1',
            'WV1 @ W1',
        ).pack(side = 'left')
        Tk.Label(detail_controls, text = 'axes', font = self.font).pack(side = 'left', padx = (10, 2))
        Tk.OptionMenu(
            detail_controls,
            self.pca_axes_var,
            'PC1 / PC2',
            'PC1 / PC3',
            'PC2 / PC3',
            'PC3 / PC4',
            'PC4 / PC5',
            'PC5 / PC6',
        ).pack(side = 'left')

        run_controls = Tk.Frame(controls)
        run_controls.pack(fill = 'x', pady = (4, 0))
        Tk.Label(run_controls, text = 'max points', font = self.font).pack(side = 'left', padx = (0, 2))
        Tk.Spinbox(run_controls, from_ = 20, to = 2000, increment = 20, textvariable = self.max_points_var, width = 6).pack(side = 'left')
        Tk.Label(run_controls, text = 'perplexity', font = self.font).pack(side = 'left', padx = (10, 2))
        Tk.Spinbox(run_controls, from_ = 2, to = 100, textvariable = self.perplexity_var, width = 5).pack(side = 'left')
        Tk.Label(run_controls, text = 'steps', font = self.font).pack(side = 'left', padx = (10, 2))
        Tk.Spinbox(run_controls, from_ = 100, to = 1000, increment = 50, textvariable = self.iterations_var, width = 5).pack(side = 'left')
        Tk.Checkbutton(
            run_controls,
            text = 'highlight correct',
            variable = self.highlight_correct_var,
            command = self._draw_result,
            font = self.font,
        ).pack(side = 'left', padx = 10)
        Tk.Checkbutton(
            run_controls,
            text = 'show colors',
            variable = self.show_feature_colors_var,
            command = self._draw_result,
            font = self.font,
        ).pack(side = 'left')

        focus_controls = Tk.Frame(controls)
        focus_controls.pack(fill = 'x', pady = (4, 0))
        Tk.Label(focus_controls, text = 'focus', font = self.font).pack(side = 'left', padx = (0, 2))
        Tk.OptionMenu(
            focus_controls,
            self.focus_piece_type_var,
            'Any type',
            'Center',
            'Edge',
            'Corner',
            command = lambda value: self._draw_result(),
        ).pack(side = 'left')
        for color in ('R', 'O', 'Y', 'W', 'G', 'B'):
            Tk.Checkbutton(
                focus_controls,
                text = self.FEATURE_COLOR_NAMES[color],
                variable = self.focus_color_vars[color],
                command = self._draw_result,
                selectcolor = self.FEATURE_COLOR_FILLS[color],
                font = ('Century Gothic', 10, 'bold'),
            ).pack(side = 'left', padx = (5, 0))
        Tk.Button(focus_controls, text = 'Clear', font = self.font, command = self._clear_focus).pack(side = 'left', padx = 8)

        self.canvas = Tk.Canvas(self, bg = '#ffffff', highlightthickness = 1, highlightbackground = '#b8bcc2')
        self.canvas.pack(fill = 'both', expand = True, padx = 8, pady = (0, 6))
        self.canvas.bind('<Configure>', lambda event: self._draw_result())
        self.metrics_var = Tk.StringVar(value = '')
        self.focus_status_var = Tk.StringVar(value = '')
        Tk.Label(self, textvariable = self.focus_status_var, anchor = 'w', justify = Tk.LEFT, font = ('Menlo', 10, 'bold')).pack(fill = 'x', padx = 8)
        self.summary_label = Tk.Label(self, textvariable = self.summary_var, anchor = 'w', justify = Tk.LEFT, font = ('Menlo', 10), wraplength = 980)
        self.summary_label.pack(fill = 'x', padx = 8)
        self.metrics_label = Tk.Label(self, textvariable = self.metrics_var, anchor = 'w', justify = Tk.LEFT, font = ('Menlo', 10), wraplength = 980)
        self.metrics_label.pack(fill = 'x', padx = 8, pady = (2, 0))
        Tk.Label(self, textvariable = self.hover_var, anchor = 'w', justify = Tk.LEFT, font = ('Menlo', 10)).pack(fill = 'x', padx = 8, pady = (2, 7))
        self.bind('<Configure>', self._update_text_wrap)

    def _clear_focus(self):
        self.focus_piece_type_var.set('Any type')
        for variable in self.focus_color_vars.values():
            variable.set(False)
        self._draw_result()

    def _update_text_wrap(self, event):
        wrap_length = max(500, event.width - 24)
        self.summary_label.configure(wraplength = wrap_length)
        self.metrics_label.configure(wraplength = wrap_length)

    def refresh(self):
        try:
            ai_index = self.frame.param_manager.selected_index(
                self.ai_index_var.get(),
                default = self.frame.AI_idx,
            )
            max_points = max(2, int(self.max_points_var.get()))
            perplexity = float(self.perplexity_var.get())
            iterations = max(100, int(self.iterations_var.get()))
            pca_axes = tuple(int(value.strip()[2:]) for value in self.pca_axes_var.get().split('/'))
        except ValueError as error:
            self.summary_var.set(str(error))
            return
        self.summary_var.set(f'Computing {self.method_var.get()}...')
        self.metrics_var.set('')
        self.hover_var.set('')
        self.update_idletasks()
        try:
            self.result = self.frame.debug_analysis_manager.w1_embedding_projection(
                ai_index,
                method = self.method_var.get(),
                point_mode = self.point_mode_var.get(),
                max_points = max_points,
                perplexity = perplexity,
                iterations = iterations,
                embedding_source = self.embedding_source_var.get(),
                pca_axes = pca_axes,
            )
        except Exception as error:
            self.result = None
            self.canvas.delete('all')
            self.summary_var.set(f'Embedding map failed: {error}')
            return
        self._draw_result()
        separation = self.result['separation']
        group_separation = self.result['group_separation']
        correct_count = int(np.count_nonzero(self.result['correct_flags']))
        self.summary_var.set(
            f"AI {ai_index}  {self.result['model_type']}  source={self.result['embedding_source']} "
            f"shape={self.result['embedding_shape']}  "
            f"points={len(self.result['labels'])}  {self.result['method']} ({self.result['method_detail']})  "
            f"correct={correct_count}  type/group separation={separation['ratio']:.3f}/{group_separation['ratio']:.3f}"
        )
        type_metric = self.result['full_metrics']['piece_type']
        group_metric = self.result['full_metrics']['solve_group']
        self.metrics_var.set(
            'Full-dimensional silhouette  '
            f"type={type_metric['silhouette']:.4f} "
            f"(shuffle {type_metric['baseline_mean']:.4f}+/-{type_metric['baseline_std']:.4f}, z={type_metric['z_score']:.2f})  "
            f"group={group_metric['silhouette']:.4f} "
            f"(shuffle {group_metric['baseline_mean']:.4f}+/-{group_metric['baseline_std']:.4f}, z={group_metric['z_score']:.2f})  "
            f"full type/group separation={type_metric['separation_ratio']:.3f}/{group_metric['separation_ratio']:.3f}"
        )

    def _draw_result(self):
        if self.result is None or not hasattr(self, 'canvas'):
            return
        self.canvas.delete('all')
        width = max(400, self.canvas.winfo_width())
        height = max(300, self.canvas.winfo_height())
        legend_width = min(210, max(145, width * 0.22))
        left, top, bottom = 54.0, 30.0, height - 42.0
        right = width - legend_width - 20.0
        coordinates = np.asarray(self.result['coordinates'], dtype = 'f')
        screen = self._screen_coordinates(coordinates, left, top, right, bottom)
        self._draw_axes(coordinates, left, top, right, bottom)

        category_values = self._category_values()
        categories = self._ordered_categories(category_values)
        colors = {
            category: self._category_color(category, index)
            for index, category in enumerate(categories)
        }
        correct_flags = np.asarray(self.result['correct_flags'], dtype = bool)
        focus_active, focus_matches, focus_text = self._focus_matches()
        match_count = int(np.count_nonzero(focus_matches))
        self.focus_status_var.set(
            f'Focus: {focus_text}  matches={match_count}/{len(focus_matches)}'
            if focus_active else ''
        )
        draw_order = sorted(
            range(len(screen)),
            key = lambda index: (bool(focus_matches[index]), bool(correct_flags[index])),
        )
        for index in draw_order:
            x, y = screen[index]
            category = category_values[index]
            focus_match = bool(focus_matches[index])
            is_correct = focus_match and bool(correct_flags[index]) and self.highlight_correct_var.get()
            if self.show_feature_colors_var.get() and focus_match:
                self._draw_feature_color_swatches(float(x), float(y), self.result['feature_colors'][index])
            if not focus_match:
                radius = 2
            elif is_correct:
                radius = 7
            elif focus_active:
                radius = 6
            else:
                radius = 4
            item = self._draw_marker(
                float(x),
                float(y),
                self.result['piece_types'][index],
                colors[category] if focus_match else '#d6d9dd',
                radius = radius,
                outline = '#111418' if is_correct else '',
                width = 2 if is_correct else 1,
            )
            self.canvas.tag_bind(item, '<Enter>', lambda event, point_index = index: self._show_point(point_index))
            self.canvas.tag_bind(item, '<Leave>', lambda event: self.hover_var.set(''))

        for category in categories:
            indices = [
                index for index, value in enumerate(category_values)
                if value == category and focus_matches[index]
            ]
            if not indices:
                continue
            center = np.mean(screen[indices], axis = 0)
            self._draw_category_center(float(center[0]), float(center[1]), category, colors[category])
        self._draw_legend(
            categories,
            category_values,
            colors,
            right + 24.0,
            top + 4.0,
            focus_active,
            match_count,
        )

    def _category_values(self):
        if self.color_mode_var.get() == 'Solve group':
            return self.result['solve_groups']
        return self.result['piece_types']

    def _focus_matches(self):
        selected_type = self.focus_piece_type_var.get()
        selected_colors = {
            color for color, variable in self.focus_color_vars.items()
            if variable.get()
        }
        active = selected_type != 'Any type' or bool(selected_colors)
        matches = []
        for piece_type, feature_colors in zip(self.result['piece_types'], self.result['feature_colors']):
            type_match = selected_type == 'Any type' or selected_type.lower() in piece_type.lower()
            color_match = selected_colors.issubset(set(str(color) for color in feature_colors))
            matches.append(type_match and color_match)
        if not active:
            matches = [True] * len(matches)
        parts = []
        if selected_type != 'Any type':
            parts.append(selected_type)
        parts.extend(self.FEATURE_COLOR_NAMES[color] for color in ('R', 'O', 'Y', 'W', 'G', 'B') if color in selected_colors)
        return active, np.asarray(matches, dtype = bool), ' + '.join(parts) if parts else 'All'

    def _screen_coordinates(self, coordinates, left, top, right, bottom):
        minimum, span = self._projection_bounds(coordinates)
        screen = np.empty_like(coordinates, dtype = 'f')
        screen[:,0] = left + (coordinates[:,0] - minimum[0]) / span[0] * (right - left)
        screen[:,1] = bottom - (coordinates[:,1] - minimum[1]) / span[1] * (bottom - top)
        return screen

    def _projection_bounds(self, coordinates):
        minimum = np.min(coordinates, axis = 0).astype('f')
        maximum = np.max(coordinates, axis = 0).astype('f')
        raw_span = maximum - minimum
        stable_span = np.maximum(raw_span, 1.0e-8)
        minimum -= stable_span * 0.06
        maximum += stable_span * 0.06
        return minimum, np.maximum(maximum - minimum, 1.0e-8)

    def _draw_axes(self, coordinates, left, top, right, bottom):
        self.canvas.create_rectangle(left, top, right, bottom, outline = '#d5d8dc')
        method = self.result['method']
        axes = self.result['pca_axes']
        x_label = f'PC{axes[0]}' if method == 'PCA' else 't-SNE 1'
        y_label = f'PC{axes[1]}' if method == 'PCA' else 't-SNE 2'
        self.canvas.create_text((left + right) / 2, bottom + 24, text = x_label, fill = '#4b5159')
        self.canvas.create_text(left - 28, (top + bottom) / 2, text = y_label, fill = '#4b5159', angle = 90)
        minimum, span = self._projection_bounds(coordinates)
        if np.min(coordinates[:,0]) <= 0.0 <= np.max(coordinates[:,0]):
            zero = left + (0.0 - minimum[0]) / span[0] * (right - left)
            self.canvas.create_line(zero, top, zero, bottom, fill = '#eceef0')
        if np.min(coordinates[:,1]) <= 0.0 <= np.max(coordinates[:,1]):
            zero = bottom - (0.0 - minimum[1]) / span[1] * (bottom - top)
            self.canvas.create_line(left, zero, right, zero, fill = '#eceef0')

    def _ordered_categories(self, piece_types):
        preferred = (
            'Corner', 'MidEdge', 'Wing-Layer2', 'Wing-Layer3',
            'XCenter-Layer2', 'XCenter-Layer3',
            'PlusCenter-Layer2', 'PlusCenter-Layer3',
            'ObliqueCenter-A', 'ObliqueCenter-B', 'CoreCenter',
            'Edge', 'Center',
        )
        values = list(dict.fromkeys(piece_types))
        return [value for value in preferred if value in values] + sorted(value for value in values if value not in preferred)

    def _category_color(self, category, index):
        if self.color_mode_var.get() == 'Piece type':
            return self._type_color(category, index)
        return self.FALLBACK_COLORS[index % len(self.FALLBACK_COLORS)]

    def _type_color(self, category, index):
        key = category.lower().replace('-', '')
        for type_name, color in self.TYPE_COLORS.items():
            if type_name in key:
                return color
        return self.FALLBACK_COLORS[index % len(self.FALLBACK_COLORS)]

    def _draw_marker(self, x, y, category, fill, radius = 4, outline = '', width = 1):
        key = category.lower()
        options = {'fill': fill, 'outline': outline, 'width': width}
        if 'corner' in key:
            return self.canvas.create_polygon(x, y - radius, x - radius, y + radius, x + radius, y + radius, **options)
        if 'center' in key:
            return self.canvas.create_rectangle(x - radius, y - radius, x + radius, y + radius, **options)
        if 'edge' in key:
            return self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, **options)
        return self.canvas.create_polygon(x, y - radius, x - radius, y, x, y + radius, x + radius, y, **options)

    def _draw_feature_color_swatches(self, x, y, colors):
        if not colors:
            return
        radius = 2.5
        start_x = x + 7.0
        for index, color in enumerate(colors[:3]):
            center_x = start_x + index * 5.5
            fill = self.FEATURE_COLOR_FILLS.get(str(color), '#6f7680')
            self.canvas.create_oval(
                center_x - radius,
                y - radius,
                center_x + radius,
                y + radius,
                fill = fill,
                outline = '#31353a',
                width = 1,
                state = Tk.DISABLED,
            )

    def _draw_category_center(self, x, y, category, color):
        self.canvas.create_oval(x - 6, y - 6, x + 6, y + 6, fill = '#ffffff', outline = color, width = 2)
        self.canvas.create_line(x - 9, y, x + 9, y, fill = '#171a1f', width = 1)
        self.canvas.create_line(x, y - 9, x, y + 9, fill = '#171a1f', width = 1)
        self.canvas.create_text(
            x + 11,
            y - 10,
            text = category,
            anchor = 'sw',
            fill = '#171a1f',
            font = ('Century Gothic', 9, 'bold'),
        )

    def _draw_legend(self, categories, category_values, colors, x, y, focus_active = False, match_count = 0):
        self.canvas.create_text(x, y, text = self.color_mode_var.get(), anchor = 'nw', fill = '#171a1f', font = self.font)
        for row, category in enumerate(categories):
            marker_y = y + 30 + row * 27
            first_index = category_values.index(category)
            piece_type = self.result['piece_types'][first_index]
            self._draw_marker(x + 7, marker_y, piece_type, colors[category], radius = 6, outline = '#171a1f')
            count = category_values.count(category)
            self.canvas.create_text(x + 20, marker_y, text = f'{category} ({count})', anchor = 'w', fill = '#2d333a')
        if self.highlight_correct_var.get():
            marker_y = y + 42 + len(categories) * 27
            self.canvas.create_oval(x + 1, marker_y - 6, x + 13, marker_y + 6, fill = '#ffffff', outline = '#111418', width = 2)
            self.canvas.create_text(x + 20, marker_y, text = 'current correct', anchor = 'w', fill = '#2d333a')
        if focus_active:
            marker_y = y + 69 + len(categories) * 27
            self.canvas.create_oval(x + 1, marker_y - 6, x + 13, marker_y + 6, fill = '#ffffff', outline = '#4b5159', width = 2)
            self.canvas.create_text(x + 20, marker_y, text = f'focus match ({match_count})', anchor = 'w', fill = '#2d333a')

    def _show_point(self, index):
        feature_index = int(self.result['feature_indices'][index])
        feature_text = 'piece mean' if feature_index < 0 else f'input column={feature_index}'
        norm = float(np.linalg.norm(self.result['embeddings'][index]))
        x, y = self.result['coordinates'][index]
        correct_text = 'yes' if self.result['correct_flags'][index] else 'no'
        color_text = '/'.join(
            self.FEATURE_COLOR_NAMES.get(str(color), str(color))
            for color in self.result['feature_colors'][index]
        ) or 'n/a'
        self.hover_var.set(
            f"{self.result['labels'][index]}  |  group={self.result['solve_groups'][index]}  |  "
            f"colors={color_text}  |  correct={correct_text}  |  {feature_text}  |  norm={norm:.6f}  |  "
            f"({float(x):.5f}, {float(y):.5f})"
        )


class DatasetInspectorDialog(Tk.Toplevel):
    """Search2 / Search3 dataset summary viewer."""

    def __init__(self, frame):
        Tk.Toplevel.__init__(self, frame)
        self.frame = frame
        self.title('dataset inspector')
        self.geometry('620x520')
        self.font = ('Century Gothic', 12, 'bold')
        self.ai_index_var = Tk.StringVar(value = self._default_ai_text())
        self.dataset_kind_var = Tk.StringVar(value = 'Search2')
        self.sample_key_var = Tk.StringVar(value = '')
        self.sample_selector_var = Tk.StringVar(value = 'perfect_key')
        self._build_widgets()
        self.refresh()

    def _default_ai_text(self):
        text = self.frame.param_index_var.get().strip()
        if text != '':
            return text.split(',')[0].strip()
        return str(self.frame.AI_idx)

    def _build_widgets(self):
        Tk.Label(self, text = 'AI idx', font = self.font).pack(anchor = 'w', padx = 6, pady = (6, 2))
        entry_frame = Tk.Frame(self)
        entry_frame.pack(fill = 'x', padx = 6)
        Tk.Entry(entry_frame, textvariable = self.ai_index_var, font = self.font, width = 8).pack(side = 'left')
        Tk.Button(entry_frame, text = 'refresh', font = self.font, command = self.refresh).pack(side = 'left', padx = 4)
        self.text = ScrolledText(self, width = 84, height = 28, font = ('Menlo', 11))
        self.text.pack(fill = 'both', expand = True, padx = 6, pady = 6)
        self.text.configure(state = Tk.DISABLED)

        sample_frame = Tk.Frame(self)
        sample_frame.pack(fill = 'x', padx = 6, pady = (0, 6))
        Tk.OptionMenu(sample_frame, self.dataset_kind_var, 'Search2', 'Search3').pack(side = 'left')
        Tk.OptionMenu(sample_frame, self.sample_selector_var, 'perfect_key', 'top_group').pack(side = 'left', padx = 4)
        Tk.Entry(sample_frame, textvariable = self.sample_key_var, font = self.font).pack(side = 'left', fill = 'x', expand = True, padx = 4)
        Tk.Button(sample_frame, text = 'open sample', font = self.font, command = self.open_sample).pack(side = 'left')
        Tk.Button(sample_frame, text = 'replay sample', font = self.font, command = self.replay_sample).pack(side = 'left', padx = 4)

    def refresh(self):
        ai_index = self.frame.param_manager.selected_index(self.ai_index_var.get(), default = self.frame.AI_idx)
        content = self.frame.search_data_manager.dataset_summary_text(ai_index)
        self.text.configure(state = Tk.NORMAL)
        self.text.delete('1.0', Tk.END)
        self.text.insert(Tk.END, content)
        self.text.configure(state = Tk.DISABLED)

    def open_sample(self):
        ai_index = self.frame.param_manager.selected_index(self.ai_index_var.get(), default = self.frame.AI_idx)
        selector_value = self.sample_key_var.get().strip()
        selector_kind = self.sample_selector_var.get()
        if selector_value == '':
            return
        content = self.frame.search_data_manager.representative_sample_text(
            ai_index,
            self.dataset_kind_var.get(),
            selector_value,
            selector_kind,
        )
        self.text.configure(state = Tk.NORMAL)
        self.text.delete('1.0', Tk.END)
        if content is None:
            self.text.insert(Tk.END, f'No sample found for {selector_kind}={selector_value} in {self.dataset_kind_var.get()}.\n')
        else:
            self.text.insert(Tk.END, content)
            self.frame.append_log(
                f'dataset sample: ai={ai_index} kind={self.dataset_kind_var.get()} {selector_kind}={selector_value}'
            )
        self.text.configure(state = Tk.DISABLED)

    def replay_sample(self):
        ai_index = self.frame.param_manager.selected_index(self.ai_index_var.get(), default = self.frame.AI_idx)
        selector_value = self.sample_key_var.get().strip()
        selector_kind = self.sample_selector_var.get()
        if selector_value == '':
            return
        sample = self.frame.search_data_manager.representative_sample(
            ai_index,
            self.dataset_kind_var.get(),
            selector_value,
            selector_kind,
        )
        if sample is None:
            self.text.configure(state = Tk.NORMAL)
            self.text.delete('1.0', Tk.END)
            self.text.insert(Tk.END, f'No sample found for {selector_kind}={selector_value} in {self.dataset_kind_var.get()}.\n')
            self.text.configure(state = Tk.DISABLED)
            return
        DatasetSampleReplayDialog(self.frame, ai_index, self.dataset_kind_var.get(), sample, selector_kind, selector_value)


class DatasetSampleReplayDialog(Tk.Toplevel):
    """Replay one dataset sample step by step."""

    def __init__(self, frame, ai_index, dataset_kind, sample, selector_kind = 'perfect_key', selector_value = None):
        Tk.Toplevel.__init__(self, frame)
        self.frame = frame
        self.ai_index = ai_index
        self.dataset_kind = dataset_kind
        self.sample = sample
        self.selector_kind = selector_kind
        self.selector_value = selector_value
        self.step_index = 0
        self.font = ('Century Gothic', 12, 'bold')
        self.title(f'{dataset_kind} replay')
        self.replay_cube = self._build_replay_cube()
        self.viewer = self._build_viewer()
        self.info_var = Tk.StringVar(value = '')
        self._build_widgets()
        self._render_step()

    def _build_replay_cube(self):
        if self.frame.puzzle_type == 'megaminx':
            return MegaminxCube(
                size = self.frame.config.cube_size,
                F2L = self.frame.config.F2L,
                OLL = self.frame.config.OLL,
                Centers = self.frame.config.Centers,
                Edges = self.frame.config.Edges,
                Cross = self.frame.config.Cross,
            )
        if self.frame.puzzle_type == 'master_pyraminx':
            return MasterPyraminxCube(
                size = self.frame.config.cube_size,
                F2L = self.frame.config.F2L,
                OLL = self.frame.config.OLL,
                Centers = self.frame.config.Centers,
                Edges = self.frame.config.Edges,
                Cross = self.frame.config.Cross,
            )
        if self.frame.puzzle_type == 'pyraminx':
            return PyraminxCube(
                size = self.frame.config.cube_size,
                F2L = self.frame.config.F2L,
                OLL = self.frame.config.OLL,
                Centers = self.frame.config.Centers,
                Edges = self.frame.config.Edges,
                Cross = self.frame.config.Cross,
            )
        if self.frame.puzzle_type == 'skewb':
            return SkewbCube(
                size = self.frame.config.cube_size,
                F2L = self.frame.config.F2L,
                OLL = self.frame.config.OLL,
                Centers = self.frame.config.Centers,
                Edges = self.frame.config.Edges,
                Cross = self.frame.config.Cross,
            )
        if self.frame.puzzle_type == 'fto':
            return FtoCube(
                size = self.frame.config.cube_size,
                F2L = self.frame.config.F2L,
                OLL = self.frame.config.OLL,
                Centers = self.frame.config.Centers,
                Edges = self.frame.config.Edges,
                Cross = self.frame.config.Cross,
            )
        if self.frame.puzzle_type == 'cto':
            return CtoCube(
                size = self.frame.config.cube_size,
                F2L = self.frame.config.F2L,
                OLL = self.frame.config.OLL,
                Centers = self.frame.config.Centers,
                Edges = self.frame.config.Edges,
                Cross = self.frame.config.Cross,
            )
        return Rubiks_3(
            size = self.frame.config.cube_size,
            F2L = self.frame.config.F2L,
            OLL = self.frame.config.OLL,
            Centers = self.frame.config.Centers,
            Edges = self.frame.config.Edges,
            Cross = self.frame.config.Cross,
        )

    def _build_viewer(self):
        if self.frame.puzzle_type == 'megaminx':
            return MegaminxStateViewer(self, mini_mode = True)
        if self.frame.puzzle_type in ('pyraminx', 'master_pyraminx'):
            return PyraminxStateViewer(self, self.frame.cube_size, mini_mode = True)
        if self.frame.puzzle_type == 'skewb':
            return SkewbStateViewer(self, mini_mode = True)
        if self.frame.puzzle_type == 'fto':
            return FtoStateViewer(self, mini_mode = True)
        if self.frame.puzzle_type == 'cto':
            return CtoStateViewer(self, mini_mode = True)
        return StateViewer(self, self.frame.cube_size, mini_mode = True)

    def _build_widgets(self):
        top = Tk.Frame(self)
        top.pack(fill = 'x', padx = 6, pady = 6)
        Tk.Button(top, text = '<<', font = self.font, command = self._rewind).pack(side = 'left')
        Tk.Button(top, text = '<', font = self.font, command = self._prev_step).pack(side = 'left', padx = 4)
        Tk.Button(top, text = '>', font = self.font, command = self._next_step).pack(side = 'left')
        Tk.Button(top, text = '>>', font = self.font, command = self._last_step).pack(side = 'left', padx = 4)
        Tk.Label(top, textvariable = self.info_var, font = self.font, justify = Tk.LEFT).pack(side = 'left', padx = 12)

        self.viewer.pack(padx = 6, pady = 6)
        self.text = ScrolledText(self, width = 72, height = 12, font = ('Menlo', 11))
        self.text.pack(fill = 'both', expand = True, padx = 6, pady = 6)
        self.text.configure(state = Tk.DISABLED)

    def _rewind(self):
        self.step_index = 0
        self._render_step()

    def _prev_step(self):
        if self.step_index > 0:
            self.step_index -= 1
            self._render_step()

    def _next_step(self):
        if self.step_index < len(self.sample.moves):
            self.step_index += 1
            self._render_step()

    def _last_step(self):
        self.step_index = len(self.sample.moves)
        self._render_step()

    def _render_step(self):
        self.replay_cube.reset()
        self.replay_cube.scramble(0, self.sample.scramble)
        applied_moves = tuple(self.sample.moves[:self.step_index])
        for move in applied_moves:
            self.replay_cube.make_move(move)
        self.viewer.set_color(self.replay_cube.state)
        self._update_info(applied_moves)
        self._update_text(applied_moves)

    def _update_info(self, applied_moves):
        total_steps = len(self.sample.moves)
        next_move = ''
        if self.step_index < total_steps:
            next_move = self.frame.display_move_sequence((self.sample.moves[self.step_index],))[0]
        self.info_var.set(
            f'step {self.step_index}/{total_steps}  next={next_move}  key={getattr(self.sample, "perfect_key", None)}'
        )

    def _update_text(self, applied_moves):
        ai = self.frame.AIs[self.ai_index]
        x = self.replay_cube.makedata().reshape(-1, 1)
        policy = ai.predict(x, policy = True, value = False).reshape(-1)
        value = float(ai.predict(x, policy = False, value = True)[0][0])
        top_indices = np.argsort(policy)[-5:][::-1]
        top_moves = [
            f"{self.frame.display_move_sequence((self.frame.move_keys[index],))[0]}: {float(policy[index]):.4f}"
            for index in top_indices
        ]
        reward_line = self._step_array_line('reward', getattr(self.sample, 'rewards', None), self.step_index - 1)
        target_line = self._step_array_line('value_target', getattr(self.sample, 'value_targets', None), self.step_index - 1)
        trace_line = self._step_array_line('value_trace', getattr(self.sample, 'value_trace', None), self.step_index)
        trace_raw_line = self._step_array_line('value_trace_raw', getattr(self.sample, 'value_trace_raw', None), self.step_index)
        lines = [
            f'{self.dataset_kind} replay',
            f'selector: {self.selector_kind}={self.selector_value}',
            f'AI: {self.ai_index}',
            f'perfect_key: {getattr(self.sample, "perfect_key", None)}',
            f'top_group: {getattr(self.sample, "top_group", None)}',
            f'step: {self.step_index}/{len(self.sample.moves)}',
            f'value: {value:.6f}',
            reward_line,
            target_line,
            trace_line,
            trace_raw_line,
            f'applied moves: {self.frame.display_move_sequence(applied_moves)}',
            f'remaining moves: {self.frame.display_move_sequence(self.sample.moves[self.step_index:])}',
            'top policy:',
        ] + [f'  {line}' for line in top_moves]
        self.text.configure(state = Tk.NORMAL)
        self.text.delete('1.0', Tk.END)
        self.text.insert(Tk.END, '\n'.join(lines))
        self.text.configure(state = Tk.DISABLED)

    def _step_array_line(self, label, values, index):
        if values is None:
            return f'{label}: n/a'
        if len(values) == 0:
            return f'{label}: []'
        if index < 0 or index >= len(values):
            return f'{label}: out-of-range ({index})'
        value = values[index]
        if isinstance(value, (float, np.floating)):
            return f'{label}: step[{index}]={float(value):.6f}'
        return f'{label}: step[{index}]={value}'
