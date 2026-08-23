"""Viewer widgets for cube state, moves, and probabilities."""

import tkinter as Tk
from tkinter import scrolledtext

import numpy as np

from core.cube_constants import R_Nums, inside_size, outside_size


class LogViewer(Tk.Frame):
    """学習ログなどの短いテキストを GUI 上に蓄積表示する。"""

    def __init__(self, master, width = 64, height = 12, line_limit = 1200):
        Tk.Frame.__init__(self, master, relief = Tk.RIDGE, bd = 4, bg = '#303030')
        self.line_limit = max(1,int(line_limit))
        self.line_count = 0
        self.text = scrolledtext.ScrolledText(
            self,
            width = width,
            height = height,
            wrap = Tk.WORD,
            bg = '#101010',
            fg = '#E8E8E8',
            insertbackground = '#E8E8E8',
            relief = Tk.FLAT,
            font = ('Menlo', 10),
        )
        self.text.pack(fill = 'both', expand = True)
        self.text.configure(state = Tk.DISABLED)

    def append_line(self, message):
        """末尾へ 1 行追加し、自動スクロールする。"""
        self.text.configure(state = Tk.NORMAL)
        self.text.insert(Tk.END, str(message) + '\n')
        self.line_count += 1
        self._trim_old_lines()
        self.text.see(Tk.END)
        self.text.configure(state = Tk.DISABLED)

    def _trim_old_lines(self):
        """Keep the Tk text buffer bounded during long solve sessions."""
        excess = self.line_count - self.line_limit
        if excess <= 0:
            return
        self.text.delete('1.0', f'{excess + 1}.0')
        self.line_count -= excess

class SuccessViewer(Tk.Frame):
    """AIごとの成功数と、直近のソルブ結果を表示する。"""

    def __init__(self, master, ai_count):
        Tk.Frame.__init__(self,master,relief = Tk.RIDGE,bd = 4,bg = '#303030')
        self.ai_count = ai_count
        self.history = []
        self.history_limit = 200
        self.history_columns = 40
        self.history_block = 4
        self.font = ('Futura',9,'bold')
        self._build_widgets()

    def _build_widgets(self):
        """成功数・現在回数・履歴表示用のWidgetを作る。"""
        self.title_label = Tk.Label(self,text = 'Success',font = self.font,fg = '#F0F0F0',bg = '#303030')
        self.title_label.grid(row = 0,column = 0,sticky = 'w')
        self.current_label = Tk.Label(self,text = '',font = self.font,fg = '#F0F0F0',bg = '#303030')
        self.current_label.grid(row = 0,column = 1,sticky = 'w')
        self.total_label = Tk.Label(self,text = '',font = self.font,fg = '#F0F0F0',bg = '#303030')
        self.total_label.grid(row = 0,column = 2,sticky = 'w')
        self.ai_label = Tk.Label(self,text = '',font = self.font,fg = '#F0F0F0',bg = '#303030',anchor = 'w',justify = Tk.LEFT)
        self.ai_label.grid(row = 1,column = 0,columnspan = 3,sticky = 'ew')
        self.history_canvas = Tk.Canvas(self,width = 300,height = 28,bg = '#202020',highlightthickness = 0)
        self.history_canvas.grid(row = 2,column = 0,columnspan = 3,sticky = 'ew')
        for column_index in range(3):
            self.grid_columnconfigure(column_index, weight = 1)

    def put_summary(self, success_counts, solve_index, ai_index):
        """成功数配列と現在のソルブ位置を表示する。"""
        self._update_labels(success_counts,solve_index,ai_index,None)
        self._draw_history()

    def put_result(self, success_counts, solve_index, ai_index, succeeded):
        """1回分のソルブ結果を履歴へ追加し、成功数表示を更新する。"""
        self.history.append((solve_index,ai_index,bool(succeeded)))
        if len(self.history) > self.history_limit:
            self.history = self.history[-self.history_limit:]
        self._update_labels(success_counts,solve_index,ai_index,succeeded)
        self._draw_history()

    def _update_labels(self, success_counts, solve_index, ai_index, succeeded):
        """現在回数・直近結果・AIごとの成功数をLabelへ反映する。"""
        result_text = self._result_text(succeeded)
        self.current_label.configure(text = 'N: ' + str(solve_index) + '  AI: ' + str(ai_index) + result_text)
        self.total_label.configure(text = 'total: ' + str(int(np.sum(success_counts))))
        self.ai_label.configure(text = self._success_counts_text(success_counts))

    def _result_text(self, succeeded):
        """直近結果を表示用文字列へ変換する。"""
        if succeeded is None:
            return ''
        if succeeded:
            return '  OK'
        return '  NG'

    def _success_counts_text(self, success_counts):
        """AIごとの成功数を横並びの短い文字列にする。"""
        parts = []
        for index,count in enumerate(success_counts):
            parts.append(str(index) + ':' + str(int(count)))
        return '  '.join(parts[:10]) + '\n' + '  '.join(parts[10:])

    def _draw_history(self):
        """直近ソルブ履歴を色付きの小さいブロックで描画する。"""
        self.history_canvas.delete('history')
        block_size = self.history_block
        margin = 4
        for index,result in enumerate(self.history[-self.history_limit:]):
            column_index = index % self.history_columns
            row_index = index // self.history_columns
            x0 = margin + column_index * (block_size + 2)
            y0 = margin + row_index * (block_size + 2)
            x1 = x0 + block_size
            y1 = y0 + block_size
            color = self._history_color(result[2])
            self.history_canvas.create_rectangle(x0,y0,x1,y1,fill = color,outline = '#101010',tags = 'history')

    def _history_color(self, succeeded):
        """成功/失敗を履歴ブロック用の色に変換する。"""
        if succeeded:
            return Red
        return Blue


class StateViewer(Tk.Canvas):
    def __init__(self,master,cube_size = 3,mini_mode = False):
        self.cube_size = cube_size
        self.surface_num = cube_size ** 2
        self._init_size_parameters(mini_mode)
        self.coordinates = self._build_coordinates()
        self.r_size = self.coordinates[-2 - self.cube_size] + self.blank
        self.c_size = self.coordinates[-1] + self.blank
        self._init_color_maps()
        Tk.Canvas.__init__(self,master,relief = Tk.FLAT , bd = 0,width = self.c_size,height = self.r_size,bg = '#5F5F5F')
        self._init_surface_positions()

    def _init_size_parameters(self, mini_mode):
        if mini_mode:
            self.margin = 0.25
            self.outside_size = 0.5 * outside_size[self.cube_size] + self.margin
            self.inside_size = 0.5 * inside_size[self.cube_size] + self.margin
            self.blank = 5.5
            self.corner_radius = 2
            self.bd_width = 5
        else:
            self.margin = 0.5
            self.outside_size = outside_size[self.cube_size] + 2 * self.margin
            self.inside_size = inside_size[self.cube_size] + 2 * self.margin
            self.blank = 11
            self.corner_radius = 4
            self.bd_width = 10

    def _build_coordinates(self):
        coordinates = np.zeros(4 * self.cube_size + 5,dtype = 'i')
        for index in range(1,4 * self.cube_size + 5):
            coordinates[index] = coordinates[index - 1] + self._coordinate_step(index)
        return coordinates

    def _coordinate_step(self, index):
        if index % (self.cube_size + 1) == 1:
            return self.blank
        if index % (self.cube_size + 1) == 2 or index % (self.cube_size + 1) == 0:
            return self.outside_size
        return self.inside_size

    def _init_color_maps(self):
        self.color = {'R':'#7F0000','W':'#BFBFBF','B':'#0000BF','G':'#007F3F','Y':'#DFDF00','O':'#FF7F00','':'#7F7F7F','X':'#7F7F7F'}
        self.bd_color = {'R':'#5F0000','W':'#9F9F9F','B':'#00009F','G':'#005F2F','Y':'#BFBF00','O':'#BF5F00','':'#5F5F5F','X':'#5F5F5F'}

    def _init_surface_positions(self):
        self.C = np.zeros(6 * self.surface_num,dtype = 'i')
        self.R = np.zeros(6 * self.surface_num,dtype = 'i')
        args = np.argsort(R_Nums[self.cube_size].reshape(-1))
        for index in range(self.surface_num):
            self._set_surface_position(index, args[index])

    def _set_surface_position(self, index, sorted_index):
        cube_stride = self.cube_size + 1
        row_offset = sorted_index // self.cube_size + 1
        col_offset = sorted_index % self.cube_size + 1

        self.R[index] = cube_stride + row_offset
        self.C[index] = cube_stride + col_offset
        self.R[index + self.surface_num] = 2 * cube_stride - row_offset
        self.C[index + self.surface_num] = 4 * cube_stride - col_offset
        self.R[index + 2 * self.surface_num] = 3 * cube_stride - row_offset
        self.C[index + 2 * self.surface_num] = 2 * cube_stride - col_offset
        self.R[index + 3 * self.surface_num] = row_offset
        self.C[index + 3 * self.surface_num] = cube_stride + col_offset
        self.R[index + 4 * self.surface_num] = 2 * cube_stride - col_offset
        self.C[index + 4 * self.surface_num] = row_offset
        self.R[index + 5 * self.surface_num] = cube_stride + col_offset
        self.C[index + 5 * self.surface_num] = 3 * cube_stride - row_offset

    def set_color(self,S):
        self.delete('squares')
        self._draw_face_backgrounds()
        for index in range(6 * self.surface_num):
            self._draw_sticker(S, index)

    def _draw_face_backgrounds(self):
        cube_stride = self.cube_size + 1
        face_offsets = [(1,0),(0,1),(1,1),(2,1),(3,1),(1,2)]
        for row_offset, col_offset in face_offsets:
            self.create_rectangle(
                self.coordinates[1 + cube_stride * row_offset],
                self.coordinates[1 + cube_stride * col_offset],
                self.coordinates[cube_stride * (row_offset + 1)],
                self.coordinates[cube_stride * (col_offset + 1)],
                fill = '#000000',
                outline = '#000000',
                width = self.bd_width,
                tags = 'squares',
            )

    def _draw_sticker(self, state_text, index):
        c0, r0, c1, r1 = self._sticker_bounds(index)
        points = self._rounded_rectangle_points(c0, r0, c1, r1)
        sticker_color = state_text[index]
        self.create_polygon(points,fill = self.color[sticker_color],outline = self.bd_color[sticker_color],smooth = True,tags = 'squares')

    def _sticker_bounds(self, index):
        c0 = self.coordinates[self.C[index]]
        r0 = self.coordinates[self.R[index]]
        c1 = self.coordinates[self.C[index] + 1]
        r1 = self.coordinates[self.R[index] + 1]
        return c0, r0, c1, r1

    def _rounded_rectangle_points(self, c0, r0, c1, r1):
        return [
            c0 + self.corner_radius * 0.3,r0 + self.corner_radius * 0.3,
            c0,r0 + self.corner_radius,
            c0,r1 - self.corner_radius,
            c0 + self.corner_radius * 0.3,r1 - self.corner_radius * 0.3,
            c0 + self.corner_radius,r1,
            c1 - self.corner_radius,r1,
            c1 - self.corner_radius * 0.3,r1 - self.corner_radius * 0.3,
            c1,r1 - self.corner_radius,
            c1,r0 + self.corner_radius,
            c1 - self.corner_radius * 0.3,r0 + self.corner_radius * 0.3,
            c1 - self.corner_radius,r0,
            c0 + self.corner_radius,r0,
        ]


class MoveViewer(Tk.Canvas):
    def __init__(self,master):
        self.fixed_r_size = 400
        self.r_size = self.fixed_r_size
        self.c_size = 700
        self.text_color = '#FFFFFF'
        self.move_color = '#000000'
        self.font = 'Futura'
        self.font_size = 10
        self.words_in_a_row = 20
        self.c_start = 150
        self.r_start = 100
        self.c_start_cube_state = 100
        self.r_start_cube_state = 20
        self.c_dist = 20
        self.r_dist = 13
        Tk.Canvas.__init__(self,master,relief = Tk.RAISED, bd = 4,width = self.c_size,height = self.r_size,bg = '#000000')
        self.value_start = 600
        self.value_width = 95
        self.key_width = 110
        self.min_move_columns = 4

    def set_str(
        self,
        scramble_state,
        move_rows,
        key_labels,
        root_values,
        leaf_values,
        step_values_per_row,
        solved_count,
        solve_count,
        search_mode,
        search2_value_loss_type = None,
    ):
        self.delete('text')
        self.delete('header')
        self.delete('squares')
        self._configure_layout(scramble_state, move_rows, key_labels)
        self._draw_cube_state(scramble_state, solved_count, solve_count)
        row_index = self._header_row_index(scramble_state)
        self._draw_header(row_index)
        row_index += 1
        for move_index in range(1,len(move_rows)):
            row_index = self._draw_log_row(
                move_index,
                move_rows,
                key_labels,
                root_values,
                leaf_values,
                step_values_per_row,
                search_mode,
                search2_value_loss_type,
                row_index,
            )

    def _draw_cube_state(self, state_text, scramble_num, total_num):
        for index, sticker in enumerate(state_text):
            self.create_text(
                self.c_start_cube_state + self.state_c_dist * (index % self.state_words_in_a_row),
                self.r_start_cube_state + self.r_dist * (index // self.state_words_in_a_row),
                text = sticker,
                tags = 'text',
                fill = self.text_color,
                font = (self.font,self.font_size,'bold'),
            )
        self.create_text(
            50,
            20,
            text = str(scramble_num) + '/' + str(total_num),
            tags = 'text',
            fill = self.text_color,
            font = (self.font,self.font_size,'bold'),
        )

    def _header_row_index(self, state_text):
        return max((len(state_text) - 1) // self.state_words_in_a_row - 4,0)

    def _draw_header(self, row_index):
        header_y = 80 + self.r_dist * row_index
        self.create_text(self.key_width * 0.5,header_y,text = 'Key',tags = 'header',fill = self.text_color,font = (self.font,self.font_size,'bold'))
        self.create_text(self.value_start,header_y,text = 'Value',tags = 'header',fill = self.text_color,font = (self.font,self.font_size,'bold'))
        self.create_text(self.c_start,header_y,text = 'Moves',tags = 'header',fill = self.text_color,font = (self.font,self.font_size,'bold'),anchor = 'w')

    def _draw_log_row(self, move_index, move_rows, key_labels, root_values, leaf_values, step_values, search_mode, search2_value_loss_type, row_index):
        row_y = 100 + self.r_dist * row_index
        self.create_text(self.key_width * 0.5,row_y,text = key_labels[move_index],tags = 'text',fill = self.text_color,font = (self.font,self.font_size,'bold'))
        self.create_text(self.value_start,row_y,text = self._format_value_text(root_values[move_index], leaf_values[move_index], step_values[move_index]),tags = 'text',fill = self.text_color,font = (self.font,self.font_size,'bold'))

        for step_index, move_label in enumerate(move_rows[move_index]):
            square_color = self._move_square_color(step_values[move_index], step_index, search_mode, search2_value_loss_type)
            self.create_rectangle(
                self.c_start + self.c_dist * (step_index % self.words_in_a_row - 0.475),
                self.r_start + self.r_dist * (row_index + step_index // self.words_in_a_row - 0.4),
                self.c_start + self.c_dist * (step_index % self.words_in_a_row + 0.475),
                self.r_start + self.r_dist * (row_index + step_index // self.words_in_a_row + 0.4),
                fill = square_color,
                width = 0,
                tags = 'squares'
            )
            self.create_text(
                self.c_start + self.c_dist * (step_index % self.words_in_a_row),
                self.r_start + self.r_dist * (row_index + step_index // self.words_in_a_row),
                text = move_label,
                tags = 'text',
                fill = self.move_color,
                font = (self.font,self.font_size,'bold'),
            )

        return row_index + self._row_height(move_rows[move_index])

    def _format_value_text(self, root_value, leaf_value, step_values):
        if len(step_values) != 0:
            return f'{root_value:.3f} -> {leaf_value:.3f}'
        return f'{leaf_value:.3f}'

    def _move_square_color(self, step_values, step_index, search_mode, search2_value_loss_type = None):
        if len(step_values) == 0:
            return self.text_color
        step_value = step_values[step_index + 1]
        if search_mode == 'search2':
            return set_color2(step_value)
        if search_mode in ('search3', 'transformer'):
            return set_color3(step_value)
        return self.text_color

    def _row_height(self, move_row):
        if len(move_row) == 0:
            return 1
        return (len(move_row) - 1) // self.words_in_a_row + 1

    def _configure_layout(self, scramble_state, move_rows, key_labels):
        labels = [str(move) for row in move_rows for move in row]
        max_move_len = max([len(label) for label in labels] + [1])
        max_key_len = max([len(str(label)) for label in key_labels] + [3])
        self.c_dist = max(20, min(72, max_move_len * 7 + 10))
        self.state_c_dist = max(12, min(28, max([len(str(sticker)) for sticker in scramble_state] + [1]) * 7 + 5))
        self.key_width = max(90, min(180, max_key_len * 7 + 24))
        self.c_start = self.key_width + 35
        self.value_start = self.c_size - self.value_width * 0.5
        move_area_width = max(self.c_dist * self.min_move_columns, self.value_start - self.value_width * 0.5 - self.c_start - 8)
        self.words_in_a_row = max(self.min_move_columns, int(move_area_width // self.c_dist))
        state_area_width = max(self.state_c_dist, self.c_size - self.c_start_cube_state - 20)
        self.state_words_in_a_row = max(1, int(state_area_width // self.state_c_dist))
        total_rows = self._header_row_index(scramble_state) + 2
        for row in move_rows[1:]:
            total_rows += self._row_height(row)
        required_height = max(260, 110 + self.r_dist * total_rows)
        self.r_size = self.fixed_r_size
        self.configure(height = self.fixed_r_size, scrollregion = (0, 0, self.c_size, required_height))


class ProbViewer(Tk.Canvas):
    def __init__(self,master,move_keys):
        self.r_size = 120
        self.c_size = 350
        self.font = ('Futura',8,'bold')
        self.column_width = 55
        self.row_height = 9
        self.columns = 6

        Tk.Canvas.__init__(self,master,relief = Tk.RAISED, bd = 0,width = self.c_size,height = self.r_size,bg = '#000000')
        self.move_keys = move_keys
        self.move_len = len(self.move_keys)


    def put_val(self,W):
        self._clear_probability_text()
        rounded_probabilities = self._rounded_percentages(W)
        for move_index, move_key in enumerate(self.move_keys):
            self._draw_probability_entry(move_index, move_key, rounded_probabilities[move_index])

    def _clear_probability_text(self):
        self.delete('text')
        self.delete('move')

    def _rounded_percentages(self, probabilities):
        return np.round(probabilities * 100,2)

    def _draw_probability_entry(self, move_index, move_key, probability_value):
        column_index, row_index = self._entry_grid_position(move_index)
        label_x, value_x, row_y = self._entry_coordinates(column_index, row_index)
        self.create_text(label_x,row_y,font = self.font,text = move_key + ':',tags = 'move',fill = '#FFFFFF')
        self.create_text(value_x,row_y,font = self.font,text = str(probability_value) + '%',tags = 'text',fill = set_color(probability_value))

    def _entry_grid_position(self, move_index):
        return move_index % self.columns, move_index // self.columns

    def _entry_coordinates(self, column_index, row_index):
        label_x = (column_index + 0.5) * self.column_width
        value_x = label_x + self.column_width * 0.5
        row_y = self.row_height * (row_index + 1)
        return label_x, value_x, row_y


# Backward-compatible aliases during viewer rename migration.
State_viewer = StateViewer
Move_viewer = MoveViewer
Prob_viewer = ProbViewer


Red = '#7F0000'
Orange = '#FF7F00'
Yellow = '#DFDF00'
Lime = '#7FFF00'
Green = '#005F2F'
Aqua = '#007FFF'
Blue = '#0000BF'
Purple = '#3F007F'
Magenta = '#BF005F'
Silver = '#7F7F7F'
LightSilver = '#BFBFBF'
White = '#FFFFFF'

def _color_from_thresholds(value, thresholds, default_color):
    """閾値表に従って表示色を決める。"""
    for threshold, color in thresholds:
        if value > threshold:
            return color
    return default_color


PROBABILITY_PERCENT_COLORS = [
    (90, Red),
    (70, Orange),
    (50, Yellow),
    (30, Lime),
    (10, Green),
    (7, Aqua),
    (5, Blue),
    (3, Purple),
    (1, Magenta),
    (0.1, Silver),
]

SEARCH2_VALUE_COLORS = [
    (0.0, White),
    (-20.0, Red),
    (-40.0, Orange),
    (-60.0, Yellow),
    (-80.0, Lime),
    (-100.0, Green),
    (-120.0, Aqua),
    (-140.0, Blue),
    (-160.0, Purple),
    (-180.0, Magenta),
    (-200.0, Silver),
]

SEARCH3_VALUE_COLORS = [
    ((0.5) ** (1/4), Red),
    ((0.5) ** (1/2), Orange),
    ((0.5) ** (3/4), Yellow),
    ((0.5) ** (1), Lime),
    ((0.5) ** (5/4), Green),
    ((0.5) ** (3/2), Aqua),
    ((0.5) ** (7/4), Blue),
    ((0.5) ** (2), Purple),
    ((0.5) ** (3), Magenta),
    ((0.5) ** (4), Silver),
]


def color_for_probability_percent(value):
    """ProbViewer の百分率表示用の色を返す。"""
    return _color_from_thresholds(value, PROBABILITY_PERCENT_COLORS, LightSilver)


def color_for_search2_value(value):
    """Search2 の value 差分表示用の色を返す。"""
    return _color_from_thresholds(value, SEARCH2_VALUE_COLORS, LightSilver)


def color_for_search3_value(value):
    """Search3 の value 表示用の色を返す。"""
    return _color_from_thresholds(value, SEARCH3_VALUE_COLORS, LightSilver)


# Backward-compatible aliases during color helper rename migration.
set_color = color_for_probability_percent
set_color2 = color_for_search2_value
set_color3 = color_for_search3_value



class MoveButton(Tk.Button):
    def __init__(self,master,m,cube,font,frame):
        self.m = m
        if hasattr(cube, 'format_move'):
            text = cube.format_move(m)
        else:
            text = m
        Tk.Button.__init__(self,master,text = text,font = font,padx = 1,pady = 1,command = self.make_move)
        self.cube = cube
        self.frame = frame

    def make_move(self):
        self.cube.make_move(self.m)
        self.frame.SV.set_color(self.cube.state)

        

Move_Button = MoveButton
