"""Myperm management helpers."""

from __future__ import annotations

from functools import reduce
import tkinter as Tk

from core.myperm_keys import myperm_base_key, resolve_myperm_key
from ui.dialogs import MakeMypermOkButton

class MyPermManager:
    """mypermの検索・適用・色キー変換を担当する。"""

    def __init__(self, frame):
        self.frame = frame

    def myperms_col_key(self, moves):
        """手順を色配置キーへ変換し、登録済みmyperm色キーがあれば返す。"""
        self.frame.cube.reset()
        for move in self.frame.cube.invert_moves(moves):
            self.frame.cube.make_move(move)

        state_key = reduce(lambda left,right: left + right,self.frame.cube.state)
        self.frame.cube.reset()
        if state_key in self.frame.myperms_col:
            return self.frame.myperms_col[state_key]
        return ''

    def search_myperms(self, prefix_moves, suffix_moves, head):
        """先頭条件・末尾条件・head文字列でmypermキーを絞り込む。"""
        if suffix_moves != ():
            return [
                key for key in self.frame.cube.myperms
                if myperm_base_key(key)[:len(head)] == head
                and self.frame.cube.myperms[key][-len(suffix_moves):] == suffix_moves
                and self.frame.cube.myperms[key][:len(prefix_moves)] == prefix_moves
            ]
        return [
            key for key in self.frame.cube.myperms
            if myperm_base_key(key)[:len(head)] == head
            and self.frame.cube.myperms[key][:len(prefix_moves)] == prefix_moves
        ]

    def open_apply_dialog(self):
        """myperm名を入力して適用するための小さなダイアログを開く。"""
        dialog = Tk.Toplevel(self.frame)
        dialog.title('make myperm')
        entry = Tk.Entry(master = dialog,width = 20)
        entry.grid(row = 0,column = 0)
        button = MakeMypermOkButton(dialog,self.frame,entry)
        button.grid(row = 1,column = 0)

    def _resolve_myperm_key(self, key):
        """Resolve text input or legacy key text to an actual myperm key."""
        return resolve_myperm_key(self.frame.cube, key)

    def apply_named_myperm(self, myperm_key):
        """指定されたmyperm名の手順を実行してStateViewerを更新する。"""
        resolved_key = self._resolve_myperm_key(myperm_key)
        if resolved_key is None:
            return
        for move in self.frame.cube.myperms[resolved_key]:
            self.frame.cube.make_move(move)
        display_moves = self.frame.cube.format_moves(self.frame.cube.myperms[resolved_key]) if hasattr(self.frame.cube, 'format_moves') else self.frame.cube.myperms[resolved_key]
        print(display_moves)
        self.frame.set_color(self.frame.cube.state)
