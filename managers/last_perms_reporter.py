"""Last-perms reporting helpers."""

from __future__ import annotations

import numpy as np
from heapdict import heapdict

from core.myperm_keys import resolve_myperm_key

class LastPermsReporter:
    """last_permsや探索済み手順の長さを比較・表示する診断処理を担当する。"""

    def __init__(self, frame):
        self.frame = frame

    def lpk(self):
        """last_permsの最短手数を既存mypermsと比較して表示する。"""
        length_dict,set_dict,value_dict = self._last_perm_lengths()
        self._print_length_comparison(length_dict,set_dict,value_dict)

    def lp_show(self, key, N = None):
        """指定keyのlast_permsから、指定手数または最短手数の手順を表示する。"""
        if N == None:
            N = self._minimum_length(key)
        hit_count = 0
        for moves in self.frame.last_perms[key]:
            if len(moves) == N:
                display_moves = self.frame.cube.format_moves(moves) if hasattr(self.frame.cube, 'format_moves') else moves
                print(display_moves)
                hit_count += 1
        self.frame.append_log(f'lp_show: key={self._display_group_name(key)} len={N} hits={hit_count}')

    def show_counter(self, N):
        """cube.counter[N]に記録された手順と回数を優先度順に表示する。"""
        counter_dict = self._counter_heap(N)
        while len(counter_dict) > 0:
            moves,count = counter_dict.popitem()
            display_moves = self.frame.cube.format_moves(moves) if hasattr(self.frame.cube, 'format_moves') else moves
            print(display_moves,count)

    def myfunc(self):
        """my_scrambles2に登録された各keyの手順数を深さごとに表示する。"""
        scramble_table = self.frame.cube.my_scrambles2
        for key in scramble_table[0].keys():
            length_limit = min(21,len(scramble_table))
            lengths = np.zeros(length_limit,dtype = 'i')
            for index in range(length_limit):
                lengths[index] = len(scramble_table[index][key])

            print(key,lengths)

    def myfunc2(self, N = 0):
        """my_scrambles2[N]の各手順について、先頭手と末尾手だけを表示する。"""
        for key in self.frame.cube.my_scrambles2[N].keys():
            for moves in self.frame.cube.my_scrambles2[N][key]:
                start_move = self.frame.cube.format_move(moves[0]) if hasattr(self.frame.cube, 'format_move') else moves[0]
                end_move = self.frame.cube.format_move(moves[-1]) if hasattr(self.frame.cube, 'format_move') else moves[-1]
                print(start_move,end_move)

    def _last_perm_lengths(self):
        """last_permsからkeyごとの全手数リストと最短手数heapを作る。"""
        length_dict = {}
        value_dict = heapdict()
        set_dict = {}
        for key in sorted(self.frame.last_perms.keys()):
            lengths = [len(moves) for moves in self.frame.last_perms[key]]
            set_dict[key] = lengths
            length_dict[key] = min(lengths)
            value_dict[key] = -min(lengths) * self.frame.last_perms_changed_number[key]
        return length_dict,set_dict,value_dict

    def _print_length_comparison(self, length_dict, set_dict,value_dict):
        """最短手数heapを取り出しながら、keyごとの比較結果を表示する。"""
        print("=======Here are the last perms=======")
        while len(value_dict) > 0:
            key,value = value_dict.popitem()
            length = length_dict[key]
            self._print_key_length_comparison(key,length,set_dict[key],-value)
        
        print("=======Here are the last perms=======")

    def _print_key_length_comparison(self, key, length, lengths,value):
        """1つのkeyについて、last_perms側と登録済みmyperms側の手数を比較表示する。"""
        display_key = self._display_group_name(key)
        registered_length = self._registered_myperm_length(key)
        if registered_length is None:
            print(value,length,display_key,set(lengths))
            return

        current_length = length
        marker = self._comparison_marker(current_length,registered_length)
        if marker == '':
            print(value,current_length,display_key,registered_length,lengths)
        else:
            print(value,current_length,display_key,registered_length,marker,lengths)

    def _registered_myperm_length(self, base_key):
        """登録済み myperm の transform 0 長を puzzle ごとに解決する。"""
        myperms_key = resolve_myperm_key(self.frame.cube, base_key)
        if myperms_key is not None:
            return len(self.frame.cube.myperms[myperms_key])

        legacy_key = self._legacy_transform_zero_key(base_key)
        if legacy_key is None:
            return None
        return len(self.frame.cube.myperms[legacy_key])

    def _legacy_transform_zero_key(self, base_key):
        """文字列 key の myperms から transform 0 を探す。"""
        for key in self.frame.cube.myperms.keys():
            if not isinstance(key, str) or not key.startswith(base_key):
                continue
            suffix = key[len(base_key):]
            if suffix.isdigit() and int(suffix) == 0:
                return key
        return None

    def _display_group_name(self, key):
        """short group key は long version に変換して表示する。"""
        if hasattr(self.frame.cube, '_group_name_map'):
            return self.frame.cube._group_name_map().get(key, key)
        return key

    def _comparison_marker(self, current_length, registered_length):
        """手数比較の結果を表示用マーカーに変換する。"""
        if current_length < registered_length:
            return '<-----------------'
        if current_length == registered_length:
            return '================='
        return ''

    def _minimum_length(self, key):
        """指定keyに対するlast_perms内の最短手数を返す。"""
        return min([len(moves) for moves in self.frame.last_perms[key]])

    def _counter_heap(self, N):
        """counterの内容を値順に取り出せるheapdictへ詰め替える。"""
        counter_dict = heapdict()
        for moves in self.frame.cube.counter[N]:
            counter_dict[moves] = self.frame.cube.counter[N][moves]
        return counter_dict
