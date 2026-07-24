"""Solve session state and orchestration helpers."""

from __future__ import annotations

import random
from functools import reduce

import numpy as np
import tkinter as Tk

from core.myperm_effects import MypermEffectAnalyzer
from core.myperm_keys import format_myperm_key, myperm_base_key, myperm_transform_index, resolve_myperm_key
from core.myperm_points import point_representative_transform
from model.search_result import SearchResult, data

PERFECT_VAL = 1.0e+8


def softmax(x):
    """1次元配列をsoftmaxで確率分布へ変換する。"""
    shifted = x - np.max(x)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x)


class SolveSessionState:
    """1回のsolve過程で蓄積されるログ・表示用データをまとめて保持する。"""

    def __init__(self):
        self.s = None
        self.phase = -1
        self.search_TF = False
        self.end_solve = False
        self.last_perfect_key = ''
        self.last_top_group = ''
        self.last_perfect_changed_number = 0
        self.last_simplified_lis = tuple([])
        self.move_lis = []
        self.key_lis = []
        self.val_lis = []
        self.val_lis2 = []
        self.display_move_lis = []
        self.display_key_lis = []
        self.display_root_lis = []
        self.display_val_lis = []
        self.display_val_lis2 = []
        self.search_history = []
        self._search3_training_sample_cache = {}

    def reset_tracking(self):
        """solve中に蓄積した手順・評価値・表示履歴を空に戻す。"""
        self.move_lis.clear()
        self.key_lis.clear()
        self.val_lis.clear()
        self.val_lis2.clear()
        self.display_move_lis.clear()
        self.display_key_lis.clear()
        self.display_root_lis.clear()
        self.display_val_lis.clear()
        self.display_val_lis2.clear()
        self.search_history.clear()
        self._search3_training_sample_cache.clear()

    def reset_session(self):
        """1回のsolveに紐づく状態をまとめて初期化する。"""
        self.reset_tracking()
        self.s = None
        self.phase = -1
        self.search_TF = False
        self.end_solve = False
        self.last_perfect_key = ''
        self.last_top_group = ''
        self.last_perfect_changed_number = 0
        self.last_simplified_lis = tuple([])


class SolveSessionManager:
    """Frameのsolve実行フローと、その途中状態の更新を担当する。"""

    def __init__(self, frame):
        self.frame = frame

    def my_solve(self):
        """1ステップ分のsolve処理を進め、表示更新と終了判定まで行う。"""
        state = self.frame.solve_state
        if self.frame.stop:
            return
        succeeded = False
        self._disable_solve_controls()
        AI = self._get_active_ai()
        if state.phase == -1:
            self._start_new_solve(AI)
        else:
            succeeded = self._advance_solve_step(AI)
            state.phase += 1

        self._update_viewers(AI)
        if state.end_solve:
            self._finalize_solve_step(succeeded)
        self.frame._schedule_next_solve()

    def _disable_solve_controls(self):
        """solve実行中は手動操作ボタンを無効化する。"""
        self.frame.my_solve_button.configure(state = Tk.DISABLED)
        for key in self.frame.move_keys:
            self.frame.movebuttons[key].configure(state = Tk.DISABLED)

    def _get_active_ai(self):
        """現在のAI indexに対応する探索主体を返す。"""
        if self.frame.AI_idx != -1:
            return self.frame.AIs[self.frame.AI_idx]
        return self.frame.myval_AI

    def _start_new_solve(self, AI):
        """新しいスクランブルを用意してsolveセッションを開始する。"""
        state = self.frame.solve_state
        self.frame.cube.reset()
        self._reset_search_engine(AI)
        if self.frame.AI_idx != -1:
            self._scramble_with_ai_settings()
        else:
            self._scramble_with_manual()
        state.phase += 1
        state.search_TF = (self.frame.AI_idx != -1)
        self._reset_solve_tracking()

    def _scramble_with_ai_settings(self):
        """現在のAI設定に従ってスクランブルを生成する。"""
        state = self.frame.solve_state
        scramble_num = self.frame.level[self.frame.AI_idx,self.frame.stage]
        move_count_policy = self._move_count_policy_for_scramble()
        max_attempts = 20
        for _ in range(max_attempts):
            self.frame.cube.reset()
            if self.frame.transform_random:
                transform_N = None
                flip_inside = None
            else:
                transform_N = self.frame.transform_idx[self.frame.AI_idx]
                flip_inside = self.frame.flip_inside_idx[self.frame.AI_idx]

            state.s = self.frame.cube.scramble(
                scramble_num,
                difficult_mode = True,
                scramble_mode = self.frame.scramble_mode[self.frame.stage],
                move_count_policy = move_count_policy,
                transform_N = transform_N,
                flip_inside = flip_inside,
            )
            if not self.frame.cube.is_perfect():
                return

        self._fallback_simple_scramble(scramble_num)

    def _fallback_simple_scramble(self, scramble_num):
        """登録済み scramble が solved に戻り続けたときの最終 fallback。"""
        state = self.frame.solve_state
        self.frame.cube.reset()
        state.s = self.frame.cube.scramble(scramble_num, scramble_mode = None)
        self.frame.append_log(
            f'scramble fallback: ai={self.frame.AI_idx} stage={self.frame.stage} length={len(state.s)}'
        )

    def _move_count_policy_for_scramble(self):
        if self.frame.N % (self.frame.AInum * 2) < self.frame.AInum:
            return 'prefer_frequent'
        return 'prefer_rare'

    def _scramble_with_manual(self):
        """手動指定されたスクランブル列を現在状態へ反映する。"""
        state = self.frame.solve_state
        self.frame.cube.reset()
        state.s = self.frame.cube.scramble(0,Move = self.frame.my_scramble[self.frame.N])

    def _reset_solve_tracking(self):
        """1回のsolveに紐づく履歴と補助状態を初期化する。"""
        state = self.frame.solve_state
        state.reset_tracking()
        state.end_solve = False
        state.last_perfect_key = ''
        state.last_perfect_changed_number = 0
        state.last_simplified_lis = tuple([])

    def _advance_solve_step(self, AI):
        """search段階かgreedy段階かを見て、次の1ステップを進める。"""
        if self.frame.solve_state.search_TF:
            return self._advance_search_step(AI)
        return self._advance_greedy_step(AI)

    def _advance_search_step(self, AI):
        """探索AIで1回探索し、結果を状態と表示用ログへ反映する。"""
        state = self.frame.solve_state
        succeeded = False
        current_state = self.frame.cube.state.copy()
        if len(state.val_lis) == 0:
            self._append_initial_value(AI)

        if self._uses_search3(AI):
            search_result = AI.search(progress_callback = lambda result: self._record_search_attempt_progress(AI, result))
        else:
            search_result = AI.search()
        search_result = self._apply_puzzle_fallback(AI, search_result)
        self._maybe_log_search2_memory(AI)
        self._release_search_step_memory(AI)

        if self._should_fallback_from_search3(AI, search_result):
            if self.frame.search3_progress[self.frame.AI_idx]:
                fallback_scramble = self._current_search_scramble()
                fallback_result = self._advance_to_search3_fallback_state(AI, search_result)
                if fallback_result is not None:
                    self._record_search_history(fallback_result, scramble = fallback_scramble)
                    self._commit_main_progress_to_scramble()
            state.search_TF = False
            return False

        self._record_search_history(search_result)
        reduced_lis = self.frame.cube.reduce(search_result.moves)
        simplified_lis = self.frame.cube.simplify(search_result.moves)
        value_deltas = self._display_value_deltas(search_result)

        self._record_search_result(reduced_lis, search_result, value_deltas)

        if search_result.succeeded:
            succeeded = True
            state.end_solve = True
            simplified_lis = self._store_perfect_key(simplified_lis)
            
        for move in reduced_lis[0]:
            self.frame.cube.make_move(move)
            self._advance_search_engine(AI, move)

        if (self.frame.cube.state == current_state).all():
            self._handle_no_progress_search(AI)

        return succeeded

    def _release_search_step_memory(self, AI):
        """Search2/torchの一時キャッシュをsolve step後に解放する。"""
        if hasattr(AI,'clear_training_cache'):
            AI.clear_training_cache(collect = False)
        if not getattr(AI,'release_search_memory_each_step',False):
            return
        if hasattr(AI,'_mark_torch_params_dirty'):
            AI._mark_torch_params_dirty()
        if hasattr(AI,'_release_torch_training_memory'):
            AI._release_torch_training_memory()

    def _maybe_log_search2_memory(self, AI):
        """Search2 frontierが大きい時だけ診断ログを出す。"""
        frontier_peak = int(getattr(AI,'_last_search2_frontier_peak',0))
        frontier_remaining = int(getattr(AI,'_last_search2_frontier_remaining',0))
        value_count = int(getattr(AI,'_last_search2_value_count',0))
        threshold = max(10000,int(getattr(AI,'search2_max_frontier',30000)) // 2)
        if frontier_peak < threshold and frontier_remaining < threshold:
            return
        self.frame.append_log(
            f'search2 memory: ai={self.frame.AI_idx} '
            f'frontier_peak={frontier_peak} frontier_remaining={frontier_remaining} '
            f'value_count={value_count}'
        )

    def _apply_puzzle_fallback(self, AI, search_result):
        """Puzzle-specific fallback hook for searches that need state-dependent legal moves."""
        if not hasattr(self.frame.cube, 'select_fallback_search_result'):
            return search_result
        fallback_result = self.frame.cube.select_fallback_search_result(AI, search_result)
        if fallback_result is None:
            return search_result
        return fallback_result

    def _record_search_result(self, reduced_lis, search_result, value_deltas, key_label = None):
        """探索結果をmainログへ追加し、必要なら表示ログにも反映する。"""
        state = self.frame.solve_state
        if key_label is None:
            key_label = str(search_result.stats[0]) + '/' + str(search_result.stats[1])

        state.move_lis.append(reduced_lis[0])
        state.key_lis.append(key_label)
        state.val_lis.append(search_result.root_value)
        state.val_lis2.append(self._reduce_value_deltas(reduced_lis[1], value_deltas))
        should_display_result = search_result.search_mode not in ('search3', 'transformer')
        if should_display_result:
            state.display_move_lis.append(reduced_lis[0])
            state.display_key_lis.append(key_label)
            state.display_root_lis.append(search_result.root_value)
            state.display_val_lis.append(search_result.best_value)
            state.display_val_lis2.append(self._reduce_value_deltas(reduced_lis[1], value_deltas))

    def _record_search_display(self, AI, search_result):
        """attempt単位の探索結果をMoveViewer表示用ログへ積む。"""
        state = self.frame.solve_state
        attempt_results = search_result.attempt_results
        if len(attempt_results) == 0:
            attempt_results = [search_result]
        for attempt_index, attempt_result in enumerate(attempt_results,1):
            reduced_lis = self.frame.cube.reduce(attempt_result.moves)
            value_deltas = self._display_value_deltas(attempt_result)
            key_label = str(attempt_result.stats[0]) + '/' + str(attempt_result.stats[1])
            if self._uses_search3(AI):
                current_attempt = attempt_result.attempt_index
                if current_attempt is None:
                    current_attempt = attempt_index
                prefix = 'TF-' if getattr(AI, 'is_transformer', False) else 'S3-'
                key_label = prefix + str(current_attempt) + ':' + key_label
            state.display_move_lis.append(reduced_lis[0])
            state.display_key_lis.append(key_label)
            state.display_root_lis.append(attempt_result.root_value)
            state.display_val_lis.append(attempt_result.best_value)
            state.display_val_lis2.append(self._reduce_value_deltas(reduced_lis[1], value_deltas))

    def _record_search_attempt_progress(self, AI, attempt_result):
        """Search3のattempt結果を表示ログへ反映し、MoveViewerを即時更新する。"""
        self._record_search_display(AI, attempt_result)
        self._refresh_search_attempt_display(
            AI,
            update_state_viewer = self.frame.search3_progress[self.frame.AI_idx],
        )

    def _refresh_search_attempt_display(self, AI, update_state_viewer = True):
        """attempt途中のMoveViewerとStateViewerをその場で描き直す。"""
        state = self.frame.solve_state
        self.frame.MV.set_str(
            self.frame.display_move_sequence(state.s),
            self.frame.display_move_rows(state.display_move_lis),
            state.display_key_lis,
            state.display_root_lis,
            state.display_val_lis,
            state.display_val_lis2,
            self.frame.perf_num[self.frame.stage],
            self.frame.N,
            AI.search_mode,
            getattr(AI,'search2_value_loss_type',None),
        )
        if update_state_viewer:
            self.frame.set_color(AI.cube.state)
        self.frame.update()

    def _display_value_deltas(self, search_result):
        """探索結果を表示用のvalue系列へ変換する。"""
        if search_result.search_mode in ('search3', 'transformer'):
            return search_result.value_trace.copy()
        return [value - self.frame.AIs[self.frame.AI_idx].perfect_val for value in search_result.value_trace]

    def _reduce_value_deltas(self, reduced_indices, value_deltas):
        """簡約後の手順indexに合わせてvalue系列を間引く。"""
        if len(value_deltas) == 0:
            return []
        reduced_values = [value_deltas[0]]
        last_value = value_deltas[-1]
        for reduced_index in reduced_indices:
            value_index = reduced_index + 1
            if value_index < len(value_deltas):
                reduced_values.append(value_deltas[value_index])
            else:
                reduced_values.append(last_value)
        return reduced_values

    def _record_search_history(self, search_result, scramble = None):
        """探索結果をSearch3学習データ化用の履歴へ保存する。"""
        if scramble is None:
            scramble = self._current_search_scramble()
        self.frame.solve_state.search_history.append({
            'scramble': tuple(scramble),
            'search_result': search_result,
            'top_group': self.frame.solve_state.last_top_group,
            'perfect_key': self.frame.solve_state.last_perfect_key,
        })

    def _record_myval_history(self, scramble, root_value, succeeded):
        """greedy段階の結果もSearchResult形式にして履歴へ保存する。"""
        state = self.frame.solve_state
        if len(state.move_lis) == 0:
            return
        best_moves = tuple(state.move_lis[-1])
        if len(best_moves) == 0:
            return
        best_value = state.val_lis[-1]
        search_result = SearchResult(
            succeeded,
            best_moves,
            root_value,
            [root_value,best_value],
            best_value,
            np.array([1,1],dtype = 'i'),
            search_mode = 'myval',
            end_reason = 'solved' if succeeded else 'greedy',
        )
        self.frame.solve_state.search_history.append({
            'scramble': tuple(scramble),
            'search_result': search_result,
            'top_group': self.frame.solve_state.last_top_group,
            'perfect_key': self.frame.solve_state.last_perfect_key,
        })

    def _sync_display_tracking_from_main(self):
        """mainログをそのまま表示ログへコピーする。"""
        state = self.frame.solve_state
        state.display_move_lis[:] = state.move_lis.copy()
        state.display_key_lis[:] = state.key_lis.copy()
        state.display_root_lis[:] = state.val_lis.copy()
        state.display_val_lis[:] = state.val_lis.copy()
        state.display_val_lis2[:] = state.val_lis2.copy()

    def _reset_search_engine(self, AI):
        """Search3を使うAIなら探索木を初期状態に戻す。"""
        if self._uses_search3(AI):
            AI.search3_engine.reset_tree()

    def _advance_search_engine(self, AI, move):
        """Search3のrootを実際に選んだ手へ進める。"""
        if self._uses_search3(AI):
            AI.search3_engine.advance_root(move)

    def _uses_search3(self, AI):
        """このAIがPUCT型の探索を使うか判定する。"""
        return hasattr(AI,'search_mode') and AI.search_mode in ('search3', 'transformer')

    def _should_fallback_from_search3(self, AI, search_result):
        """Search3がbudget終了したときにgreedy側へ落とすか判定する。"""
        return self._uses_search3(AI) and search_result.end_reason in ('budget', 'square1-fallback')

    def _advance_to_search3_fallback_state(self, AI, search_result):
        """Search3 budget 時は、最良評価だった prefix 状態まで進めてから greedy へ落とす。"""
        fallback_result = self._search3_fallback_result(search_result)
        if fallback_result is None or len(fallback_result.moves) == 0:
            return None

        reduced_lis = self.frame.cube.reduce(fallback_result.moves)
        value_deltas = self._display_value_deltas(fallback_result)
        fallback_prefix = 'TFF:' if getattr(self.frame.AIs[self.frame.AI_idx], 'is_transformer', False) else 'S3F:'
        fallback_key_label = fallback_prefix + str(fallback_result.stats[0]) + '/' + str(fallback_result.stats[1])
        self._record_search_result(
            reduced_lis,
            fallback_result,
            value_deltas,
            key_label = fallback_key_label,
        )
        for move in reduced_lis[0]:
            self.frame.cube.make_move(move)
            self._advance_search_engine(AI, move)
        return fallback_result

    def _commit_main_progress_to_scramble(self):
        """main ログ上の進行分を基底 scramble に反映し、以後の greedy 基準にする。"""
        state = self.frame.solve_state
        committed_moves = tuple([])
        for moves in state.move_lis:
            committed_moves += tuple(moves)
        if len(committed_moves) > 0:
            state.s += committed_moves
        state.move_lis.clear()
        state.key_lis.clear()
        state.val_lis.clear()
        state.val_lis2.clear()

    def _search3_fallback_result(self, search_result):
        """Search3 の value trace で最良だった prefix だけを fallback 用結果にする。"""
        best_trace_index = self._best_search3_trace_index(search_result)
        if best_trace_index <= 0:
            return None

        fallback_moves = tuple(search_result.moves[:best_trace_index])
        fallback_value_trace = list(search_result.value_trace[:best_trace_index + 1])
        fallback_raw_trace = list(search_result.value_trace_raw[:best_trace_index + 1])
        fallback_best_value = fallback_value_trace[-1]
        fallback_best_value_raw = fallback_raw_trace[-1]
        return SearchResult(
            False,
            fallback_moves,
            search_result.root_value,
            fallback_value_trace,
            fallback_best_value,
            search_result.stats,
            search_mode = search_result.search_mode,
            end_reason = 'budget-greedy',
            root_value_raw = search_result.root_value_raw,
            value_trace_raw = fallback_raw_trace,
            best_value_raw = fallback_best_value_raw,
        )

    def _best_search3_trace_index(self, search_result):
        """Search3 fallback 用に、trace 上で最良評価だった state の位置を返す。"""
        trace = search_result.value_trace_raw
        if len(trace) == 0:
            return 0
        return int(np.argmax(trace))

    def _current_search_scramble(self):
        """現在地点までに実行した手を含むスクランブル列を作る。"""
        state = self.frame.solve_state
        current_scramble = tuple(state.s)
        for moves in state.move_lis:
            current_scramble += tuple(moves)
        return current_scramble

    def _store_perfect_key(self, simplified_lis):
        """完成局面に対応するmypermキー情報を保存する。"""
        x = self.frame.cube.makedata().reshape(-1,1)
        perfect_key = ''
        perfect_changed_number = 0;
        if hasattr(self.frame.cube, 'state_to_str'):
            state_key = self.frame.cube.state_to_str()
        else:
            state_key = ''.join(self.frame.cube.state)

        matched_key = self.frame.myperms_col.get(state_key)
        if matched_key is not None:
            perfect_key = myperm_base_key(matched_key)
            simplified_lis = self._registered_base_transform_last_perms(
                tuple(simplified_lis),
                matched_key,
            )
        elif hasattr(self.frame.cube, 'last_perms_key'):
            perfect_key = self.frame.cube.last_perms_key()
            if hasattr(self.frame.cube, 'last_perms_changed_number'):
                perfect_changed_number = self.frame.cube.last_perms_changed_number()
        else:
            group_key, perfect_changed_number = self._group_changed_info(x)
            if getattr(self.frame.cube, 'myperms', None) and len(simplified_lis) > 0:
                simplified_lis = self._point_representative_last_perms(tuple(simplified_lis))
                effect = MypermEffectAnalyzer(self.frame.cube).analyze(simplified_lis)
                perfect_key = 'LP:' + effect.concise_name()
            else:
                perfect_key = group_key
        self.frame.solve_state.last_perfect_key = perfect_key
        self.frame.solve_state.last_perfect_changed_number = perfect_changed_number
        self.frame.solve_state.last_simplified_lis = tuple(simplified_lis)
        return simplified_lis

    def _registered_base_transform_last_perms(self, moves, matched_key):
        """登録済み last_perms を、その myperm 系列の #00 向きへ正規化する。"""
        transform_index = myperm_transform_index(matched_key)
        if transform_index is None or not hasattr(self.frame.cube, 'transform'):
            return tuple(moves)

        base_key = myperm_base_key(matched_key)
        source_transform_index = self._registered_actual_transform_index(base_key, transform_index)
        target_transform_index = self._registered_actual_transform_index(base_key, 0)
        normalized_moves = self.frame.cube.transform(moves, source_transform_index)
        if target_transform_index != 0:
            normalized_moves = self.frame.cube.transform(
                normalized_moves,
                target_transform_index,
                invert = True,
            )
        return tuple(normalized_moves)

    def _registered_actual_transform_index(self, base_key, transform_index):
        """point reindex 後の #NN から、実際の対称変換 index を取り出す。"""
        point_rows = getattr(self.frame.cube, 'myperm_transform_points', {}).get(base_key)
        if point_rows:
            for row in point_rows:
                if row.get("new_index") == transform_index:
                    return int(row["old_index"])
        return int(transform_index)

    def _point_representative_last_perms(self, moves):
        """LP fallback 用に、myperms_point が最大になる対称変換へ寄せる。"""
        try:
            return point_representative_transform(self.frame.cube, moves).moves
        except (OSError, AttributeError, KeyError, TypeError, ValueError):
            return tuple(moves)

    def _handle_no_progress_search(self, AI):
        """探索で状態が進まなかったときにsearch段階を打ち切る。"""
        state = self.frame.solve_state
        state.search_TF = False
        if self.frame.AIs[self.frame.AI_idx].search_mode == 'search2':
            for moves in state.move_lis:
                state.s += tuple(moves)
            self.frame.solve_state.reset_tracking()



    def _advance_greedy_step(self, AI):
        """myvalベースのgreedy選択で次の手順を進める。"""
        state = self.frame.solve_state
        if not self._supports_myperms_greedy():
            state.end_solve = True
            return False
        W = self._myperms_sort_weights(AI)
        if self.frame.AI_idx in list(range(self.frame.AInum)):
            AI = self.frame.myval_AI

        if len(state.val_lis) == 0:
            self._append_initial_value(AI)

        current_scramble = self._current_search_scramble()
        root_value = state.val_lis[-1]
        succeeded = self._choose_and_apply_myperms(AI,W)
        self._record_myval_history(current_scramble, root_value, succeeded)
        return succeeded

    def _myperms_sort_weights(self, AI):
        """myperm候補の並び替え用 move cost を返す。"""
        if getattr(AI, 'is_transformer', False):
            return np.ones(len(self.frame.cube.move_keys), dtype='f')
        x = self.frame.cube.makedata().reshape(-1,1)
        policy_logits = AI.predict(x,policy = True,value = False).reshape(-1)
        return -np.log(softmax(policy_logits) + 1.0e-8)

    def _append_initial_value(self, AI):
        """現在局面の初期valueをログへ追加する。"""
        state = self.frame.solve_state
        state_array = self.frame.cube.state.reshape(1,-1)
        if AI.myval:
            value = self._myval_values(AI, state_array).reshape(-1)
        else:
            x = self.frame.cube.makedata().reshape(-1,1)
            value = AI.predict(x,policy = False,value = True).reshape(-1)

        state.move_lis.append(tuple([]))
        state.key_lis.append('')
        state.val_lis.append(value[0])
        state.val_lis2.append([])
        state.display_move_lis.append(tuple([]))
        state.display_key_lis.append('')
        state.display_root_lis.append(value[0])
        state.display_val_lis.append(value[0])
        state.display_val_lis2.append([])

    def _choose_and_apply_myperms(self, AI,W):
        """候補mypermを評価して、最良の手順を実際に適用する。"""
        x = self.frame.cube.makedata().reshape(-1,1)
        top_group = self._select_top_group(x)
        self.frame.solve_state.last_top_group = top_group
        myperms_key, top_key = self._collect_myperms_keys(top_group,W)
        return self._evaluate_myperms_candidates(AI, myperms_key, top_key)

    def _select_top_group(self, x):
        """優先順位に従って、次に注目するgroup名を選ぶ。"""
        top_group = 'Corner'
        top_val = 0
        if not self._supports_group_values():
            return top_group
        for key in self.frame.priority_list[self.frame.AI_idx]:
            if key not in self.frame.cube.total_val or key not in self.frame.cube.group_val:
                continue
            value = self.frame.cube.total_val[key] - (self.frame.cube.group_val[key] @ x)[0][0]
            if value > 0.001 and value > top_val:
                top_val = value
                top_group = key
                break
        return top_group

    def _supports_group_values(self):
        return hasattr(self.frame.cube, 'group_val') and hasattr(self.frame.cube, 'total_val')

    def _group_value_names(self):
        if hasattr(self.frame.cube, '_group_name_map'):
            return self.frame.cube._group_name_map().values()
        return getattr(self.frame.cube, 'group_val', {}).keys()

    def _group_changed_info(self, state_data):
        """従来仕様のgroup別変更量とfallback名を返す。"""
        if not self._supports_group_values():
            return '', 0
        key_parts = []
        changed_number = 0
        for key in self._group_value_names():
            value = self.frame.cube.total_val[key] - (self.frame.cube.group_val[key] @ state_data)[0][0]
            if value <= 0.01:
                continue
            rounded_value = int(round(value, 0))
            key_parts.append(key + '(' + str(rounded_value) + ')')
            changed_number += rounded_value
        return ''.join(key_parts), changed_number

    def _supports_myperms_greedy(self):
        required_attrs = (
            'myperms_order',
            'num_to_piece',
            'default_color',
            'myperms_dict',
            'total_val',
            'group_val',
        )
        return all(hasattr(self.frame.cube, attr) for attr in required_attrs)

    def _collect_myperms_keys(self, top_group,W):
        """注目groupに対応するmyperm候補キー集合を集める。"""
        myperms_key = set(self.frame.cube.collect_single_move_and_rotate())
        top_key = set([])
        for group in self.frame.priority_list[self.frame.AI_idx]:
            for piece_index in self.frame.cube.myperms_order[group]:
                piece = self.frame.cube.num_to_piece[piece_index]
                color_key = reduce(lambda left,right:left+right,[self.frame.cube.state[index] for index in piece])
                if color_key != self.frame.cube.default_color[piece]:
                    top_key.add((piece,color_key))
                    myperms_key |= set(self.frame.cube.myperms_dict[(piece,color_key)])
                    

        myperms_key = list(myperms_key)
        myperms_key = sorted(myperms_key,key = lambda x : self._myperms_log_weight(W,x))
        return myperms_key, top_key

    def _myperms_log_weight(self,W,key):
        move_sequence = self.frame.cube.myperms[key]
        weight = 0.0
        for move in move_sequence:
            index = self.frame.cube.key_to_num[move]
            weight += W[index]
        
        return weight

    def _evaluate_myperms_candidates(self, AI, myperms_key, top_key):
        """myperm候補をvalueで評価し、継続できるかを判定する。"""
        state = self.frame.solve_state
        eval_num = 100
        state_size = len(self.frame.cube.state)
        X = None if AI.myval else np.empty((self.frame.cube.ips,eval_num),dtype = 'f')
        S = np.empty((eval_num,state_size),dtype = str)

        total_keys = len(myperms_key)
        idx = 0
        solved = False
        perfect_key = ''
        base = 0
        should_continue = False
        root_val = self._current_value_for_ai(AI)
        for key in myperms_key:
            for move in AI.myperms[key]:
                self.frame.cube.make_move(move)

            if self.frame.cube.is_perfect():
                solved = True
                perfect_key = key

            if not AI.myval:
                x = self.frame.cube.makedata()
                X[:,idx] = x
            S[idx,:] = self.frame.cube.state

            for move in self.frame.cube.invert_moves(AI.myperms[key]):
                self.frame.cube.make_move(move)

            idx += 1
            if solved:
                break

            if idx == eval_num or base + idx == total_keys:
                batch_count = idx
                if AI.myval:
                    values = self._myval_values(AI, S[:batch_count,:]).reshape(-1)
                else:
                    values = AI.predict(X[:,:batch_count],policy = False,value = True).reshape(-1)

                if root_val + 0.0001 < np.max(values):
                    should_continue = self._apply_best_myperms(AI, values, myperms_key, base,root_val)
                    if should_continue:
                        break
                else:
                    base += batch_count
                idx = 0

        if solved:
            self._finish_with_perfect_myperms(AI, perfect_key)
            return True
        if not should_continue:
            print(top_key)
            state.end_solve = True
            if self.frame.AI_idx != -1:
                for moves in state.move_lis:
                    state.s += tuple(moves)
                self.frame.my_scramble.append(state.s)
        return False

    def _apply_best_myperms(self, AI, values, myperms_key, base,root_val):
        """評価が最良のmyperm候補を選んでログと状態へ反映する。"""
        state = self.frame.solve_state
        args = np.where(values >= np.max(root_val) + 1.0e-4)[0]
        new_keys = []
        new_moves = []
        state_size = len(self.frame.cube.state)
        X_columns = []
        S_rows = []
        for arg_index in args[:1]:
            key = myperms_key[arg_index + base]
            move_count = 0
            for move in AI.myperms[key]:
                move_count += 1
                self.frame.cube.make_move(move)
                if not AI.myval:
                    X_columns.append(self.frame.cube.makedata())
                S_rows.append(self.frame.cube.state.copy())
                new_keys.append(f"{format_myperm_key(key)}({move_count})")
                new_moves.append(AI.myperms[key][:move_count])

            for move in self.frame.cube.invert_moves(AI.myperms[key]):
                self.frame.cube.make_move(move)

        if AI.myval:
            S = np.array(S_rows, dtype=str).reshape(-1, state_size)
            next_values = self._myval_values(AI, S).reshape(-1)
        else:
            X = np.array(X_columns, dtype='f').T
            next_values = AI.predict(X,policy = False,value = True).reshape(-1)

        top_arg = np.argmax(next_values)
        best_value = next_values[top_arg]
        best_key = new_keys[top_arg]
        best_move = new_moves[top_arg]

        state.key_lis.append(best_key)
        state.move_lis.append(best_move)
        state.val_lis.append(best_value)
        state.val_lis2.append([])
        state.display_key_lis.append(best_key)
        state.display_move_lis.append(best_move)
        state.display_root_lis.append(state.display_val_lis[-1])
        state.display_val_lis.append(best_value)
        state.display_val_lis2.append([])
        for move in best_move:
            self.frame.cube.make_move(move)
        return True

    def _current_value_for_ai(self, AI):
        if AI.myval:
            return self._myval_values(AI, self.frame.cube.state.reshape(1,-1)).reshape(-1)[0]
        x = self.frame.cube.makedata().reshape(-1,1)
        return AI.predict(x,policy = False,value = True).reshape(-1)[0]

    def _myval_values(self, AI, state_rows):
        if hasattr(AI, 'myval_predict_fast'):
            return AI.myval_predict_fast(state_rows).reshape(1,-1)
        if hasattr(AI, 'myval_predict'):
            return AI.myval_predict(None, state_rows).reshape(1,-1)
        return self._fallback_myval_values(state_rows).reshape(1,-1)

    def _fallback_myval_values(self, state_rows):
        state_rows = np.asarray(state_rows)
        solved_state = self.frame.cube.state_0
        return np.sum(state_rows == solved_state.reshape(1,-1), axis=1).astype('f')

    def _finish_with_perfect_myperms(self, AI, perfect_key):
        """完成が見つかったmyperm手順をログへ追加してsolve終了にする。"""
        state = self.frame.solve_state
        display_key = format_myperm_key(perfect_key)
        state.val_lis.append(PERFECT_VAL)
        state.key_lis.append(display_key)
        state.val_lis2.append([])
        state.move_lis.append(AI.myperms[perfect_key])
        state.display_key_lis.append(display_key)
        state.display_move_lis.append(AI.myperms[perfect_key])
        state.display_root_lis.append(state.display_val_lis[-1])
        state.display_val_lis.append(PERFECT_VAL)
        state.display_val_lis2.append([])
        state.end_solve = True

    def _update_viewers(self, AI):
        """現在局面のpolicy表示とMoveViewer表示を更新する。"""
        state = self.frame.solve_state
        W = self._viewer_policy_distribution(AI)
        self.frame.PV.put_val(W)
        if self._should_update_debug_viewer(AI):
            self.frame.debug_analysis_manager.show_current_viewer(self.frame.AI_idx,self.frame.cube_size ** 2)
        self.frame.MV.set_str(
            self.frame.display_move_sequence(state.s),
            self.frame.display_move_rows(state.display_move_lis),
            state.display_key_lis,
            state.display_root_lis,
            state.display_val_lis,
            state.display_val_lis2,
            self.frame.perf_num[self.frame.stage],
            self.frame.N,
            AI.search_mode,
            getattr(AI,'search2_value_loss_type',None),
        )
        self.frame.set_color(AI.cube.state)

    def _viewer_policy_distribution(self, AI):
        """表示用policy分布を返す。Transformerのgreedy fallback中は軽量表示にする。"""
        if self._is_transformer_greedy_fallback(AI):
            move_count = len(self.frame.cube.move_keys)
            return np.ones(move_count, dtype='f') / max(1, move_count)
        x = self.frame.cube.makedata().reshape(-1,1)
        return softmax(AI.predict(x,policy = True,value = False).reshape(-1))

    def _should_update_debug_viewer(self, AI):
        """自動solve中の重いGradViewer更新を抑制するか判定する。"""
        return not self._is_transformer_greedy_fallback(AI)

    def _is_transformer_greedy_fallback(self, AI):
        return (
            getattr(AI, 'is_transformer', False)
            and not self.frame.solve_state.search_TF
            and self.frame.AI_idx != -1
        )

    def _finalize_solve_step(self, succeeded):
        """1回のsolve終了時に成功集計・学習データ追加・次AI準備を行う。"""
        state = self.frame.solve_state
        result_recorded = False
        if state.phase > 0 and succeeded:
            if state.search_TF:
                self.frame.perf_num[self.frame.stage] += 1
                self.frame.success[self.frame.AI_idx] += 1
                result_recorded = True

            combined_moves = ()
            if state.search_TF or self.frame.AIs[self.frame.AI_idx].search_mode == "search2":
                self._store_connected_myloss_training_sample()
                for moves in state.move_lis:
                    combined_moves += moves
                    if self._store_search2_training_sample(state.s, combined_moves):
                        state.s += combined_moves
                        combined_moves = ()

            if state.search_TF and len(state.move_lis) >= 2:
                simplified_moves = tuple(state.last_simplified_lis)
                if simplified_moves not in self.frame.myperms_vals:
                    myperms_key = resolve_myperm_key(self.frame.cube, state.last_perfect_key)
                    improved = (
                        myperms_key is not None
                        and len(self.frame.cube.myperms[myperms_key]) > len(simplified_moves)
                    )
                    if myperms_key is not None and len(self.frame.cube.myperms[myperms_key]) > len(simplified_moves):
                        print(self.frame.AI_idx,state.last_perfect_key,len(simplified_moves),'<-----------')
                    else:
                        print(self.frame.AI_idx,state.last_perfect_key,len(simplified_moves))
                    if state.last_perfect_key not in self.frame.last_perms:
                        self.frame.last_perms[state.last_perfect_key] = set([])
                        self.frame.last_perms_changed_number[state.last_perfect_key] = state.last_perfect_changed_number
                    self.frame.last_perms[state.last_perfect_key].add(simplified_moves)
                    if improved:
                        self.frame.append_log(
                            f"last_perm improved: ai={self.frame.AI_idx} key={state.last_perfect_key} len={len(simplified_moves)}"
                        )
                    else:
                        self.frame.append_log(
                            f"last_perm found: ai={self.frame.AI_idx} key={state.last_perfect_key} len={len(simplified_moves)}"
                        )

            if state.search_TF and self.frame.AIs[self.frame.AI_idx].search_mode == 'search2' and self.frame.AIs[self.frame.AI_idx].search2_value_loss_type == "myloss":
                max_level = max(1,int(- state.val_lis2[1][0] / 5))
                if max_level >= len(self.frame.cube.my_scrambles2):
                    for _ in range(max_level - len(self.frame.cube.my_scrambles2) + 1):
                        self.frame.cube.create_new_set()

                for move_index in range(1,len(state.move_lis)):
                    inverted_moves = self.frame._normalize_move_sequence(self.frame.cube.invert_moves(state.move_lis[move_index]))
                    level = max(0,int(- state.val_lis2[move_index][0] / 5))
                    self.frame.cube.register_scramble_sequence(level, inverted_moves)

            if state.search_TF or self.frame.AIs[self.frame.AI_idx].search_mode in ('search3', 'transformer'):
                for ai_index in range(self.frame.AInum):
                    self.frame.search_data_manager.store_search3_data(ai_index)

        if state.phase > 0:
            self.frame.success_viewer.put_result(self.frame.success,self.frame.N,self.frame.AI_idx,result_recorded)

        self.frame.N += 1
        self.frame.AI_idx += 1
        self.frame.AI_idx %= self.frame.AInum
        search2_data_delta = len(self.frame.AIs[0].datas) - self.frame.data_len
        search3_data_delta = self.frame.current_search3_data_len() - self.frame.data_search3_len
        if search2_data_delta >= 400 or search3_data_delta >= 400:
            self.frame.learn()
            self.frame.data_len = len(self.frame.AIs[0].datas)
            self.frame.data_search3_len = self.frame.current_search3_data_len()

        if self.frame.N == 200:
            self.frame.N = 0
            self.frame.stage += 1
            if self.frame.stage == self.frame.stage_num:
                self.frame.stage = 0

                if np.sum(self.frame.perf_num) >= 180:
                    self.frame.level[:] += 1

                print(self.frame.level[:])
                self.frame.my_scramble = []
                self.frame.perf_num[:] = 0
                self.frame.learn()

        state.phase = -1

    def _store_connected_myloss_training_sample(self):
        """Pairwise Search2成功時に、MyLoss向けの連結手順データを追加する。"""
        state = self.frame.solve_state
        if not state.search_TF:
            return False
        source_ai = self._current_source_ai()
        if getattr(source_ai,'search2_value_loss_type',None) != 'myloss2_pairwise':
            return False
        if len(state.move_lis) <= 1:
            return False
        target_ai_indices = self._myloss_search2_ai_indices()
        if not target_ai_indices:
            return False
        connected_moves = tuple([])
        for moves in state.move_lis:
            connected_moves += tuple(moves)
        return self._store_search2_training_sample(
            state.s,
            connected_moves,
            source_ai = source_ai,
            source_search_mode = getattr(source_ai,'search_mode','search2'),
            source_value_loss_type = 'myloss_connected',
            target_ai_indices = target_ai_indices,
        )

    def _myloss_search2_ai_indices(self):
        indices = []
        for ai_index, ai in enumerate(self.frame.AIs):
            if (
                getattr(ai,'search_mode',None) == 'search2'
                and getattr(ai,'search2_value_loss_type',None) == 'myloss'
            ):
                indices.append(ai_index)
        return indices

    def _current_source_ai(self):
        if self.frame.AI_idx in range(self.frame.AInum):
            return self.frame.AIs[self.frame.AI_idx]
        return self.frame.myval_AI

    def _store_search2_training_sample(
        self,
        scramble,
        moves,
        source_ai = None,
        source_search_mode = None,
        source_value_loss_type = None,
        target_ai_indices = None,
    ):
        """保存前に手順列を簡約してから Search2 学習データへ追加する。"""
        simplified_moves = self.frame.cube.simplify(moves)
        if len(simplified_moves) == 0:
            return False

        rewards = np.zeros(len(simplified_moves),dtype = 'f')
        rewards[-1] = 10
        datas = self.frame.cube.make_transformations(scramble,simplified_moves)

        state = self.frame.solve_state
        if source_ai is None:
            source_ai = self._current_source_ai()
        if source_search_mode is None:
            source_search_mode = getattr(source_ai,'search_mode','search2')
        if source_value_loss_type is None:
            source_value_loss_type = getattr(source_ai,'search2_value_loss_type',None)
        if not state.search_TF:
            source_search_mode = 'myval'
            source_value_loss_type = 'myval'
        data_item = data(
            datas[0][0],
            datas[1][0],
            rewards,
            perfect_key = state.last_perfect_key,
            top_group = state.last_top_group,
            source_ai_index = self.frame.AI_idx,
            source_search_mode = source_search_mode,
            source_search2_value_loss_type = source_value_loss_type,
        )
        data_item.succeeded = True
        self.frame.AIs[self.frame.AI_idx].datas.append(data_item)

        data_index = len(self.frame.AIs[self.frame.AI_idx].datas) - 1
        if target_ai_indices is None:
            target_ai_indices = self._default_search2_target_ai_indices(source_value_loss_type)
        for ai_index in target_ai_indices:
            self.frame.AIs[ai_index].indices.append(data_index)
        return True

    def _default_search2_target_ai_indices(self, source_value_loss_type):
        """通常Search2データを学習対象にするAI indexを返す。"""
        if source_value_loss_type == 'myloss2_pairwise':
            return [
                ai_index
                for ai_index, ai in enumerate(self.frame.AIs)
                if not (
                    getattr(ai,'search_mode',None) == 'search2'
                    and getattr(ai,'search2_value_loss_type',None) == 'myloss'
                )
            ]
        return range(self.frame.AInum)




    
