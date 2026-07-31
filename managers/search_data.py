"""Search data generation helpers."""

from __future__ import annotations

import numpy as np
from collections import Counter

from model.search_result import data_search3


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class SearchDataManager:
    """solve履歴をSearch3用の学習データへ変換して蓄積する。"""

    def __init__(self, frame):
        self.frame = frame

    def append_bootstrap_search3_datas(self, bootstrap_datas = None):
        """Append bootstrap move sequences as Search3-format training data."""
        if bootstrap_datas is None:
            return

        for ai_index, ai in enumerate(self.frame.AIs):
            if ai.search_mode not in ('search3', 'transformer'):
                continue
            next_index = len(ai.datas_search3)
            for moves in bootstrap_datas:
                search_data = self.build_bootstrap_search3_sample(moves, ai)
                if search_data is None:
                    continue
                search_data.succeeded = True
                ai.datas_search3.append(search_data)
                ai.indices_search3.append(next_index)
                next_index += 1

    def build_bootstrap_search3_sample(self, moves, ai):
        """Convert one bootstrap move sequence into a Search3 training sample."""
        scramble_moves = self.frame.cube.simplify(moves)
        if len(scramble_moves) == 0:
            return None

        segment_moves = self.frame.cube.simplify(self.frame.cube.invert_moves(scramble_moves))
        if len(segment_moves) == 0:
            return None

        scramble = scramble_moves
        value_targets = self._bootstrap_value_targets(len(segment_moves))
        rewards = value_targets.copy()
        value_trace = value_targets.tolist()
        value_trace_raw = value_trace
        return data_search3(
            scramble,
            segment_moves,
            rewards,
            value_trace[0],
            value_trace,
            value_trace[-1],
            {'source': 'bootstrap'},
            policy_target = self._one_hot_policy_target(segment_moves),
            search_mode = 'bootstrap',
            sample_weight = 0.1,
            value_targets = value_targets,
            root_value_raw = value_trace_raw[0],
            value_trace_raw = value_trace_raw,
            best_value_raw = value_trace_raw[-1],
            perfect_key = 'bootstrap',
            top_group = None,
            end_reason = 'bootstrap',
            source_succeeded = True,
            solve_succeeded = True,
        )

    def _bootstrap_value_targets(self, segment_length):
        value_targets = np.zeros(segment_length + 1, dtype = 'f')
        gamma = self.frame.value_target_gamma
        for move_index in range(segment_length + 1):
            value_targets[move_index] = gamma ** max(segment_length - move_index, 0)
        return value_targets

    def _one_hot_policy_target(self, moves):
        policy_target = np.zeros((len(self.frame.move_keys),), dtype = 'f')
        policy_target[self.frame.cube.key_to_num[moves[0]]] = 1.0
        return policy_target

    def store_search3_data(self, ai_index):
        """現在のsolve履歴を、指定AIのSearch3学習データとして追加する。"""
        state = self.frame.solve_state
        if self.frame.AI_idx == -1 or len(state.search_history) == 0:
            return

        ai = self.frame.AIs[ai_index]
        source_ai = self.frame.AIs[self.frame.AI_idx]
        next_index = len(ai.datas_search3)
        sample_cache = self._search3_sample_cache(state)
        for history_index, history_item in enumerate(state.search_history):
            if history_index not in sample_cache:
                sample_cache[history_index] = self.build_search3_training_sample(
                    history_index,
                    history_item,
                    source_ai,
                )
            search_data = sample_cache[history_index]
            if search_data is None:
                continue
            search_data.solve_succeeded = True
            ai.datas_search3.append(search_data)
            ai.indices_search3.append(next_index)
            next_index += 1

    def _search3_sample_cache(self, state):
        cache = getattr(state, '_search3_training_sample_cache', None)
        if cache is None:
            cache = {}
            state._search3_training_sample_cache = cache
        return cache

    def search3_training_moves(self, search_result):
        """Search3 学習用に手順列を simplify したものを返す。"""
        return self.frame.cube.simplify(search_result.moves)

    def build_search3_training_sample(self, history_index, history_item, source_ai):
        """保存前に手順列を簡約してから Search3 学習データ 1 件を作る。"""
        search_result = history_item['search_result']
        segment_moves = self.search3_training_moves(search_result)
        if len(segment_moves) == 0:
            return None

        value_targets = self.build_segment_value_targets(history_index, len(segment_moves))
        rewards = value_targets.copy()
        value_trace_raw = self.rebuild_search3_value_trace_raw(
            history_item['scramble'],
            segment_moves,
            source_ai,
        )
        value_trace = [sigmoid(value) for value in value_trace_raw]
        return data_search3(
            history_item['scramble'],
            segment_moves,
            rewards,
            value_trace[0],
            value_trace,
            value_trace[-1],
            search_result.stats,
            policy_target = self.build_search3_policy_target(search_result, segment_moves),
            search_mode = search_result.search_mode,
            sample_weight = self.search3_sample_weight(search_result),
            value_targets = value_targets,
            root_value_raw = value_trace_raw[0],
            value_trace_raw = value_trace_raw,
            best_value_raw = value_trace_raw[-1],
            perfect_key = history_item.get('perfect_key'),
            top_group = history_item.get('top_group'),
            end_reason = getattr(search_result,'end_reason',None),
            source_succeeded = getattr(search_result,'succeeded',False),
            solve_succeeded = False,
        )

    def build_segment_value_targets(self, history_index, segment_length):
        """残り手数とgammaから、その探索区間のvalue target列を作る。"""
        remaining_length = self.search3_remaining_length(history_index)
        value_targets = np.zeros(segment_length + 1,dtype = 'f')
        gamma = self.frame.value_target_gamma
        for move_index in range(segment_length + 1):
            value_targets[move_index] = gamma ** max(remaining_length - move_index,0)
        return value_targets

    def search3_remaining_length(self, history_index):
        """指定履歴以降に残っている総手数を数える。"""
        remaining_length = 0
        for history_item in self.frame.solve_state.search_history[history_index:]:
            remaining_length += len(
                self.search3_training_moves(history_item['search_result'])
            )
        return remaining_length

    def build_search3_policy_target(self, search_result, segment_moves):
        """探索結果からpolicy targetを作る。soft targetがあれば正規化して使う。"""
        if search_result.policy_target is not None:
            policy_target = np.array(search_result.policy_target,dtype = 'f').reshape(-1)
            total = np.sum(policy_target)
            if total > 0:
                return policy_target / total

        policy_target = np.zeros((len(self.frame.move_keys),),dtype = 'f')
        if len(segment_moves) > 0:
            first_move = segment_moves[0]
            policy_target[self.frame.cube.key_to_num[first_move]] = 1.0
            return policy_target

        return np.ones((len(self.frame.move_keys),),dtype = 'f') / len(self.frame.move_keys)

    def rebuild_search3_value_trace_raw(self, scramble, moves, ai):
        """簡約後の全中間局面をまとめて推論して raw value trace を再計算する。"""
        if hasattr(ai, '_predict_state_batch'):
            state_snapshots = self._collect_search3_trace_state_snapshots(scramble, moves)
            prediction = ai._predict_state_batch(state_snapshots)
            return prediction[-1, :].reshape(-1).tolist()
        features = self._collect_search3_trace_features(scramble, moves)
        prediction = ai._predict_search2(features)
        return prediction[-1, :].reshape(-1).tolist()

    def _collect_search3_trace_state_snapshots(self, scramble, moves):
        """Search3 value trace 用に中間局面の state snapshot 列をまとめて作る。"""
        current_snapshot = self._snapshot_current_cube_state()
        self.frame.cube.reset()
        self.frame.cube.scramble(0, scramble)

        state_snapshots = [self._snapshot_current_cube_state()]
        for move in moves:
            self.frame.cube.make_move(move)
            state_snapshots.append(self._snapshot_current_cube_state())

        self._restore_cube_snapshot(current_snapshot)
        return state_snapshots

    def _snapshot_current_cube_state(self):
        if hasattr(self.frame.cube, '_snapshot'):
            return self.frame.cube._snapshot()
        return self.frame.cube.state.copy()

    def _restore_cube_snapshot(self, snapshot):
        if hasattr(self.frame.cube, '_restore'):
            self.frame.cube._restore(snapshot)
        else:
            self.frame.cube.state[:] = snapshot

    def _collect_search3_trace_features(self, scramble, moves):
        """Search3 value trace 用に中間局面の特徴量列をまとめて作る。"""
        current_state = self.frame.cube.state.copy()
        self.frame.cube.reset()
        self.frame.cube.scramble(0, scramble)

        features = np.zeros((self.frame.cube.ips, len(moves) + 1), dtype = 'f')
        features[:, 0] = self.frame.cube.makedata()
        for move_index, move in enumerate(moves, 1):
            self.frame.cube.make_move(move)
            features[:, move_index] = self.frame.cube.makedata()

        self.frame.cube.state[:] = current_state
        return features

    def search3_sample_weight(self, search_result):
        """探索モードごとにSearch3学習のサンプル重みを決める。"""
        if search_result.search_mode == 'search3':
            return 1.0
        if search_result.search_mode == 'search2':
            return 1.0
        if search_result.search_mode == 'myval':
            return 0.2
        return 1.0

    def dataset_summary_text(self, ai_index):
        """Return a human-readable summary of Search2/Search3 datasets for one AI."""
        ai = self.frame.AIs[ai_index]
        search2_summary = self._search2_dataset_summary(ai)
        search3_summary = self._search3_dataset_summary(ai)
        return '\n'.join(
            [f'AI {ai_index} dataset summary', ''] +
            self._format_search2_summary(search2_summary) +
            [''] +
            self._format_search3_summary(search3_summary)
        )

    def representative_sample_text(self, ai_index, dataset_kind, selector_value, selector_kind = 'perfect_key'):
        """Return a formatted representative sample for the requested selector."""
        sample = self.representative_sample(ai_index, dataset_kind, selector_value, selector_kind)
        if sample is None:
            return None
        return self._format_sample_text(ai_index, dataset_kind, sample)

    def representative_sample(self, ai_index, dataset_kind, selector_value, selector_kind = 'perfect_key'):
        """Return the representative sample object for the requested selector."""
        return self._representative_sample(ai_index, dataset_kind, selector_value, selector_kind)

    def _search2_dataset_summary(self, ai):
        """Aggregate Search2 dataset statistics."""
        items = ai.datas
        lengths = [len(item.moves) for item in items]
        first_moves = Counter(item.moves[0] for item in items if len(item.moves) > 0)
        successes = sum(1 for item in items if getattr(item, 'succeeded', False))
        perfect_keys = Counter(item.perfect_key for item in items if getattr(item, 'perfect_key', None))
        top_groups = Counter(item.top_group for item in items if getattr(item, 'top_group', None))
        source_loss_types = Counter(
            getattr(item, 'source_search2_value_loss_type', None) or 'unknown'
            for item in items
        )
        source_ai_indices = Counter(
            getattr(item, 'source_ai_index', None)
            for item in items
            if getattr(item, 'source_ai_index', None) is not None
        )
        source_modes = Counter(
            getattr(item, 'source_search_mode', None) or 'unknown'
            for item in items
        )
        return {
            'count': len(items),
            'lengths': lengths,
            'successes': successes,
            'first_moves': first_moves,
            'perfect_keys': perfect_keys,
            'top_groups': top_groups,
            'source_loss_types': source_loss_types,
            'source_ai_indices': source_ai_indices,
            'source_modes': source_modes,
        }

    def _search3_dataset_summary(self, ai):
        """Aggregate Search3 dataset statistics."""
        items = ai.datas_search3
        lengths = [len(item.moves) for item in items]
        trace_lengths = [len(item.value_trace) for item in items]
        first_moves = Counter(item.moves[0] for item in items if len(item.moves) > 0)
        mode_counts = Counter(item.search_mode for item in items)
        end_reasons = Counter(getattr(item, 'end_reason', None) for item in items)
        source_successes = sum(1 for item in items if getattr(item, 'source_succeeded', getattr(item, 'succeeded', False)))
        solve_successes = sum(1 for item in items if getattr(item, 'solve_succeeded', getattr(item, 'succeeded', False)))
        policy_targets = sum(1 for item in items if item.policy_target is not None)
        rewards = [float(np.mean(item.rewards)) for item in items if item.rewards is not None and len(item.rewards) > 0]
        targets = [float(np.mean(item.value_targets)) for item in items if item.value_targets is not None and len(item.value_targets) > 0]
        perfect_keys = Counter(item.perfect_key for item in items if getattr(item, 'perfect_key', None))
        top_groups = Counter(item.top_group for item in items if getattr(item, 'top_group', None))
        return {
            'count': len(items),
            'lengths': lengths,
            'trace_lengths': trace_lengths,
            'first_moves': first_moves,
            'mode_counts': mode_counts,
            'end_reasons': end_reasons,
            'source_successes': source_successes,
            'solve_successes': solve_successes,
            'policy_targets': policy_targets,
            'rewards': rewards,
            'targets': targets,
            'perfect_keys': perfect_keys,
            'top_groups': top_groups,
        }

    def _format_search2_summary(self, summary):
        """Format Search2 dataset summary into text lines."""
        lines = ['[Search2]']
        lines += self._basic_length_lines(summary['count'], summary['lengths'])
        lines.append(f"succeeded: {summary['successes']}")
        lines += self._counter_lines('top first moves', summary['first_moves'])
        lines += self._counter_lines('source value loss types', summary['source_loss_types'])
        lines += self._counter_lines('source AI indices', summary['source_ai_indices'])
        lines += self._counter_lines('source modes', summary['source_modes'])
        lines += self._counter_lines('top perfect keys', summary['perfect_keys'])
        lines += self._counter_lines('top groups', summary['top_groups'])
        return lines

    def _format_search3_summary(self, summary):
        """Format Search3 dataset summary into text lines."""
        lines = ['[Search3]']
        lines += self._basic_length_lines(summary['count'], summary['lengths'])
        lines += self._average_length_lines('trace len', summary['trace_lengths'])
        lines.append(f"source succeeded: {summary['source_successes']}")
        lines.append(f"solve succeeded: {summary['solve_successes']}")
        lines.append(f"policy targets: {summary['policy_targets']}")
        lines += self._average_scalar_lines('avg reward', summary['rewards'])
        lines += self._average_scalar_lines('avg value target', summary['targets'])
        lines += self._counter_lines('search modes', summary['mode_counts'])
        lines += self._counter_lines('end reasons', summary['end_reasons'])
        lines += self._counter_lines('top first moves', summary['first_moves'])
        lines += self._counter_lines('top perfect keys', summary['perfect_keys'])
        lines += self._counter_lines('top groups', summary['top_groups'])
        return lines

    def _basic_length_lines(self, count, lengths):
        """Format count/min/max/avg length stats."""
        if count == 0:
            return ['samples: 0']
        return [
            f'samples: {count}',
            f'avg len: {float(np.mean(lengths)):.3f}',
            f'min/max len: {int(np.min(lengths))} / {int(np.max(lengths))}',
        ]

    def _average_length_lines(self, label, lengths):
        """Format an average length line if data exists."""
        if len(lengths) == 0:
            return [f'{label}: n/a']
        return [f'{label}: {float(np.mean(lengths)):.3f}']

    def _average_scalar_lines(self, label, values):
        """Format an average scalar line if data exists."""
        if len(values) == 0:
            return [f'{label}: n/a']
        return [f'{label}: {float(np.mean(values)):.6f}']

    def _counter_lines(self, label, counter, top_n = 8):
        """Format top entries from a Counter."""
        lines = [f'{label}:']
        if len(counter) == 0:
            lines.append('  (none)')
            return lines
        for key, count in counter.most_common(top_n):
            lines.append(f'  {key}: {count}')
        return lines

    def _representative_sample(self, ai_index, dataset_kind, selector_value, selector_kind = 'perfect_key'):
        """Return the shortest sample matching the requested selector."""
        ai = self.frame.AIs[ai_index]
        if dataset_kind == 'Search2':
            items = ai.datas
        else:
            items = ai.datas_search3

        selector_attr = 'perfect_key' if selector_kind == 'perfect_key' else 'top_group'
        candidates = [
            item for item in items
            if getattr(item, selector_attr, None) == selector_value
        ]
        if len(candidates) == 0:
            return None
        return min(candidates, key = lambda item: len(item.moves))

    def _format_sample_text(self, ai_index, dataset_kind, sample):
        """Format one representative sample as readable text."""
        display_scramble = self.frame.display_move_sequence(sample.scramble)
        display_moves = self.frame.display_move_sequence(sample.moves)
        lines = [
            f'{dataset_kind} representative sample',
            f'AI: {ai_index}',
            f'perfect_key: {getattr(sample, "perfect_key", None)}',
            f'top_group: {getattr(sample, "top_group", None)}',
            f'succeeded: {getattr(sample, "succeeded", False)}',
            f'source_ai_index: {getattr(sample, "source_ai_index", None)}',
            f'source_search_mode: {getattr(sample, "source_search_mode", None)}',
            f'source_search2_value_loss_type: {getattr(sample, "source_search2_value_loss_type", None)}',
            f'steps_to_goal: {getattr(sample, "steps_to_goal", None)}',
            f'move length: {len(sample.moves)}',
            f'scramble: {display_scramble}',
            f'moves: {display_moves}',
        ]
        if getattr(sample, 'rewards', None) is not None:
            lines.append(f'rewards: {np.array2string(np.asarray(sample.rewards), precision = 4, suppress_small = True)}')
        if dataset_kind == 'Search3':
            lines.append(f'search_mode: {sample.search_mode}')
            lines.append(f'end_reason: {getattr(sample, "end_reason", None)}')
            lines.append(f'source_succeeded: {getattr(sample, "source_succeeded", getattr(sample, "succeeded", False))}')
            lines.append(f'solve_succeeded: {getattr(sample, "solve_succeeded", getattr(sample, "succeeded", False))}')
            if getattr(sample, 'value_targets', None) is not None:
                lines.append(
                    'value_targets: ' +
                    np.array2string(np.asarray(sample.value_targets), precision = 4, suppress_small = True)
                )
            if getattr(sample, 'value_trace', None) is not None:
                lines.append(
                    'value_trace: ' +
                    np.array2string(np.asarray(sample.value_trace), precision = 4, suppress_small = True)
                )
        return '\n'.join(lines)
