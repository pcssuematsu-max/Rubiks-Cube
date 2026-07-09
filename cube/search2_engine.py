"""Search2 engine implementation."""

import numpy as np
from heapdict import heapdict

from model.search_result import SearchResult


def softmax(x):
    """Column-wise softmax used by Search2 candidate expansion."""
    shifted = x - np.max(x, axis=0, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)


class Search2Engine:
    def __init__(self, ai):
        self.ai = ai
        self.cube = ai.cube
        self.ips = ai.ips

    def search2(self, N):
        value_by_key = {}
        prediction_cache = {} if self._uses_transformer_state_batch() else None
        frontier = heapdict()
        frontier[()] = -N * 2
        best_value = -1000000000
        best_key = ()
        counter = np.zeros(2, dtype='i')
        move_count = len(self.cube.move_keys)
        continue_searching = True
        root_value = None
        frontier_peak = 1

        while continue_searching:
            search_batch = self._pop_search_batch(frontier)
            key_list, priorities = self._extract_search_keys(search_batch)
            features, state_snapshots, priority_weights, invalid_move_mask, invalid_key_mask, early_result = self._build_search_inputs(
                key_list, priorities, move_count, value_by_key, counter
            )
            if early_result is not None:
                self._record_search_stats(frontier_peak,value_by_key,frontier)
                return early_result

            if self._uses_transformer_state_batch():
                prediction = self._predict_transformer_states(state_snapshots, prediction_cache)
            else:
                prediction = self.ai._predict_search2(features)
            state_values, move_priors = self._split_search_outputs(prediction, invalid_move_mask, priority_weights)
            state_values[invalid_key_mask] = -np.inf

            best_value, best_key, root_value, done = self._update_search_progress(
                key_list, state_values, value_by_key, counter, best_value, best_key, root_value, N
            )
            if done is not None:
                self._record_search_stats(frontier_peak,value_by_key,frontier)
                return done

            self._enqueue_search_candidates(frontier, key_list, move_priors)
            frontier_peak = max(frontier_peak,len(frontier))
            self._prune_frontier(frontier)
            continue_searching = (len(frontier) > 0)

        result = self._finalize_search(best_key, value_by_key, counter)
        self._record_search_stats(frontier_peak,value_by_key,frontier)
        return result

    def _pop_search_batch(self, frontier):
        batch_size = min(len(frontier), 100)
        search_batch = []
        for _ in range(batch_size):
            search_batch.append(frontier.popitem())
        return search_batch

    def _uses_transformer_state_batch(self):
        return hasattr(self.ai, '_predict_state_batch') and getattr(self.ai, 'is_transformer', False)

    def _extract_search_keys(self, search_batch):
        key_list = [item[0] for item in search_batch]
        priorities = [item[1] for item in search_batch]
        return key_list, priorities

    def _build_search_inputs(self, key_list, priorities, move_count, value_by_key, counter):
        batch_size = len(key_list)
        use_state_batch = self._uses_transformer_state_batch()
        features = None if use_state_batch else np.zeros((self.ips, batch_size))
        state_snapshots = []
        priority_weights = np.zeros(batch_size)
        invalid_move_mask = np.zeros((move_count, batch_size), dtype=bool)
        invalid_key_mask = np.zeros(batch_size, dtype=bool)

        for index, key in enumerate(key_list):
            applied_moves = self._try_apply_moves(key)
            if applied_moves is None:
                invalid_move_mask[:, index] = True
                invalid_key_mask[index] = True
                value_by_key[key] = -np.inf
                if use_state_batch:
                    state_snapshots.append(self._snapshot_state())
                continue
            self._apply_inverse_mask(key, invalid_move_mask, index)
            self._apply_legal_move_mask(invalid_move_mask, index)
            if self.cube.is_perfect():
                self._revert_moves(applied_moves)
                value_history = self._build_value_history(value_by_key, key, self.ai.perfect_val)
                early_result = SearchResult(True, key, value_history[0], value_history, self.ai.perfect_val, counter, search_mode='search2', end_reason='solved')
                if features is not None:
                    features = features[:, :1]
                return features, state_snapshots[:1], priority_weights[:1], invalid_move_mask[:, :1], invalid_key_mask[:1], early_result

            if use_state_batch:
                state_snapshots.append(self._snapshot_state())
            else:
                features[:, index] = self.cube.makedata().reshape(-1)
            priority_weights[index] = -priorities[index]
            self._revert_moves(applied_moves)

        return features, state_snapshots, priority_weights, invalid_move_mask, invalid_key_mask, None

    def _predict_transformer_states(self, state_snapshots, prediction_cache):
        if prediction_cache is None:
            return self.ai._predict_state_batch(state_snapshots)

        output_size = len(self.cube.move_keys) + 1
        prediction = np.zeros((output_size, len(state_snapshots)), dtype='f')
        uncached_indices = []
        uncached_snapshots = []
        uncached_keys = []

        for index, snapshot in enumerate(state_snapshots):
            cache_key = self._state_cache_key(snapshot)
            cached = prediction_cache.get(cache_key)
            if cached is None:
                uncached_indices.append(index)
                uncached_snapshots.append(snapshot)
                uncached_keys.append(cache_key)
                continue
            prediction[:, index] = cached

        if len(uncached_snapshots) > 0:
            uncached_prediction = self.ai._predict_state_batch(uncached_snapshots)
            for offset, index in enumerate(uncached_indices):
                column = uncached_prediction[:, offset].astype('f', copy=True)
                prediction[:, index] = column
                prediction_cache[uncached_keys[offset]] = column

        return prediction

    def _state_cache_key(self, snapshot):
        if isinstance(snapshot, np.ndarray):
            return ('ndarray', snapshot.shape, str(snapshot.dtype), snapshot.tobytes())
        if isinstance(snapshot, (tuple, list)):
            return (type(snapshot).__name__, tuple(self._state_cache_key(item) for item in snapshot))
        if isinstance(snapshot, dict):
            return ('dict', tuple(sorted((key, self._state_cache_key(value)) for key, value in snapshot.items())))
        try:
            hash(snapshot)
        except TypeError:
            return ('repr', repr(snapshot))
        return ('value', snapshot)

    def _snapshot_state(self):
        if hasattr(self.cube, '_snapshot'):
            return self.cube._snapshot()
        return self.cube.state.copy()

    def _apply_inverse_mask(self, key, invalid_move_mask, column_index):
        if len(key) > 0:
            removed_moves = {self.cube.invert_str(key[-1])}
        else:
            removed_moves = set([])
        filtered_moves = filter(
            lambda move: move in removed_moves or move[:2] in removed_moves,
            self.cube.move_keys,
        )
        for move in filtered_moves:
            move_index = self.cube.move_keys.index(move)
            invalid_move_mask[move_index, column_index] = True

    def _apply_legal_move_mask(self, invalid_move_mask, column_index):
        if not hasattr(self.cube, 'legal_move_mask'):
            return
        invalid_move_mask[:, column_index] |= ~self.cube.legal_move_mask()

    def _split_search_outputs(self, prediction, invalid_move_mask, priority_weights):
        state_values = prediction[-1].reshape(-1)
        move_priors = prediction[:-1]
        move_priors[invalid_move_mask] = -10000
        move_priors = softmax(move_priors) * priority_weights
        return state_values, move_priors

    def _update_search_progress(self, key_list, state_values, value_by_key, counter, best_value, best_key, root_value, search_budget):
        for index, key in enumerate(key_list):
            value_by_key[key] = state_values[index]
            if key == ():
                root_value = state_values[index]
                best_value = root_value + 0.0001
                best_key = ()

            counter[1] += 1
            if root_value is None:
                continue

            node_value = state_values[index]
            if node_value > best_value:
                best_value = node_value
                best_key = key

            if node_value > root_value + 0.0001:
                counter[0] += 1

            if self._should_stop(best_key, node_value, root_value, counter, search_budget):
                result_key, value_history = self._improving_result_key_and_history(value_by_key, best_key, root_value)
                result = SearchResult(False, result_key, root_value, value_history, value_history[-1], counter, search_mode='search2', end_reason='budget')
                return best_value, best_key, root_value, result

        return best_value, best_key, root_value, None

    def _should_stop(self, best_key, node_value, root_value, counter, search_budget):
        search_improved = (
            len(best_key) >= 1
            and node_value > root_value + self.ai.skip_difference
            and self.ai.skip_search
        )
        budget_exhausted = (counter[1] >= search_budget)
        return search_improved or budget_exhausted

    def _enqueue_search_candidates(self, frontier, key_list, move_priors):
        candidate_indices = np.where(move_priors >= 1)
        for move_index, key_index in zip(candidate_indices[0], candidate_indices[1]):
            if len(key_list[key_index]) < 200:
                next_key = key_list[key_index] + (self.cube.move_keys[move_index],)
                frontier[next_key] = -move_priors[move_index, key_index]

    def _prune_frontier(self, frontier):
        limit = int(getattr(self.ai,'search2_max_frontier',0) or 0)
        if limit <= 0 or len(frontier) <= limit:
            return
        kept_items = sorted(frontier.items(), key = lambda item:item[1])[:limit]
        frontier.clear()
        for key,priority in kept_items:
            frontier[key] = priority

    def _record_search_stats(self, frontier_peak, value_by_key, frontier):
        self.ai._last_search2_frontier_peak = int(frontier_peak)
        self.ai._last_search2_value_count = int(len(value_by_key))
        self.ai._last_search2_frontier_remaining = int(len(frontier))

    def _finalize_search(self, best_key, value_by_key, counter):
        for move in best_key:
            self.cube.make_move(move)
        is_perfect = self.cube.is_perfect()
        for move in self.cube.invert_moves(best_key):
            self.cube.make_move(move)
        best_key, value_history = self._improving_result_key_and_history(value_by_key, best_key, value_by_key[()])
        end_reason = 'solved' if is_perfect else 'budget'
        return SearchResult(is_perfect, best_key, value_history[0], value_history, value_history[-1], counter, search_mode='search2', end_reason=end_reason)

    def _apply_moves(self, key):
        for move in key:
            self.cube.make_move(move)

    def _try_apply_moves(self, key):
        applied_moves = []
        try:
            for move in key:
                if hasattr(self.cube, 'is_legal_move') and not self.cube.is_legal_move(move):
                    raise ValueError(move)
                self.cube.make_move(move)
                applied_moves.append(move)
        except Exception:
            self._revert_moves(tuple(applied_moves))
            return None
        return tuple(applied_moves)

    def _revert_moves(self, key):
        for move in self.cube.invert_moves(key):
            self.cube.make_move(move)

    def _build_value_history(self, value_by_key, key, tail_value=None):
        value_history = []
        for index in range(len(key)):
            value_history.append(value_by_key[key[:index]])
        if tail_value is None:
            value_history.append(value_by_key[key])
        else:
            value_history.append(tail_value)
        return value_history

    def _improving_result_key_and_history(self, value_by_key, key, root_value):
        if key in value_by_key and value_by_key[key] > root_value + 1.0e-4:
            return key, self._build_value_history(value_by_key, key)
        best_key = ()
        best_value = root_value
        for candidate_key, candidate_value in value_by_key.items():
            if candidate_value > best_value + 1.0e-4:
                best_key = candidate_key
                best_value = candidate_value
        return best_key, self._build_value_history(value_by_key, best_key)
