"""Search3 / PUCT engine implementation."""

import numpy as np

from model.search_result import SearchResult


def sigmoid(x):
    """Scalar sigmoid for Search3 value normalization."""
    return 1 / (1 + np.exp(-x))


def softmax_1d(x, temperature=1.0):
    temperature = max(float(temperature), 1.0e-6)
    scaled = np.array(x, dtype='f') / temperature
    if not np.any(np.isfinite(scaled)):
        return np.ones_like(scaled, dtype='f') / len(scaled)
    scaled[~np.isfinite(scaled)] = -np.inf
    shifted = scaled - np.max(scaled)
    exp_x = np.exp(shifted)
    total = np.sum(exp_x)
    if total <= 0:
        return np.ones_like(x, dtype='f') / len(x)
    return exp_x / total


class Search3Engine:
    def __init__(self, ai):
        self.ai = ai
        self.cube = ai.cube
        self.ips = ai.ips
        self.root_node = None
        self.root_state_key = None
        self.node_cache = {}
        self.prediction_cache = {}
        self.last_move = None
    
    def reset_tree(self):
        self.root_node = None
        self.root_state_key = None
        self.node_cache = {}
        self.prediction_cache = {}
        self.last_move = None

    def prune_caches(self):
        """Keep long-running Search3 caches bounded."""
        self._prune_prediction_cache()
        self._prune_node_cache()

    def _cache_limit(self, attr_name, default_value):
        return max(0,int(getattr(self.ai,attr_name,default_value)))

    def _prune_prediction_cache(self):
        limit = self._cache_limit('search3_max_prediction_cache',5000)
        if limit <= 0 or len(self.prediction_cache) <= limit:
            return
        for state_key in list(self.prediction_cache.keys())[:len(self.prediction_cache) - limit]:
            del self.prediction_cache[state_key]

    def _prune_node_cache(self):
        limit = self._cache_limit('search3_max_node_cache',5000)
        if limit <= 0 or len(self.node_cache) <= limit:
            return
        if self.root_node is None:
            self.node_cache = {}
            return
        self.root_node.children.clear()
        self.node_cache = {self.root_node.state_key: self.root_node}

    def _state_key(self):
        return ''.join(self.cube.state)

    def _sync_root(self):
        state_key = self._state_key()
        if self.root_state_key != state_key:
            self.root_node = self.node_cache.get(state_key)
            self.root_state_key = state_key

    def _predict_current_state(self):
        if hasattr(self.ai, '_predict_state_batch'):
            return self._predict_feature(None, self._state_key())
        return self._predict_feature(self.cube.makedata().reshape(-1), self._state_key())

    def _predict_feature(self, feature, state_key):
        cached_prediction = self.prediction_cache.get(state_key)
        if cached_prediction is not None:
            return cached_prediction
        if hasattr(self.ai, '_predict_state_batch'):
            prediction = self.ai._predict_state_batch([self._snapshot_state()])
            self.prediction_cache[state_key] = prediction
            return prediction
        prediction = self.ai._predict_search2(feature.reshape(-1, 1))
        self.prediction_cache[state_key] = prediction
        return prediction

    def _predict_leaf_batch(self, pending_leaves):
        if len(pending_leaves) == 0:
            return
        if hasattr(self.ai, '_predict_state_batch'):
            predictions = self._transformer_leaf_predictions(pending_leaves)
        else:
            features = np.zeros((self.ips, len(pending_leaves)))
            for index, pending_leaf in enumerate(pending_leaves):
                features[:, index] = pending_leaf['feature']
            predictions = self.ai._predict_search2(features)
        for index, pending_leaf in enumerate(pending_leaves):
            state_key = pending_leaf['state_key']
            prediction = predictions[:, index:index + 1]
            self.prediction_cache[state_key] = prediction
            node_P = self._node_policy_from_prediction(prediction)
            node_V = prediction[-1, 0]
            child_node = self.node_cache.get(state_key)
            if child_node is None:
                child_node = self._create_node(state_key, node_P, node_V)
            else:
                child_node.P[:] = node_P
                child_node.value = node_V
            pending_leaf['parent_node'].children[pending_leaf['move_label']] = child_node
            pending_leaf['path_nodes'].append(child_node)
            pending_leaf['node_value'] = sigmoid(node_V)

    def _transformer_leaf_predictions(self, pending_leaves):
        predictions = np.zeros((len(self.cube.move_keys) + 1, len(pending_leaves)), dtype='f')
        uncached_snapshots = []
        uncached_keys = []
        key_to_columns = {}

        for column_index, pending_leaf in enumerate(pending_leaves):
            state_key = pending_leaf['state_key']
            cached_prediction = self.prediction_cache.get(state_key)
            if cached_prediction is not None:
                predictions[:, column_index:column_index + 1] = cached_prediction
                continue
            if state_key not in key_to_columns:
                key_to_columns[state_key] = []
                uncached_keys.append(state_key)
                uncached_snapshots.append(pending_leaf['state_snapshot'])
            key_to_columns[state_key].append(column_index)

        if len(uncached_snapshots) > 0:
            uncached_predictions = self.ai._predict_state_batch(uncached_snapshots)
            for prediction_index, state_key in enumerate(uncached_keys):
                prediction = uncached_predictions[:, prediction_index:prediction_index + 1]
                self.prediction_cache[state_key] = prediction
                for column_index in key_to_columns[state_key]:
                    predictions[:, column_index:column_index + 1] = prediction

        return predictions

    def _snapshot_state(self):
        if hasattr(self.cube, '_snapshot'):
            return self.cube._snapshot()
        return self.cube.state.copy()

    def _create_node(self, state_key, node_P, node_value=None, C=None):
        node_C = self.ai.search3_C if C is None else C
        node = Node(node_P, state_key, node_value, C=node_C)
        self.node_cache[state_key] = node
        return node

    def _create_node_from_current_state(self, C=0.05):
        state_key = self._state_key()
        node_PV = self._predict_current_state()
        node_P = self._node_policy_from_prediction(node_PV)
        node_V = node_PV[-1, 0]
        node = self.node_cache.get(state_key)
        if node is None:
            node = self._create_node(state_key, node_P, node_V, C=C)
        else:
            node.value = node_V
        return node, node_V

    def _node_policy_from_prediction(self, prediction):
        policy = prediction[:-1].reshape(-1)
        temperature = getattr(self.ai, 'policy_temperature', 1.0)
        return softmax_1d(policy, temperature=temperature).astype('f')

    def _find_invalid_indices(self, path_moves):
        invalid_indices = set([])
        if len(path_moves) > 0:
            last_move = path_moves[-1]
        else:
            last_move = self.last_move
        if last_move is not None:
            inverse_moves = self.cube.invert_str(last_move)
            if not isinstance(inverse_moves, tuple):
                inverse_moves = (inverse_moves,)
            for inverse_move in inverse_moves:
                if inverse_move in self.cube.key_to_num:
                    invalid_indices.add(self.cube.key_to_num[inverse_move])
        if hasattr(self.cube, 'illegal_move_indices'):
            invalid_indices.update(int(index) for index in self.cube.illegal_move_indices())
        return invalid_indices

    def _ensure_root_node(self, C=0.05):
        self._sync_root()
        root_prediction = self._predict_current_state()
        root_value = root_prediction[-1, 0]
        if self.root_node is None:
            root_P = self._node_policy_from_prediction(root_prediction)
            self.root_node = self._create_node(self._state_key(), root_P, root_value, C=C)
            self.root_state_key = self._state_key()
        return root_value

    def _evaluate_move_sequence(self, moves):
        moves = self._legal_prefix(moves)
        if len(moves) == 0:
            return sigmoid(self._predict_current_state()[-1, 0])
        for move in moves:
            self.cube.make_move(move)
        leaf_value = sigmoid(self._predict_current_state()[-1, 0])
        for move in self.cube.invert_moves(tuple(moves)):
            self.cube.make_move(move)
        return leaf_value

    def _value_trace_for_moves(self, root_value, moves):
        moves = self._legal_prefix(moves)
        value_trace = [root_value]
        if len(moves) == 0:
            return value_trace
        for move in moves:
            self.cube.make_move(move)
            value_trace.append(sigmoid(self._predict_current_state()[-1, 0]))
        for move in self.cube.invert_moves(tuple(moves)):
            self.cube.make_move(move)
        return value_trace

    def _raw_value_trace_for_moves(self, root_value_raw, moves):
        moves = self._legal_prefix(moves)
        value_trace = [root_value_raw]
        if len(moves) == 0:
            return value_trace
        for move in moves:
            self.cube.make_move(move)
            value_trace.append(self._predict_current_state()[-1, 0])
        for move in self.cube.invert_moves(tuple(moves)):
            self.cube.make_move(move)
        return value_trace

    def _legal_prefix(self, moves):
        if not hasattr(self.cube, 'is_legal_move'):
            return tuple(moves)
        legal_moves = []
        applied_moves = []
        try:
            for move in tuple(moves):
                if not self.cube.is_legal_move(move):
                    break
                self.cube.make_move(move)
                legal_moves.append(move)
                applied_moves.append(move)
        finally:
            for move in self.cube.invert_moves(tuple(applied_moves)):
                self.cube.make_move(move)
        return tuple(legal_moves)

    def advance_root(self, move):
        state_key = self._state_key()
        child_node = None
        if self.root_node is not None and move in self.root_node.children:
            child_node = self.root_node.children[move]
            if child_node.state_key != state_key:
                child_node = None
        if child_node is None:
            child_node = self.node_cache.get(state_key)
        self.root_node = child_node
        self.root_state_key = state_key
        self.last_move = move

    def _backup_path(self, path_nodes, path_indices, node_value):
        gamma = float(getattr(self.ai, 'value_target_gamma', 1.0))
        backed_value = float(node_value)
        for depth in range(len(path_indices) - 1, -1, -1):
            backed_value *= gamma
            parent_node = path_nodes[depth]
            parent_node.val[path_indices[depth]] += backed_value

    def _revert_path(self, path_moves):
        for move in self.cube.invert_moves(tuple(path_moves)):
            self.cube.make_move(move)

    def _collect_leaf_path(self, root_node, max_depth):
        node = root_node
        path_nodes = [root_node]
        path_indices = []
        path_moves = []
        path_state_keys = {root_node.state_key}
        while True:
            index = node.select_node(self._find_invalid_indices(path_moves))
            if index is None:
                return {
                    'resolved': True,
                    'solved_move': None,
                    'path_nodes': path_nodes,
                    'path_indices': path_indices,
                    'path_moves': path_moves,
                    'node_value': 0.0,
                }
            move_label = self.cube.move_keys[index]
            path_indices.append(index)
            path_moves.append(move_label)
            node.visited[index] += 1
            node.S += 1
            if hasattr(self.cube, 'is_legal_move') and not self.cube.is_legal_move(move_label):
                node.blocked[index] = True
                path_moves.pop()
                path_indices.pop()
                return {
                    'resolved': True,
                    'solved_move': None,
                    'path_nodes': path_nodes,
                    'path_indices': path_indices,
                    'path_moves': path_moves,
                    'node_value': 0.0,
                }
            try:
                self.cube.make_move(move_label)
            except Exception:
                node.blocked[index] = True
                path_moves.pop()
                path_indices.pop()
                return {
                    'resolved': True,
                    'solved_move': None,
                    'path_nodes': path_nodes,
                    'path_indices': path_indices,
                    'path_moves': path_moves,
                    'node_value': 0.0,
                }
            if self.cube.is_perfect():
                return {
                    'resolved': True,
                    'solved_move': tuple(path_moves),
                    'path_nodes': path_nodes,
                    'path_indices': path_indices,
                    'path_moves': path_moves,
                    'node_value': 1.0,
                }

            state_key = self._state_key()
            if state_key in path_state_keys:
                node.blocked[index] = True
                return {
                    'resolved': True,
                    'solved_move': None,
                    'path_nodes': path_nodes,
                    'path_indices': path_indices,
                    'path_moves': path_moves,
                    'node_value': 0.0,
                }
            if len(path_moves) >= max_depth:
                return {
                    'resolved': False,
                    'parent_node': node,
                    'move_label': move_label,
                    'state_key': state_key,
                    'feature': self._leaf_feature(),
                    'state_snapshot': self._snapshot_state(),
                    'path_nodes': path_nodes,
                    'path_indices': path_indices,
                    'path_moves': path_moves,
                }
            child_node = node.children.get(move_label)
            if child_node is None or child_node.state_key != state_key:
                child_node = self.node_cache.get(state_key)
                if child_node is not None:
                    node.children[move_label] = child_node
            if child_node is None:
                return {
                    'resolved': False,
                    'parent_node': node,
                    'move_label': move_label,
                    'state_key': state_key,
                    'feature': self._leaf_feature(),
                    'state_snapshot': self._snapshot_state(),
                    'path_nodes': path_nodes,
                    'path_indices': path_indices,
                    'path_moves': path_moves,
                }
            path_state_keys.add(state_key)
            path_nodes.append(child_node)
            node = child_node

    def _principal_variation(self, root_node):
        pv_moves = []
        node = root_node
        visited_states = set([])
        while node is not None and node.S > 0 and node.state_key not in visited_states:
            visited_states.add(node.state_key)
            best_index = int(np.argmax(node.visited))
            if node.visited[best_index] == 0:
                break
            best_move = self.cube.move_keys[best_index]
            if best_move not in node.children:
                break
            pv_moves.append(best_move)
            node = node.children[best_move]
        return tuple(pv_moves)

    def _leaf_feature(self):
        if hasattr(self.ai, '_predict_state_batch'):
            return None
        return self.cube.makedata().reshape(-1)

    def search3(self, N, C=0.05):
        root_value_raw = self._ensure_root_node(C=C)
        root_value = sigmoid(root_value_raw)
        root_node = self.root_node
        solved_move = None
        remaining_playouts = N
        batch_size = min(getattr(self.ai, 'search_batch3', 32), N)
        max_depth = max(1, getattr(self.ai, 'search_depth3', 100))
        start_root_visits = root_node.visited.copy()
        total_playouts = 0

        while remaining_playouts > 0 and solved_move is None:
            current_batch_size = min(batch_size, remaining_playouts)
            pending_leaves = []
            processed_playouts = 0
            for _ in range(current_batch_size):
                leaf_path = self._collect_leaf_path(root_node, max_depth)
                processed_playouts += 1
                self._revert_path(leaf_path['path_moves'])
                if leaf_path['resolved']:
                    self._backup_path(leaf_path['path_nodes'], leaf_path['path_indices'], leaf_path['node_value'])
                    if leaf_path.get('solved_move') is not None:
                        solved_move = leaf_path['solved_move']
                        break
                else:
                    pending_leaves.append(leaf_path)

            self._predict_leaf_batch(pending_leaves)
            for pending_leaf in pending_leaves:
                self._backup_path(pending_leaf['path_nodes'], pending_leaf['path_indices'], pending_leaf['node_value'])

            total_playouts += processed_playouts
            remaining_playouts -= processed_playouts

        local_visits = root_node.visited - start_root_visits
        stats = np.array([np.max(local_visits), total_playouts], dtype='i')

        if solved_move is not None:
            best_moves = self._legal_prefix(solved_move)
            best_value = 1.0
            raw_value_trace = self._raw_value_trace_for_moves(root_value_raw, best_moves)
            raw_best_value = raw_value_trace[-1]
            value_trace = self._value_trace_for_moves(root_value, best_moves)
            if len(value_trace) > 0:
                value_trace[-1] = best_value
            result = SearchResult(True, best_moves, root_value, value_trace, best_value, stats, local_visits.copy(), self.ai.search_mode, 'solved', root_value_raw=root_value_raw, value_trace_raw=raw_value_trace, best_value_raw=raw_best_value)
            self.prune_caches()
            return result

        best_moves = self._legal_prefix(self._principal_variation(root_node))
        if len(best_moves) == 0:
            fallback_move = self.cube.move_keys[np.argmax(root_node.visited)]
            best_moves = (fallback_move,) if not hasattr(self.cube, 'is_legal_move') or self.cube.is_legal_move(fallback_move) else tuple([])
        best_moves = self._bounded_budget_moves(root_value, best_moves)
        raw_value_trace = self._raw_value_trace_for_moves(root_value_raw, best_moves)
        best_value_raw = raw_value_trace[-1]
        best_value = sigmoid(best_value_raw)
        value_trace = self._value_trace_for_moves(root_value, best_moves)
        result = SearchResult(False, best_moves, root_value, value_trace, best_value, stats, local_visits.copy(), self.ai.search_mode, 'budget', root_value_raw=root_value_raw, value_trace_raw=raw_value_trace, best_value_raw=best_value_raw)
        self.prune_caches()
        return result

    def _bounded_budget_moves(self, root_value, moves):
        if not getattr(self.ai, 'is_transformer', False) or len(moves) <= 1:
            return moves
        max_return_moves = max(1, int(getattr(self.ai, 'max_return_moves', len(moves))))
        capped_moves = tuple(moves[:max_return_moves])
        value_trace = self._value_trace_for_moves(root_value, capped_moves)
        if len(value_trace) <= 1:
            return capped_moves
        best_index = int(np.argmax(value_trace))
        min_improvement = float(getattr(self.ai, 'min_value_improvement', 0.0))
        if best_index <= 0 or value_trace[best_index] < root_value + min_improvement:
            return tuple(capped_moves[:1])
        return tuple(capped_moves[:best_index])


class Node:
    def __init__(self, P, state_key=None, value=None, C=0.05):
        self.P = P
        self.state_key = state_key
        self.value = sigmoid(value)
        self.val = np.zeros_like(P, dtype='f')
        self.visited = np.zeros_like(P, dtype='i')
        self.blocked = np.zeros_like(P, dtype=bool)
        self.S = 0
        self.C = C
        self.children = {}
        self.score = None

    def select_node(self, invalid_index=None):
        masked = self.blocked.copy()
        if invalid_index is not None:
            if isinstance(invalid_index, (set, list, tuple, np.ndarray)):
                for index in invalid_index:
                    masked[index] = True
            else:
                masked[invalid_index] = True
        if np.all(masked):
            return None
        average_value = np.full_like(self.P, self.value, dtype='f')
        np.divide(self.val, self.visited, out=average_value, where=self.visited != 0)
        self.score = average_value + self.C * self.P * np.sqrt(max(1, self.S)) / (1 + self.visited)
        self.score[masked] = -np.inf
        if self.S == 0:
            prior = self.P.copy()
            prior[masked] = -np.inf
            return np.argmax(prior)
        return np.argmax(self.score)
