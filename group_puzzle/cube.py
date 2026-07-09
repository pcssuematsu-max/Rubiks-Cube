"""Generic finite group puzzle models compatible with the Rubiks AI interface.

States use one-line notation for symmetric groups and row-major flattened
matrices for linear groups.  Every move acts on the left, so applying ``x``
to a state ``g`` produces ``x * g``.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping

import numpy as np


def _is_prime(value):
    value = int(value)
    if value < 2:
        return False
    divisor = 2
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 1
    return True


def _matrix_inverse_mod(matrix, modulus):
    """Return a matrix inverse over the prime field F_modulus."""
    matrix = np.asarray(matrix, dtype=int) % modulus
    size = matrix.shape[0]
    augmented = np.concatenate([matrix.copy(), np.eye(size, dtype=int)], axis=1)
    for column in range(size):
        pivot_rows = np.where(augmented[column:, column] % modulus != 0)[0]
        if pivot_rows.size == 0:
            raise ValueError(f"matrix is singular over F_{modulus}: {matrix.tolist()}")
        pivot = column + int(pivot_rows[0])
        if pivot != column:
            augmented[[column, pivot]] = augmented[[pivot, column]]
        augmented[column] = (augmented[column] * pow(int(augmented[column, column]), -1, modulus)) % modulus
        for row in range(size):
            if row == column:
                continue
            augmented[row] = (augmented[row] - augmented[row, column] * augmented[column]) % modulus
    return augmented[:, size:] % modulus


def _matrix_determinant_mod(matrix, modulus):
    """Return det(matrix) in the prime field F_modulus."""
    work = np.asarray(matrix, dtype=int).copy() % modulus
    determinant = 1
    for column in range(work.shape[0]):
        pivot_rows = np.where(work[column:, column] != 0)[0]
        if pivot_rows.size == 0:
            return 0
        pivot = column + int(pivot_rows[0])
        if pivot != column:
            work[[column, pivot]] = work[[pivot, column]]
            determinant = -determinant
        pivot_value = int(work[column, column])
        determinant = determinant * pivot_value % modulus
        pivot_inverse = pow(pivot_value, -1, modulus)
        for row in range(column + 1, work.shape[0]):
            factor = int(work[row, column]) * pivot_inverse % modulus
            work[row] = (work[row] - factor * work[column]) % modulus
    return determinant % modulus


class GroupPuzzle:
    """Base implementation shared by symmetric and linear group puzzles."""

    group_kind = "group"

    def __init__(self, generators, auto_add_inverses=False, display_name=None):
        if not isinstance(generators, Mapping) or not generators:
            raise ValueError("group_generators must be a non-empty mapping of move labels to elements")

        normalized_generators = {
            str(label): self._normalize_generator(element)
            for label, element in generators.items()
        }
        if len(normalized_generators) != len(generators):
            raise ValueError("generator labels must be unique after conversion to strings")
        self.generators = self._with_inverse_generators(normalized_generators) if auto_add_inverses else normalized_generators
        self.auto_add_inverses = bool(auto_add_inverses)
        self.display_name = display_name

        self.move_keys = tuple(self.generators.keys())
        self.move_len = len(self.move_keys)
        self.key_to_num = {key: index for index, key in enumerate(self.move_keys)}
        self.inverse_move = self._build_inverse_move_map()
        self.parameter_namespace = self._parameter_namespace()
        self.move = self.generators
        self.state_0 = self._identity_state()
        self.state = self.state_0.copy()
        self.state_size = int(self.state.size)
        self.surface_num = self.state_size

        # No algorithms are registered as myperms for a generic group puzzle.
        self.myperms = {}
        self.myperms2 = {}
        self.myperms_sequence_group = {}
        self.single_and_rotate = []
        self.transformation_keys = (0,)
        self.tf_invert = {0: 0}

        self._init_feature_layout()
        self.perfect_data = self.makedata()
        self._init_group_values()
        self._init_scramble_registry()

    def _with_inverse_generators(self, generators):
        """Return X with missing inverse elements inserted next to their generators."""
        expanded = {}
        original_elements = tuple(generators.values())
        for label, element in generators.items():
            expanded[label] = element
            inverse = self._inverse_element(element)
            if any(np.array_equal(candidate, inverse) for candidate in original_elements):
                continue
            inverse_label = self._available_inverse_label(label, expanded, generators)
            expanded[inverse_label] = inverse
        return expanded

    def _available_inverse_label(self, label, expanded, original_generators):
        candidate = f"{label}^-1"
        suffix = 2
        while candidate in expanded or candidate in original_generators:
            candidate = f"{label}^-{suffix}"
            suffix += 1
        return candidate

    def _parameter_namespace(self):
        """Stable directory key so group models do not overwrite cube weights."""
        digest = hashlib.sha256()
        digest.update(self.group_kind.encode("utf-8"))
        for label, element in self.generators.items():
            digest.update(label.encode("utf-8"))
            array = np.asarray(element, dtype=np.int64)
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
        if self.group_kind == "linear":
            descriptor = f"linear_d{self.dimension}_p{self.modulus}"
        else:
            descriptor = f"symmetric_n{self.degree}"
        return f"{descriptor}_{digest.hexdigest()[:12]}"

    def _normalize_generator(self, element):
        raise NotImplementedError

    def _identity_state(self):
        raise NotImplementedError

    def _inverse_element(self, element):
        raise NotImplementedError

    def _left_multiply(self, element, state):
        raise NotImplementedError

    def _feature_value_count(self):
        raise NotImplementedError

    def _feature_index(self, value):
        return int(value)

    def _feature_value(self, feature_index):
        return int(feature_index)

    def _group_name(self):
        raise NotImplementedError

    def _build_inverse_move_map(self):
        inverse_move = {}
        for label, element in self.generators.items():
            inverse = self._inverse_element(element)
            matches = [
                candidate_label
                for candidate_label, candidate in self.generators.items()
                if np.array_equal(candidate, inverse)
            ]
            if not matches:
                raise ValueError(
                    f"generator set X is not inverse-closed: inverse of {label!r} is missing"
                )
            inverse_move[label] = matches[0]
        return inverse_move

    def _init_feature_layout(self):
        value_count = self._feature_value_count()
        group_name = self._group_name()
        pieces = [(index,) for index in range(self.state_size)]
        self.group_pieces = {group_name: pieces}
        self.group_indices = {group_name: list(range(self.state_size))}
        self.piece_feature_offsets = {}
        self.feature_index_to_piece_color = {}
        for index, piece in enumerate(pieces):
            offset = index * value_count
            self.piece_feature_offsets[piece] = (offset, value_count)
            for feature_index in range(value_count):
                value = self._feature_value(feature_index)
                self.feature_index_to_piece_color[offset + feature_index] = (piece, (value,))
        self.ips = self.state_size * value_count

    def _init_group_values(self):
        group_name = self._group_name()
        solved_features = self.perfect_data.reshape(1, -1)
        self.group_val = {group_name: solved_features.copy()}
        self.total_val = {group_name: float(self.state_size)}

    def _init_scramble_registry(self):
        self.my_scrambles = []
        self.my_scrambles2 = {0: {move: set() for move in self.move_keys}}
        self.my_scramble_changed_piece_keys = {0: {}}
        self.counter = {0: {}}
        self.piece_color_counter = {}

    def _group_name_map(self):
        return {"A": self._group_name()}

    def create_new_set(self):
        level = len(self.my_scrambles2)
        self.my_scrambles2[level] = {move: set() for move in self.move_keys}
        self.my_scramble_changed_piece_keys[level] = {}
        self.counter[level] = {}

    def register_scramble_sequence(self, level, moves):
        """Keep compatibility metadata; random scrambles never consume it."""
        moves = self.normalize_move_sequence(moves)
        if not moves:
            return
        while level not in self.my_scrambles2:
            self.create_new_set()
        self.my_scrambles2[level][moves[-1]].add(moves)
        self.my_scramble_changed_piece_keys[level][moves] = tuple(
            self.get_chenged_pieces_keys_from_moves(moves)
        )

    def get_registered_scramble_changed_piece_keys(self, level, moves):
        return self.my_scramble_changed_piece_keys.get(level, {}).get(self.normalize_move_sequence(moves), ())

    def normalize_move_key(self, move):
        if isinstance(move, (tuple, list)) and len(move) == 1:
            move = move[0]
        move = str(move)
        if move not in self.generators:
            raise KeyError(move)
        return move

    def normalize_move_sequence(self, moves):
        return tuple(self.normalize_move_key(move) for move in moves)

    def format_move(self, move):
        return self.normalize_move_key(move)

    def format_moves(self, moves):
        return tuple(self.format_move(move) for move in moves)

    def make_move(self, key):
        key = self.normalize_move_key(key)
        self.state = self._left_multiply(self.generators[key], self.state)

    def scramble(self, N, Move=None, difficult_mode=False, scramble_mode=None,
                 flip=None, rotate=None, swap=False, add_moves=None,
                 transform_N=None, flip_inside=None, move_count_policy="prefer_rare"):
        """Apply an exact-length word in X; the construction word is not retained by the state."""
        if Move is None:
            moves = self._random_reduced_word(max(0, int(N)))
        else:
            moves = self.normalize_move_sequence(Move)
        for move in moves:
            self.make_move(move)
        return moves

    def _random_reduced_word(self, length):
        """Generate a word with no adjacent generator/inverse cancellation."""
        if length == 0:
            return ()
        moves = [random.choice(self.move_keys)]
        while len(moves) < length:
            forbidden_move = self.invert_str(moves[-1])
            candidates = [move for move in self.move_keys if move != forbidden_move]
            if not candidates:
                raise ValueError(
                    "cannot generate the requested scramble length without adjacent inverses; "
                    "add another generator to X"
                )
            moves.append(random.choice(candidates))
        return tuple(moves)

    def makedata(self):
        value_count = self._feature_value_count()
        data = np.zeros(self.ips, dtype="f")
        for position, value in enumerate(self.state):
            data[position * value_count + self._feature_index(value)] = 1.0
        return data

    def reset(self):
        self.state = self.state_0.copy()

    def is_perfect(self):
        return bool(np.array_equal(self.state, self.state_0))

    def state_to_str(self):
        return ",".join(str(int(value)) for value in self.state)

    def last_perms_changed_number(self):
        """Number of coordinates differing from the identity, used for reports."""
        return int(np.count_nonzero(self.state != self.state_0))

    def last_perms_key(self):
        """Return an exact, human-readable name for the current group element."""
        raise NotImplementedError

    def set_state(self, state):
        values = np.asarray(state, dtype=int).reshape(-1)
        if values.size != self.state_size:
            raise ValueError(f"expected {self.state_size} state values, got {values.size}")
        self._validate_state(values)
        self.state = values.copy()

    def _validate_state(self, state):
        raise NotImplementedError

    def invert_str(self, move):
        return self.inverse_move[self.normalize_move_key(move)]

    def invert_moves(self, moves):
        return tuple(self.invert_str(move) for move in self.normalize_move_sequence(moves)[::-1])

    def simplify(self, moves):
        simplified = []
        for move in self.normalize_move_sequence(moves):
            if simplified and self.invert_str(move) == simplified[-1]:
                simplified.pop()
            else:
                simplified.append(move)
        return tuple(simplified)

    def reduce(self, moves):
        moves = self.normalize_move_sequence(moves)
        original_state = self.state.copy()
        reduced = []
        kept_indices = []
        visited = {self.state.tobytes(): 0}
        try:
            for original_index, move in enumerate(moves):
                self.make_move(move)
                state_key = self.state.tobytes()
                if state_key in visited:
                    keep_count = visited[state_key]
                    reduced = reduced[:keep_count]
                    kept_indices = kept_indices[:keep_count]
                    visited = {original_state.tobytes(): 0}
                    replay_state = original_state.copy()
                    for replay_index, replay_move in enumerate(reduced, 1):
                        replay_state = self._left_multiply(self.generators[replay_move], replay_state)
                        visited[replay_state.tobytes()] = replay_index
                else:
                    reduced.append(move)
                    kept_indices.append(original_index)
                    visited[state_key] = len(reduced)
        finally:
            self.state = original_state
        return tuple(reduced), kept_indices

    def conjugate(self, A, B):
        return self.simplify(tuple(A) + tuple(B) + self.invert_moves(A))

    def commutator(self, A, B):
        return self.simplify(tuple(A) + tuple(B) + self.invert_moves(A) + self.invert_moves(B))

    def transform(self, moves, index, flip_inside=False, invert=False):
        return self.normalize_move_sequence(moves)

    def make_transformations(self, scramble, moves):
        return [self.normalize_move_sequence(scramble)], [self.normalize_move_sequence(moves)]

    def flip_inside_moves(self, moves):
        return self.normalize_move_sequence(moves)

    def collect_single_move_and_rotate(self):
        return []

    def collect_single_moves_and_rotate(self):
        return []

    def get_chenged_pieces_keys_from_moves(self, moves):
        original_state = self.state.copy()
        try:
            self.reset()
            for move in self.normalize_move_sequence(moves):
                self.make_move(move)
            return [
                (self._group_name(), (index,))
                for index in range(self.state_size)
                if self.state[index] != self.state_0[index]
            ]
        finally:
            self.state = original_state

    def get_correct_group_count(self, group_name):
        if group_name != self._group_name():
            return 0
        return int(np.count_nonzero(self.state == self.state_0))

    def get_correct_group_index(self, group_name):
        if group_name != self._group_name():
            return []
        return np.where(self.state == self.state_0)[0].astype(int).tolist()

    def piece_display_name(self, piece_type, piece):
        return self.coordinate_label(piece[0])

    def _embedding_position(self, feature_index):
        return int(feature_index) // self._feature_value_count()

    def embedding_feature_value_label(self, feature_index):
        local_index = int(feature_index) % self._feature_value_count()
        return f"value={self._feature_value(local_index)}"

    def embedding_piece_type(self, feature_index):
        return self._group_name()

    def embedding_solve_group(self, feature_index):
        return self._group_name()

    def coordinate_label(self, index):
        return str(index)


class SymmetricGroupPuzzle(GroupPuzzle):
    """Puzzle on a subgroup of S_n using 1-based one-line permutations."""

    group_kind = "symmetric"

    def __init__(self, degree=3, generators=None, auto_add_inverses=False, display_name=None):
        self.degree = int(degree)
        if self.degree < 2:
            raise ValueError("symmetric group degree must be at least 2")
        self.size = self.degree
        self.order = self.degree
        if generators is None:
            cycle = np.roll(np.arange(1, self.degree + 1), -1)
            cycle_inverse = np.roll(np.arange(1, self.degree + 1), 1)
            transposition = np.arange(1, self.degree + 1)
            transposition[:2] = (2, 1)
            generators = {"s": transposition}
            if not np.array_equal(cycle, transposition):
                generators["r"] = cycle
            if not any(np.array_equal(cycle_inverse, value) for value in generators.values()):
                generators["r^-1"] = cycle_inverse
        super().__init__(generators, auto_add_inverses=auto_add_inverses, display_name=display_name)

    def _normalize_generator(self, element):
        permutation = np.asarray(element, dtype=int).reshape(-1)
        if permutation.size != self.degree or set(permutation.tolist()) != set(range(1, self.degree + 1)):
            raise ValueError(
                f"each symmetric generator must be a permutation of 1..{self.degree}: {permutation.tolist()}"
            )
        return permutation.copy()

    def _identity_state(self):
        return np.arange(1, self.degree + 1, dtype=int)

    def _inverse_element(self, element):
        inverse = np.empty(self.degree, dtype=int)
        inverse[element - 1] = np.arange(1, self.degree + 1)
        return inverse

    def _left_multiply(self, element, state):
        return element[np.asarray(state, dtype=int) - 1].copy()

    def _feature_value_count(self):
        return self.degree

    def _feature_index(self, value):
        return int(value) - 1

    def _feature_value(self, feature_index):
        return int(feature_index) + 1

    def _group_name(self):
        return "Position"

    def _validate_state(self, state):
        if set(state.tolist()) != set(range(1, self.degree + 1)):
            raise ValueError(f"state must be a permutation of 1..{self.degree}")

    def coordinate_label(self, index):
        return f"position {index + 1}"

    def last_perms_key(self):
        values = ",".join(str(int(value)) for value in self.state)
        group_name = self.display_name or f"S_{self.degree}"
        return f"{group_name}:[{values}]"

    def embedding_feature_value_label(self, feature_index):
        local_index = int(feature_index) % self._feature_value_count()
        return f"contains={self._feature_value(local_index)}"

    def embedding_piece_type(self, feature_index):
        return "Position"

    def embedding_solve_group(self, feature_index):
        return f"Position {self._embedding_position(feature_index) + 1}"


class LinearGroupPuzzle(GroupPuzzle):
    """Puzzle on a subgroup of GL(d, F_p), for prime p."""

    group_kind = "linear"

    def __init__(self, dimension=2, modulus=2, generators=None, family="GL",
                 auto_add_inverses=False, display_name=None):
        self.dimension = int(dimension)
        self.modulus = int(modulus)
        self.family = str(family).strip().upper()
        if self.dimension < 1:
            raise ValueError("linear group dimension must be positive")
        if not _is_prime(self.modulus):
            raise ValueError("this implementation supports prime finite fields F_p; modulus must be prime")
        if self.family not in ("GL", "SL"):
            raise ValueError("linear group family must be 'GL' or 'SL'")
        self.size = self.dimension
        self.order = self.dimension
        if generators is None:
            if self.dimension != 2:
                raise ValueError("custom inverse-closed generators are required when dimension is not 2")
            shear = np.array([[1, 1], [0, 1]], dtype=int) % self.modulus
            swap = np.array([[0, 1], [1, 0]], dtype=int) % self.modulus
            shear_inverse = _matrix_inverse_mod(shear, self.modulus)
            generators = {"A": shear, "B": swap}
            if not np.array_equal(shear_inverse, shear):
                generators["A^-1"] = shear_inverse
        super().__init__(generators, auto_add_inverses=auto_add_inverses, display_name=display_name)

    def _normalize_generator(self, element):
        matrix = np.asarray(element, dtype=int)
        expected = (self.dimension, self.dimension)
        if matrix.shape != expected:
            raise ValueError(f"each linear generator must have shape {expected}, got {matrix.shape}")
        matrix = matrix % self.modulus
        _matrix_inverse_mod(matrix, self.modulus)
        if self.family == "SL" and _matrix_determinant_mod(matrix, self.modulus) != 1:
            raise ValueError(
                f"SL generators must have determinant 1 over F_{self.modulus}: {matrix.tolist()}"
            )
        return matrix.copy()

    def _identity_state(self):
        return np.eye(self.dimension, dtype=int).reshape(-1)

    def _inverse_element(self, element):
        return _matrix_inverse_mod(element, self.modulus)

    def _left_multiply(self, element, state):
        matrix = np.asarray(state, dtype=int).reshape(self.dimension, self.dimension)
        return ((element @ matrix) % self.modulus).reshape(-1)

    def _feature_value_count(self):
        return self.modulus

    def _group_name(self):
        return "Entry"

    def _validate_state(self, state):
        matrix = state.reshape(self.dimension, self.dimension) % self.modulus
        _matrix_inverse_mod(matrix, self.modulus)

    def coordinate_label(self, index):
        row, column = divmod(index, self.dimension)
        return f"a{row}{column}"

    def last_perms_key(self):
        matrix = self.state.reshape(self.dimension, self.dimension)
        rows = ["[" + ",".join(str(int(value)) for value in row) + "]" for row in matrix]
        group_name = self.display_name or f"{self.family}_{self.dimension}(F_{self.modulus})"
        return f"{group_name}:[{','.join(rows)}]"

    def embedding_piece_type(self, feature_index):
        row, column = divmod(self._embedding_position(feature_index), self.dimension)
        return "Diagonal" if row == column else "OffDiagonal"

    def embedding_solve_group(self, feature_index):
        row, _ = divmod(self._embedding_position(feature_index), self.dimension)
        return f"Row {row}"


def create_group_puzzle(kind="symmetric", degree=3, dimension=2, modulus=2,
                        generators=None, family="GL", auto_add_inverses=False,
                        display_name=None):
    """Factory used by the Tk frame configuration."""
    normalized = str(kind).strip().lower().replace("-", "_")
    if normalized in ("symmetric", "permutation", "sn"):
        return SymmetricGroupPuzzle(
            degree=degree,
            generators=generators,
            auto_add_inverses=auto_add_inverses,
            display_name=display_name,
        )
    if normalized in ("linear", "gl", "matrix"):
        return LinearGroupPuzzle(
            dimension=dimension,
            modulus=modulus,
            generators=generators,
            family=family,
            auto_add_inverses=auto_add_inverses,
            display_name=display_name,
        )
    raise ValueError(f"unknown group_kind: {kind!r}")
