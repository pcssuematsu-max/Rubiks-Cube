"""Helpers for inspecting what a myperm changes on a solved cube."""

from collections import Counter

import numpy as np

from core.myperm_keys import format_myperm_key, resolve_myperm_key


class MypermAnalyzer:
    """Analyze myperms without depending on Frame or viewer classes."""

    def __init__(self, cube):
        self.cube = cube
        self.edge_index = tuple(getattr(self.cube, "edge_index", ()))
        self.corner_index = tuple(getattr(self.cube, "corner_index", ()))
        self.center_index = tuple(getattr(self.cube, "center_index", ()))
        self._edge_lookup = self._build_piece_lookup(self.edge_index)
        self._corner_lookup = self._build_piece_lookup(self.corner_index)

    def analyze(self, key_or_moves, use_inverse = True, labeled_centers = True):
        """Return piece-diff analysis for one myperm key or move tuple."""
        key, resolved_moves, applied_moves = self._resolved_analysis_moves(
            key_or_moves,
            use_inverse = use_inverse,
        )
        original_state = self.cube.state.copy()

        try:
            self.cube.reset()
            self.cube.scramble(0, applied_moves)
            edge_result = self._analyze_oriented_pieces(
                self.edge_index,
                self._edge_lookup,
                piece_type = "edge",
            )
            corner_result = self._analyze_oriented_pieces(
                self.corner_index,
                self._corner_lookup,
                piece_type = "corner",
            )
            center_result = self._analyze_centers()

            labeled_center_result = None
            if labeled_centers:
                labeled_center_result = self._analyze_labeled_centers(applied_moves)
        finally:
            self.cube.state = original_state.copy()

        return {
            "key": key,
            "display_key": format_myperm_key(key) if key is not None else None,
            "moves": tuple(resolved_moves),
            "display_moves": self._format_moves(resolved_moves),
            "applied_moves": tuple(applied_moves),
            "display_applied_moves": self._format_moves(applied_moves),
            "use_inverse": use_inverse,
            "edge": edge_result,
            "corner": corner_result,
            "center": center_result,
            "labeled_center": labeled_center_result,
        }

    def summarize(self, key_or_moves, use_inverse = True, labeled_centers = True):
        """Return a compact human-readable summary."""
        result = self.analyze(
            key_or_moves,
            use_inverse = use_inverse,
            labeled_centers = labeled_centers,
        )
        parts = []
        edge_summary = self._piece_summary("edge", result["edge"])
        corner_summary = self._piece_summary("corner", result["corner"])
        center_summary = self._center_summary(result["center"])
        labeled_center_summary = self._labeled_center_summary(result["labeled_center"])
        for summary in [edge_summary, corner_summary, center_summary, labeled_center_summary]:
            if summary:
                parts.append(summary)
        return ", ".join(parts)

    def analyze_inverse(self, key_or_moves, labeled_centers = True):
        """Analyze the inverse-applied effect on solved state."""
        return self.analyze(
            key_or_moves,
            use_inverse = True,
            labeled_centers = labeled_centers,
        )

    def _resolved_analysis_moves(self, key_or_moves, use_inverse = True):
        """Resolve original moves and the moves actually applied for analysis."""
        key, moves = self._resolve_moves(key_or_moves)
        applied_moves = self.cube.invert_moves(moves) if use_inverse else moves
        return key, tuple(moves), tuple(applied_moves)

    def _resolve_moves(self, key_or_moves):
        """Resolve either a myperm key or a raw move tuple."""
        if isinstance(key_or_moves, str):
            normalized_key = key_or_moves.strip()
            if normalized_key in self.cube.myperms2:
                return normalized_key, tuple(self.cube.myperms2[normalized_key])
            resolved_key = self._resolve_expanded_myperm_key(normalized_key)
            if resolved_key is not None:
                return resolved_key, tuple(self.cube.myperms[resolved_key])
            raise KeyError(key_or_moves)
        if isinstance(key_or_moves, tuple) and key_or_moves in self.cube.myperms:
            return key_or_moves, tuple(self.cube.myperms[key_or_moves])
        return None, tuple(key_or_moves)

    def _resolve_expanded_myperm_key(self, key_text):
        """Resolve display text or base key text to an expanded myperm tuple key."""
        return resolve_myperm_key(self.cube, key_text)

    def _format_moves(self, moves):
        """Return moves in puzzle-specific display notation when available."""
        if hasattr(self.cube, 'format_moves'):
            return tuple(self.cube.format_moves(moves))
        return tuple(moves)

    def _build_piece_lookup(self, piece_indices):
        """Map normalized sticker colors to the solved piece location."""
        lookup = {}
        for piece in piece_indices:
            lookup[self._normalize_piece_colors(self.cube.default_color[piece])] = piece
        return lookup

    def _analyze_oriented_pieces(self, piece_indices, lookup, piece_type):
        """Analyze moved edge/corner pieces including permutation and orientation."""
        moved = []
        permutation_map = {}
        orientation_count = 0

        for piece in piece_indices:
            current_colors = self._piece_colors(piece)
            solved_colors = self.cube.default_color[piece]
            if current_colors == solved_colors:
                continue

            source_piece = lookup[self._normalize_piece_colors(current_colors)]
            piece_entry = {
                "location": piece,
                "location_label": self._piece_label(piece),
                "source": source_piece,
                "source_label": self._piece_label(source_piece),
                "current_colors": current_colors,
                "solved_colors": solved_colors,
            }
            moved.append(piece_entry)

            if source_piece == piece:
                orientation_count += 1
            else:
                permutation_map[piece] = source_piece

        return {
            "piece_type": piece_type,
            "moved_count": len(moved),
            "moved": moved,
            "cycle_lengths": self._cycle_lengths(permutation_map),
            "orientation_count": orientation_count,
        }

    def _analyze_centers(self):
        """Analyze moved center stickers by location and solved color."""
        if not self.center_index:
            return {"available": False, "moved_count": 0, "moved": [], "solved_color_counts": {}}

        moved = []
        color_counter = Counter()
        for piece in self.center_index:
            current_colors = self._piece_colors(piece)
            solved_colors = self.cube.default_color[piece]
            if current_colors == solved_colors:
                continue
            moved.append(
                {
                    "location": piece,
                    "location_label": self._piece_label(piece),
                    "current_colors": current_colors,
                    "solved_colors": solved_colors,
                }
            )
            color_counter[solved_colors] += 1

        return {
            "moved_count": len(moved),
            "moved": moved,
            "solved_color_counts": dict(color_counter),
        }

    def _analyze_labeled_centers(self, applied_moves):
        """Track center motion with unique center labels so same-face moves are visible."""
        if not self.center_index:
            return None

        original_state = self.cube.state.copy()
        try:
            labeled_state, label_lookup = self._build_labeled_center_state()
            self.cube.state = labeled_state
            self.cube.scramble(0, applied_moves)
            permutation_map = {}
            moved = []
            for piece in self.center_index:
                label = self.cube.state[piece[0]]
                source_piece = label_lookup[label]
                if source_piece == piece:
                    continue
                permutation_map[piece] = source_piece
                moved.append(
                    {
                        "location": piece,
                        "location_label": self._center_location_label(piece),
                        "source": source_piece,
                        "source_label": self._center_location_label(source_piece),
                        "label": label,
                    }
                )
            return {
                "moved_count": len(moved),
                "moved": moved,
                "cycle_lengths": self._cycle_lengths(permutation_map),
            }
        finally:
            self.cube.state = original_state.copy()

    def _build_labeled_center_state(self):
        """Create an analysis-only solved state with unique labels on all center stickers."""
        labeled_state = np.array(self.cube.state_0, dtype = object)
        label_lookup = {}
        for center_index, piece in enumerate(self.center_index):
            label = f"C{center_index:03d}"
            labeled_state[piece[0]] = label
            label_lookup[label] = piece
        return labeled_state, label_lookup

    def _cycle_lengths(self, permutation_map):
        """Return cycle lengths for destination->source piece permutations."""
        cycle_lengths = []
        visited = set()
        for start in permutation_map:
            if start in visited:
                continue
            current = start
            cycle_length = 0
            while current not in visited and current in permutation_map:
                visited.add(current)
                current = permutation_map[current]
                cycle_length += 1
            if cycle_length > 0:
                cycle_lengths.append(cycle_length)
        cycle_lengths.sort()
        return cycle_lengths

    def _piece_colors(self, piece):
        """Read the current sticker colors for one piece location."""
        return "".join(str(self.cube.state[index]) for index in piece)

    def _normalize_piece_colors(self, colors):
        """Normalize oriented colors so the same physical piece compares equal."""
        return "".join(sorted(colors))

    def _piece_label(self, piece):
        """Use solved colors as a stable label for a piece location."""
        return self.cube.default_color[piece]

    def _center_location_label(self, piece):
        """Use solved color plus sticker index as a stable center location label."""
        return f"{self.cube.default_color[piece]}@{piece[0]}"

    def _piece_summary(self, piece_name, result):
        """Summarize moved count, cycles, and pure orientation changes."""
        if result["moved_count"] == 0:
            return ""
        parts = [f"{piece_name}:{result['moved_count']}"]
        if result["cycle_lengths"]:
            parts.append(f"cycles={result['cycle_lengths']}")
        if result["orientation_count"] > 0:
            parts.append(f"orientation_only={result['orientation_count']}")
        return " ".join(parts)

    def _center_summary(self, result):
        """Summarize moved centers by count and solved-color buckets."""
        if not result.get("available", True):
            return "center:n/a"
        if result["moved_count"] == 0:
            return ""
        return f"center:{result['moved_count']} colors={result['solved_color_counts']}"

    def _labeled_center_summary(self, result):
        """Summarize center permutations from uniquely-labeled center stickers."""
        if result is None or result["moved_count"] == 0:
            return ""
        parts = [f"center_perm:{result['moved_count']}"]
        if result["cycle_lengths"]:
            parts.append(f"cycles={result['cycle_lengths']}")
        return " ".join(parts)
