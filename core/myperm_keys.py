"""Canonical keys and names for puzzle myperm registries."""

from __future__ import annotations

from typing import NamedTuple


class MypermKey(NamedTuple):
    """A myperm algorithm name paired with its symmetry-transform index."""

    name: str
    transform_index: int


def normalize_myperm_name(name):
    """Return the canonical base name used by every puzzle's myperms dict."""
    if not isinstance(name, str):
        raise TypeError("myperm name must be a string")
    normalized = name.strip().rstrip("-")
    if not normalized:
        raise ValueError("myperm name must not be empty")
    if "#" in normalized:
        raise ValueError("myperm name must not contain the transform separator '#'")
    return normalized


def single_move_myperm_name(move_key):
    """Return the canonical base name for a one-move myperm."""
    move_name = str(move_key).strip()
    if not move_name:
        raise ValueError("move key must not be empty")
    return f"SingleMove-{move_name}"


def normalize_myperm_registry(registry):
    """Canonicalize registry names and reject normalization collisions."""
    normalized_registry = {}
    original_names = {}
    for name, moves in registry.items():
        normalized_name = normalize_myperm_name(name)
        if normalized_name in normalized_registry:
            previous = original_names[normalized_name]
            raise ValueError(
                f"myperm names {previous!r} and {name!r} normalize to "
                f"the same key {normalized_name!r}"
            )
        normalized_registry[normalized_name] = moves
        original_names[normalized_name] = name
    return normalized_registry


def make_myperm_key(base_key, transform_index):
    """Return a canonical expanded myperm key."""
    index = int(transform_index)
    if index < 0:
        raise ValueError("myperm transform index must be non-negative")
    return MypermKey(normalize_myperm_name(base_key), index)


def myperm_base_key(key):
    """Return the base algorithm name without guessing from trailing digits."""
    if isinstance(key, tuple) and len(key) == 2:
        return key[0]
    if isinstance(key, str) and "#" in key:
        base_key, separator, suffix = key.rpartition("#")
        if separator and suffix.isdigit():
            return normalize_myperm_name(base_key)
    return key


def myperm_transform_index(key):
    """Return a key's transform index, or None for an unexpanded base name."""
    if isinstance(key, tuple) and len(key) == 2:
        return int(key[1])
    if isinstance(key, str) and "#" in key:
        _, separator, suffix = key.rpartition("#")
        if separator and suffix.isdigit():
            return int(suffix)
    return None


def format_myperm_key(key):
    """Format a myperm key using an unambiguous transform separator."""
    transform_index = myperm_transform_index(key)
    if transform_index is None:
        return key
    return f"{myperm_base_key(key)}#{transform_index:02d}"


def resolve_myperm_key(cube, key, default_transform_index = 0):
    """Resolve a current key, base name, formatted key, or explicit key alias."""
    try:
        if key in cube.myperms:
            return key
    except TypeError:
        return None

    aliases = getattr(cube, "myperm_key_aliases", {})
    if key in aliases:
        return aliases[key]

    if not isinstance(key, str):
        return None

    transform_index = myperm_transform_index(key)
    base_name = myperm_base_key(key)
    if transform_index is None:
        transform_index = int(default_transform_index)

    name_aliases = getattr(cube, "myperm_name_aliases", {})
    base_name = name_aliases.get(base_name, base_name)
    candidate = make_myperm_key(base_name, transform_index)
    if candidate in cube.myperms:
        return candidate
    return None
