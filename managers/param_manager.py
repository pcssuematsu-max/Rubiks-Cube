"""Parameter save/load helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

class ParamManager:
    """AIパラメータの保存・読込と、対象AI indexの解釈を担当する。"""

    def __init__(self, frame):
        self.frame = frame
        self.project_root = Path(__file__).resolve().parents[1]

    def selected_indices(self, text):
        """カンマ区切りの入力文字列から、有効なAI indexだけを重複なしで取り出す。"""
        if text.strip() == '':
            return []
        indices = []
        for part in text.split(','):
            index = self._parse_index(part)
            if index is not None and index not in indices:
                indices.append(index)
        return indices

    def selected_index(self, text, default = None):
        """単一AI indexを返す。不正なときは default を返す。"""
        index = self._parse_index(text)
        if index is None:
            return default
        return index

    def _parse_index(self, text):
        """1つの文字列をAI indexとして解釈し、範囲外や不正値ならNoneを返す。"""
        text = text.strip()
        if text == '':
            return None
        try:
            index = int(text)
        except ValueError:
            return None
        if 0 <= index < self.frame.AInum:
            return index
        return None

    def load(self, index, keylis = None):
        """指定AIのパラメータ・optimizer状態をAIdatasから読み込む。"""
        ai = self.frame.AIs[index]
        data_dir = self._data_dir(index)
        if self._load_model_file(ai, data_dir):
            self._after_load(ai)
            return
        for key in self._target_keys(ai, keylis):
            self._load_param_set(ai, data_dir, key)
        self._after_load(ai)

    def save(self, index, keylis = None):
        """指定AIのパラメータ・optimizer状態をAIdatasへ保存する。"""
        ai = self.frame.AIs[index]
        data_dir = self._data_dir(index)
        if self._save_model_file(ai, data_dir):
            return
        for key in self._target_keys(ai, keylis):
            self._save_param_set(ai, data_dir, key)

    def load_all(self, keylis = None):
        """全AIのパラメータを順番に読み込む。"""
        for index in range(self.frame.AInum):
            self.load(index, keylis = keylis)
        print("Load Completed!!")

    def save_all(self, keylis = None):
        """全AIのパラメータを順番に保存する。"""
        for index in range(self.frame.AInum):
            self.save(index, keylis = keylis)
        print("Save Completed!!")

    def load_selected(self, text, keylis = None):
        """入力欄で指定されたAI indexだけを読み込む。"""
        for index in self.selected_indices(text):
            self.load(index, keylis = keylis)
        print("Load Completed!!")

    def save_selected(self, text, keylis = None):
        """入力欄で指定されたAI indexだけを保存する。"""
        for index in self.selected_indices(text):
            self.save(index, keylis = keylis)
        print("Save Completed!!")

    def param_keys(self, index):
        """指定AIの param key を表示順で返す。"""
        return sorted(self.frame.AIs[index].params.keys())

    def param_summary(self, index, key):
        """表示用の基本統計を返す。"""
        array = self.frame.AIs[index].params[key]
        return {
            'shape': array.shape,
            'size': int(array.size),
            'dtype': str(array.dtype),
            'min': float(np.min(array)),
            'max': float(np.max(array)),
            'mean': float(np.mean(array)),
            'std': float(np.std(array)),
        }

    def param_preview(self, index, key):
        """配列の先頭側を読みやすい文字列で返す。"""
        array = self.frame.AIs[index].params[key]
        return np.array2string(
            array,
            precision = 4,
            suppress_small = True,
            threshold = 40,
            edgeitems = 3,
        )

    def param_value(self, index, key, index_text = ''):
        """指定 index の現在値を返す。"""
        array = self.frame.AIs[index].params[key]
        param_index = self._parse_param_index(array, index_text)
        return float(array[param_index])

    def set_param_value(self, index, key, index_text, value_text):
        """指定 index の値を更新し、推論キャッシュを無効化する。"""
        ai = self.frame.AIs[index]
        array = ai.params[key]
        param_index = self._parse_param_index(array, index_text)
        array[param_index] = float(value_text)
        ai.mark_params_dirty()
        return float(array[param_index])


    def _data_dir(self, index):
        """AI indexに対応する保存ディレクトリ名を作る。"""
        namespace = getattr(self.frame.cube, 'parameter_namespace', None)
        if namespace:
            return self.project_root / 'GroupAIdatas' / namespace / ('AI' + str(index))
        return self.project_root / ('AIdatas' + str(index))

    def _target_keys(self, ai, keylis):
        """除外リストに含まれないパラメータkeyだけを列挙する。"""
        skipped_keys = set([] if keylis is None else keylis)
        return [key for key in ai.params.keys() if key not in skipped_keys]

    def _save_model_file(self, ai, data_dir):
        if not hasattr(ai, 'save_model'):
            return False
        return bool(ai.save_model(data_dir))

    def _load_model_file(self, ai, data_dir):
        if not hasattr(ai, 'load_model'):
            return False
        return bool(ai.load_model(data_dir))

    def _load_param_set(self, ai, data_dir, key):
        """1つのkeyについて、重み・v・hを読み込む。"""
        data_dir = Path(data_dir)
        param_path = data_dir / f'{key}.npy'
        v_path = data_dir / f'{key}_v.npy'
        h_path = data_dir / f'{key}_h.npy'
        if not param_path.exists():
            print(f"Skip missing param: {param_path}")
            return
        loaded_param = np.load(param_path)
        if loaded_param.shape != ai.params[key].shape:
            print(f"Skip shape mismatch param: {param_path} {loaded_param.shape} != {ai.params[key].shape}")
            return
        ai.params[key][:] = loaded_param
        if v_path.exists():
            loaded_v = np.load(v_path)
            if loaded_v.shape == ai.v[key].shape:
                ai.v[key][:] = loaded_v
        if h_path.exists():
            loaded_h = np.load(h_path)
            if loaded_h.shape == ai.h[key].shape:
                ai.h[key][:] = loaded_h

    def _save_param_set(self, ai, data_dir, key):
        """1つのkeyについて、重み・v・hを保存する。"""
        data_dir = Path(data_dir)
        data_dir.mkdir(parents = True, exist_ok = True)
        np.save(data_dir / f'{key}.npy', ai.params[key])
        np.save(data_dir / f'{key}_v.npy', ai.v[key])
        np.save(data_dir / f'{key}_h.npy', ai.h[key])

    def _after_load(self, ai):
        """読込後に完成状態の評価値と推論キャッシュを更新する。"""
        ai.mark_params_dirty()
        ai.set_perfect_val()

    def _parse_param_index(self, array, text):
        """Entry 文字列を ndarray index に変換する。"""
        text = text.strip()
        if text == '':
            if array.ndim == 0:
                return ()
            if array.size == 1:
                return np.unravel_index(0, array.shape)
            raise ValueError('index is required for non-scalar params')

        parts = [part.strip() for part in text.split(',') if part.strip() != '']
        if len(parts) != array.ndim:
            raise ValueError('index rank does not match param rank')

        indices = []
        for axis, part in enumerate(parts):
            try:
                axis_index = int(part)
            except ValueError as exc:
                raise ValueError('index must be integer') from exc
            if not 0 <= axis_index < array.shape[axis]:
                raise ValueError('index out of range')
            indices.append(axis_index)
        return tuple(indices)
