"""Configuration object for the main Tkinter frame."""

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple


MoveSequence = Tuple[str, ...]
ScrambleGroups = Sequence[Sequence[MoveSequence]]
UpdateScale = Tuple[float, float, float]


@dataclass
class FrameConfig:
    """Frame 起動時に渡す実験設定をまとめた dataclass。"""

    # Cube mode selection.
    puzzle_type: str = 'rubiks'
    cube_size: int = 3
    F2L: bool = False
    OLL: bool = False
    Centers: bool = False
    Edges: bool = False
    Cross: bool = False

    # Generic finite group puzzle.  Symmetric generators use 1-based one-line
    # permutations; linear generators use square matrices over F_p.
    group_kind: str = 'symmetric'
    group_degree: int = 3
    group_dimension: int = 2
    group_modulus: int = 2
    group_generators: Optional[Mapping[str, Any]] = None
    group_family: str = 'GL'
    group_name: Optional[str] = None
    group_auto_add_inverses: bool = False

    # Search / scramble initialization.
    ai_search_modes: Optional[Sequence[str]] = None
    initial_scramble_groups: Optional[ScrambleGroups] = None
    transform_random: bool = False
    search3_progress: Optional[Sequence[bool]] = None

    # Optimizer / learning toggles.
    lrs: Optional[Sequence[float]] = None
    wdlrs: Optional[Sequence[float]] = None
    skip_search: Optional[Sequence[bool]] = None
    weight_decay: Optional[Sequence[bool]] = None
    adam: Optional[Sequence[bool]] = None
    activations: Optional[Sequence[str]] = None

    # Additional runtime coefficients for each AI.
    lr_vs: Optional[Sequence[float]] = None
    lr_hs: Optional[Sequence[float]] = None
    out_cs: Optional[Sequence[float]] = None
    search3_cs: Optional[Sequence[float]] = None
    search2_max_frontiers: Optional[Sequence[int]] = None
    search2_torch_batch_sizes: Optional[Sequence[int]] = None
    search2_value_loss_types: Optional[Sequence[str]] = None
    search2_value_loss_margins: Optional[Sequence[float]] = None
    search2_rank_loss_mixes: Optional[Sequence[float]] = None
    search2_rank_loss_apply_types: Optional[Sequence[str]] = None
    search3_rank_loss_mixes: Optional[Sequence[float]] = None
    torch_training_devices: Optional[Sequence[str]] = None
    use_torch: Optional[Sequence[bool]] = None
    use_torch_predict: Optional[Sequence[bool]] = None
    use_torch_training: Optional[Sequence[bool]] = None
    update_scales: Optional[Sequence[UpdateScale]] = None
    residuals: Optional[Sequence[bool]] = None
    original_transformer_attention: Optional[Sequence[bool]] = None
    original_transformer_attention_dims: Optional[Sequence[int]] = None
    original_transformer_attention_token_modes: Optional[Sequence[str]] = None
    original_piece_attention_backward_chunk_sizes: Optional[Sequence[int]] = None
    original_train_batch_sizes: Optional[Sequence[int]] = None
    original_train_state_batch_sizes: Optional[Sequence[int]] = None
    original_train_max_batches: Optional[Sequence[int]] = None
    original_train_recent_ratios: Optional[Sequence[float]] = None
    w1_initializers: Optional[Sequence[Any]] = None
    max_search2_data: int = 8000
    max_search3_data_per_ai: int = 2000

    # Solve-time transform / priority settings.
    transform_idx: Optional[Sequence[int]] = None
    flip_inside_idx: Optional[Sequence[bool]] = None
    priority_list: Optional[Sequence[Sequence[str]]] = None

    # Extra bootstrap training data appended at startup.
    bootstrap_datas: Optional[Sequence[MoveSequence]] = None
    bootstrap_search3_datas: Optional[Sequence[MoveSequence]] = None

    def __post_init__(self):
        """AI 数に依存する設定長の整合性を確認する。"""
        ai_count = self._ai_count()
        self._validate_ai_sequence_length('lrs', self.lrs, ai_count)
        self._validate_ai_sequence_length('wdlrs', self.wdlrs, ai_count)
        self._validate_ai_sequence_length('skip_search', self.skip_search, ai_count)
        self._validate_ai_sequence_length('weight_decay', self.weight_decay, ai_count)
        self._validate_ai_sequence_length('adam', self.adam, ai_count)
        self._validate_ai_sequence_length('activations', self.activations, ai_count)
        self._validate_ai_sequence_length('lr_vs', self.lr_vs, ai_count)
        self._validate_ai_sequence_length('lr_hs', self.lr_hs, ai_count)
        self._validate_ai_sequence_length('out_cs', self.out_cs, ai_count)
        self._validate_ai_sequence_length('search3_cs', self.search3_cs, ai_count)
        self._validate_ai_sequence_length('search2_max_frontiers', self.search2_max_frontiers, ai_count)
        self._validate_ai_sequence_length('search2_torch_batch_sizes', self.search2_torch_batch_sizes, ai_count)
        self._validate_ai_sequence_length('search2_value_loss_types', self.search2_value_loss_types, ai_count)
        self._validate_ai_sequence_length('search2_value_loss_margins', self.search2_value_loss_margins, ai_count)
        self._validate_ai_sequence_length('search2_rank_loss_mixes', self.search2_rank_loss_mixes, ai_count)
        self._validate_ai_sequence_length('search2_rank_loss_apply_types', self.search2_rank_loss_apply_types, ai_count)
        self._validate_ai_sequence_length('search3_rank_loss_mixes', self.search3_rank_loss_mixes, ai_count)
        self._validate_ai_sequence_length('torch_training_devices', self.torch_training_devices, ai_count)
        self._validate_ai_sequence_length('use_torch', self.use_torch, ai_count)
        self._validate_ai_sequence_length('use_torch_predict', self.use_torch_predict, ai_count)
        self._validate_ai_sequence_length('use_torch_training', self.use_torch_training, ai_count)
        self._validate_ai_sequence_length('update_scales', self.update_scales, ai_count)
        self._validate_ai_sequence_length('residuals', self.residuals, ai_count)
        self._validate_ai_sequence_length('original_transformer_attention', self.original_transformer_attention, ai_count)
        self._validate_ai_sequence_length('original_transformer_attention_dims', self.original_transformer_attention_dims, ai_count)
        self._validate_ai_sequence_length('original_transformer_attention_token_modes', self.original_transformer_attention_token_modes, ai_count)
        self._validate_ai_sequence_length('original_piece_attention_backward_chunk_sizes', self.original_piece_attention_backward_chunk_sizes, ai_count)
        self._validate_ai_sequence_length('original_train_batch_sizes', self.original_train_batch_sizes, ai_count)
        self._validate_ai_sequence_length('original_train_state_batch_sizes', self.original_train_state_batch_sizes, ai_count)
        self._validate_ai_sequence_length('original_train_max_batches', self.original_train_max_batches, ai_count)
        self._validate_ai_sequence_length('original_train_recent_ratios', self.original_train_recent_ratios, ai_count)
        self._validate_ai_sequence_length('w1_initializers', self.w1_initializers, ai_count)
        self._validate_ai_sequence_length('transform_idx', self.transform_idx, ai_count)
        self._validate_ai_sequence_length('flip_inside_idx', self.flip_inside_idx, ai_count)
        self._validate_ai_sequence_length('priority_list', self.priority_list, ai_count)
        self._validate_ai_sequence_length('search3_progress', self.search3_progress, ai_count)
        self._validate_initial_scramble_groups()

    def _ai_count(self):
        """設定から想定される AI 数を返す。"""
        if self.ai_search_modes is None:
            return 20
        return len(self.ai_search_modes)

    def _validate_ai_sequence_length(self, field_name, values, ai_count):
        """AI ごとの設定列長が AI 数と一致するか確認する。"""
        if values is None:
            return
        if len(values) != ai_count:
            raise ValueError(
                f"FrameConfig.{field_name} must have length {ai_count}, got {len(values)}."
            )

    def _validate_initial_scramble_groups(self):
        """初期 scramble 群が stage 数に対応する形か確認する。"""
        if self.initial_scramble_groups is None:
            return
        if len(self.initial_scramble_groups) != 8:
            raise ValueError(
                "FrameConfig.initial_scramble_groups must have length 8."
            )
