"""Editable finite-group experiment configuration.

Change the constants in the CONFIG section, then run:

    python3 group_experiment.py
"""

from group_main import build_group_frame_config
from ui.frame import Frame


# --- CONFIG: freely edit this block ---------------------------------------

GROUP_KIND = "linear"
GROUP_FAMILY = "SL"
GROUP_NAME = "SL_2(F_43)"
DIMENSION = 2
MODULUS = 43

# Matrices are reduced modulo MODULUS.  Missing inverse generators are added
# automatically, so these two entries become X = {A, A^-1, B, B^-1}.
GENERATORS = {
    "A": [[42, 1], [42, 0]],
    "B": [[0, 1], [42, 0]],
}
AUTO_ADD_INVERSES = True

# Search2 models: AI 0-9 are the regular affine (Linear) network and AI 10-19
# are the piece-token Transformer network, matching the cube experiment layout.
LINEAR_AI_COUNT = 10
TRANSFORMER_AI_COUNT = 10
AI_COUNT = LINEAR_AI_COUNT + TRANSFORMER_AI_COUNT
AI_SEARCH_MODES = ("search2",) * AI_COUNT
TRANSFORMER_FLAGS = (False,) * LINEAR_AI_COUNT + (True,) * TRANSFORMER_AI_COUNT

# Every sequence-valued option must contain AI_COUNT entries.
FRAME_OPTIONS = {
    "original_transformer_attention": TRANSFORMER_FLAGS,
    "original_transformer_attention_dims": (64,) * AI_COUNT,
    "original_transformer_attention_token_modes": ("piece",) * AI_COUNT,
    "original_piece_attention_backward_chunk_sizes": (32,) * AI_COUNT,
    "use_torch": (False,) * AI_COUNT,
    "use_torch_predict": TRANSFORMER_FLAGS,
    "use_torch_training": TRANSFORMER_FLAGS,
    "torch_training_devices": ("auto",) * LINEAR_AI_COUNT + ("cpu",) * TRANSFORMER_AI_COUNT,
    "original_train_batch_sizes": (100,) * LINEAR_AI_COUNT + (20,) * TRANSFORMER_AI_COUNT,
    "original_train_state_batch_sizes": (0,) * LINEAR_AI_COUNT + (16,) * TRANSFORMER_AI_COUNT,
    "original_train_max_batches": (0,) * LINEAR_AI_COUNT + (100,) * TRANSFORMER_AI_COUNT,
    "original_train_recent_ratios": (0.0,) * LINEAR_AI_COUNT + (1.0,) * TRANSFORMER_AI_COUNT,
    "lrs": (2.0e-6,) * LINEAR_AI_COUNT + (1.0e-5,) * TRANSFORMER_AI_COUNT,
    "wdlrs": (1.0e-7,) * AI_COUNT,
    "skip_search": (True,) * AI_COUNT,
    "weight_decay": (True,) * AI_COUNT,
    "adam": (True,) * AI_COUNT,
    "search2_max_frontiers": (30000,) * AI_COUNT,
    "search2_torch_batch_sizes": (100,) * LINEAR_AI_COUNT + (64,) * TRANSFORMER_AI_COUNT,
    "search2_value_loss_types": ("myloss",) * AI_COUNT,
    "update_scales": ((5.0, 1.0, 20.0),) * AI_COUNT,
}

# -------------------------------------------------------------------------


def build_experiment_config():
    return build_group_frame_config(
        kind=GROUP_KIND,
        family=GROUP_FAMILY,
        name=GROUP_NAME,
        dimension=DIMENSION,
        modulus=MODULUS,
        generators=GENERATORS,
        auto_add_inverses=AUTO_ADD_INVERSES,
        ai_search_modes=AI_SEARCH_MODES,
        **FRAME_OPTIONS,
    )


if __name__ == "__main__":
    frame = Frame(config=build_experiment_config())
    frame.pack()
    frame.mainloop()
