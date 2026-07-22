"""Main Tkinter application frame."""

import pickle
from functools import reduce

import numpy as np
import tkinter as Tk

from ai.rubiks_ai import Rubiks_3_AI
from cto.cube import CtoCube
from fto.cube import FtoCube
from group_puzzle.cube import create_group_puzzle
from megaminx.cube import MegaminxCube
from pyraminx.cube import MasterPyraminxCube, PyraminxCube
from square1.cube import Square1Cube
from skewb.cube import SkewbCube
from cube.rubiks_cube import Rubiks_3
from managers.debug_analysis import DebugAnalysisManager
from managers.last_perms_reporter import LastPermsReporter
from managers.learn_manager import LearnManager
from managers.myperm_manager import MyPermManager
from managers.param_manager import ParamManager
from managers.search_data import SearchDataManager
from managers.solve_session import SolveSessionManager, SolveSessionState
from model.search_result import data
from ui.control_panel import ControlPanel
from ui.cto.state_viewer import CtoStateViewer
from ui.dialogs import AnalysisScoresDialog, DatasetInspectorDialog, LpShowKeyButton, ParamEditorDialog, ToolsDialog, W1EmbeddingDialog
from ui.frame_config import FrameConfig
from ui.fto.state_viewer import FtoStateViewer
from ui.group_puzzle.state_viewer import GroupStateViewer
from ui.megaminx.state_viewer import MegaminxStateViewer
from ui.pyraminx.state_viewer import PyraminxStateViewer
from ui.square1.state_viewer import Square1StateViewer
from ui.skewb.state_viewer import SkewbStateViewer
from ui.viewers import (
    LogViewer,
    MoveButton,
    MoveViewer,
    ProbViewer,
    StateViewer,
    SuccessViewer,
)

np.set_printoptions(suppress=True)


class MoveControlProxy:
    """Proxy used when one compact control represents many internal moves."""

    def __init__(self, widgets):
        self.widgets = widgets

    def configure(self, **kwargs):
        for widget in self.widgets:
            widget.configure(**kwargs)


def build_default_bootstrap_datas(cube_size):
    """既定の追加学習データ用手順列を返す。"""
    cube = Rubiks_3(size = cube_size)
    bootstrap_datas = []

    bootstrap_datas += [(" x ",),
                        (" x'",),
                        (" x2",),
                        (" y ",),
                        (" y'",),
                        (" y2",),
                        (" z ",),
                        (" z'",),
                        (" z2",),
                        ] * 16
    
    if cube_size >= 4:
        bootstrap_datas += [("2L'","2R'",'2D2','2R ',' U ',"2R'",'2D2','2R '," U'"),
                            ("2B ","2R ","2U ","2R'"," U'","2R ","2U'","2R'"," U "),
                            ("2U ","2D ","2R2","2U ","2D'","2R2"),
                            ("2R ","2F2","2R ","2F2","2U2","2R ","2U2"),
                            ("2U ","2D "," R2"," L2","2U ","2D'"," L2"," R2"),
                            ("2U ","2D "," F2"," B2","2U ","2D'"," F2"," B2"),
                            ] * 20

    if cube_size in [5,7]:
        bootstrap_datas += [("2L'","2R'",' E2','2R ',' U ',"2R'",' E2','2R '," U'"),
                            ("2B ","2R "," E'","2R'"," U'","2R "," E ","2R'"," U "),
                            ("2U ","2D "," M2","2U ","2D'"," M2"),
                            ("2R "," S2","2R "," S2"," E2","2R "," E2"),
                            ] * 20

    if cube_size >= 6:
        bootstrap_datas += [("2L'","2R'",'3D2','2R ',' U ',"2R'",'3D2','2R '," U'"),
                            ("2B ","2R ","3U ","2R'"," U'","2R ","3U'","2R'"," U "),
                            ("2U ","2D ","3R2","2U ","2D'","3R2"),
                            ("2R ","3F2","2R ","3F2","3U2","2R ","3U2"),
                            ] * 20


    if cube_size >= 6:
        bootstrap_datas = bootstrap_datas + [cube.flip_inside_moves(x) for x in bootstrap_datas]

    if cube_size >= 6:
        return (
            bootstrap_datas
            + [cube.transform(x,48) for x in bootstrap_datas]
            + [cube.transform(x,24) for x in bootstrap_datas]
            + [cube.transform(x,72) for x in bootstrap_datas]
        )
    else:
        return (
            bootstrap_datas
            + [cube.transform(x,24) for x in bootstrap_datas]
        )        


class Frame(Tk.Frame):
    def __init__(self, config = None):
        if config is None:
            config = FrameConfig()

        self.config = config
        self.puzzle_type = config.puzzle_type
        self.cube_size = config.cube_size
        Tk.Frame.__init__(self,None)
        self.master.title(self._window_title())
        self.cube = self._create_cube(config)

        # Register built-in scramble candidates and initialize stage-level state.
        self._register_initial_scrambles(
            config.initial_scramble_groups,
            F2L = config.F2L,
            OLL = config.OLL,
            Centers = config.Centers,
            Edges = config.Edges,
            Cross = config.Cross,
        )
        self.transform_random = config.transform_random
        self.search3_progress = self._build_search3_progress_flags(config.search3_progress)
        self._init_stage_settings()

        # Build AI instances and apply experiment-specific runtime settings.
        self.AIs = self._create_ai_collection(config.ai_search_modes)
        self._init_ai_state(config.cube_size, config.priority_list)
        self._link_ai_data()
        self._apply_ai_runtime_settings(
            lrs = config.lrs,
            wdlrs = config.wdlrs,
            skip_search = config.skip_search,
            weight_decay = config.weight_decay,
            adam = config.adam,
            lr_vs = config.lr_vs,
            lr_hs = config.lr_hs,
            out_cs = config.out_cs,
            search3_cs = config.search3_cs,
            search2_max_frontiers = config.search2_max_frontiers,
            search2_torch_batch_sizes = config.search2_torch_batch_sizes,
            search2_value_loss_types = config.search2_value_loss_types,
            search2_value_loss_margins = config.search2_value_loss_margins,
            search2_rank_loss_mixes = config.search2_rank_loss_mixes,
            search2_rank_loss_apply_types = config.search2_rank_loss_apply_types,
            search3_rank_loss_mixes = config.search3_rank_loss_mixes,
            torch_training_devices = config.torch_training_devices,
            use_torch = config.use_torch,
            use_torch_predict = config.use_torch_predict,
            use_torch_training = config.use_torch_training,
            update_scales = config.update_scales,
            transform_idx = config.transform_idx,
            flip_inside_idx = config.flip_inside_idx,
        )

        # Prepare myperm lookup tables and bootstrap training data.
        self._init_myperms_metadata()
        if self.puzzle_type in ['rubiks', 'cube']:
            self._append_bootstrap_datas(config.bootstrap_datas)

        self.grad_index = 0
        self.grad_mode = "Grad"
        self.grad_layer = "WO_V"

        # Construct managers first, then create the visible UI widgets.
        self._init_managers()
        self.search_data_manager.append_bootstrap_search3_datas(config.bootstrap_search3_datas)
        control_font = self._build_control_panel()
        self._build_move_pad(control_font)
        self._build_viewers(config.cube_size)

        # Finalize solve/runtime state after all dependencies are in place.
        self._init_runtime_state()
        
        
    def _register_initial_scrambles(self, initial_scramble_groups = None, F2L = False, OLL = False, Centers = False, Edges = False, Cross = False):
        """初期 scramble 候補を stage ごとの registry に登録する。"""
        if initial_scramble_groups is None:
            initial_scramble_groups = self._build_default_scramble_groups(F2L = F2L, OLL = OLL, Centers = Centers, Edges = Edges, Cross = Cross)

        for _ in range(8):
            self.cube.create_new_set()

        for stage_index, scramble_group in enumerate(initial_scramble_groups):
            for moves in scramble_group:
                normalized_moves = self._normalize_move_sequence(moves)
                self.cube.register_scramble_sequence(stage_index, normalized_moves)

    def _build_default_scramble_groups(self, F2L = False, OLL = False, Centers = False, Edges = False, Cross = False):
        """cube size と mode に応じた既定の初期 scramble 候補群を作る。"""
        S0 = []
        S1 = []
        S2 = []
        S3 = []
        S4 = []
        S5 = []
        S6 = []
        S7 = []

        return (S0, S1, S2, S3, S4, S5, S6, S7)

    def _create_cube(self, config):
        """puzzle type に応じた cube 実装を生成する。"""
        if self.puzzle_type in ('group', 'symmetric_group', 'linear_group'):
            group_kind = config.group_kind
            if self.puzzle_type == 'symmetric_group':
                group_kind = 'symmetric'
            elif self.puzzle_type == 'linear_group':
                group_kind = 'linear'
            return create_group_puzzle(
                kind = group_kind,
                degree = config.group_degree,
                dimension = config.group_dimension,
                modulus = config.group_modulus,
                generators = config.group_generators,
                family = config.group_family,
                auto_add_inverses = config.group_auto_add_inverses,
                display_name = config.group_name,
            )
        if self.puzzle_type == 'megaminx':
            return MegaminxCube(
                size = config.cube_size,
                F2L = config.F2L,
                OLL = config.OLL,
                Centers = config.Centers,
                Edges = config.Edges,
                Cross = config.Cross,
            )
        if self.puzzle_type == 'master_pyraminx':
            return MasterPyraminxCube(
                size = config.cube_size,
                F2L = config.F2L,
                OLL = config.OLL,
                Centers = config.Centers,
                Edges = config.Edges,
                Cross = config.Cross,
            )
        if self.puzzle_type == 'pyraminx':
            return PyraminxCube(
                size = config.cube_size,
                F2L = config.F2L,
                OLL = config.OLL,
                Centers = config.Centers,
                Edges = config.Edges,
                Cross = config.Cross,
            )
        if self.puzzle_type == 'skewb':
            return SkewbCube(
                size = config.cube_size,
                F2L = config.F2L,
                OLL = config.OLL,
                Centers = config.Centers,
                Edges = config.Edges,
                Cross = config.Cross,
            )
        if self.puzzle_type == 'square1':
            return Square1Cube(
                size = config.cube_size,
                F2L = config.F2L,
                OLL = config.OLL,
                Centers = config.Centers,
                Edges = config.Edges,
                Cross = config.Cross,
            )
        if self.puzzle_type == 'fto':
            return FtoCube(
                size = config.cube_size,
                F2L = config.F2L,
                OLL = config.OLL,
                Centers = config.Centers,
                Edges = config.Edges,
                Cross = config.Cross,
            )
        if self.puzzle_type == 'cto':
            return CtoCube(
                size = config.cube_size,
                F2L = config.F2L,
                OLL = config.OLL,
                Centers = config.Centers,
                Edges = config.Edges,
                Cross = config.Cross,
            )

        return Rubiks_3(
            size = config.cube_size,
            F2L = config.F2L,
            OLL = config.OLL,
            Centers = config.Centers,
            Edges = config.Edges,
            Cross = config.Cross,
        )

    def _window_title(self):
        if self.puzzle_type in ('group', 'symmetric_group', 'linear_group'):
            return 'Finite Group Puzzle'
        if self.puzzle_type == 'megaminx':
            return 'Megaminx'
        if self.puzzle_type == 'master_pyraminx':
            return 'Master Pyraminx'
        if self.puzzle_type == 'pyraminx':
            return 'Pyraminx'
        if self.puzzle_type == 'skewb':
            return 'Skewb'
        if self.puzzle_type == 'square1':
            return 'Square-1'
        if self.puzzle_type == 'fto':
            return 'Face Turning Octahedron'
        if self.puzzle_type == 'cto':
            return 'Corner Turning Octahedron'
        return 'Rubiks'

    def _init_stage_settings(self):
        """stage 進行と scramble mode の初期値を設定する。"""
        self.scramble_mode = ['myperms']

        self.stage = 0
        self.stage_num = len(self.scramble_mode)
        self.N = 0
        self.AI_idx = 0
        self.value_target_gamma = 0.5 ** (1/20)
        self.perf_num = np.zeros(self.stage_num,dtype = 'i')

    @staticmethod
    def default_ai_search_modes():
        """既定の AI mode 構成を返す。"""
        return ['search2'] * 10 + ['search3'] * 10

    def _create_ai_collection(self, ai_search_modes = None):
        """指定 mode 列に従って AI 配列を生成する。"""
        if ai_search_modes is None:
            ai_search_modes = self.default_ai_search_modes()

        mid_size = [512] * 8
        ais = []
        for ai_index, search_mode in enumerate(ai_search_modes):
            ais.append(Rubiks_3_AI(
                mid_size,
                cube_size = self.cube_size,
                Activation = self._ai_activation(ai_index),
                cube = self.cube,
                search_mode = search_mode,
                residual = self._ai_residual_enabled(ai_index, search_mode),
                use_transformer_attention = self._ai_original_transformer_attention_enabled(ai_index),
                transformer_attention_dim = self._ai_original_transformer_attention_dim(ai_index),
                transformer_attention_token_mode = self._ai_original_transformer_attention_token_mode(ai_index),
                piece_attention_backward_chunk_size = self._ai_original_piece_attention_backward_chunk_size(ai_index),
                train_batch_size = self._ai_original_train_batch_size(ai_index),
                train_state_batch_size = self._ai_original_train_state_batch_size(ai_index),
                train_max_batches = self._ai_original_train_max_batches(ai_index),
                train_recent_ratio = self._ai_original_train_recent_ratio(ai_index),
                search2_value_loss_type = self._ai_search2_value_loss_type(ai_index),
                search2_value_loss_margin = self._ai_search2_value_loss_margin(ai_index),
                search2_rank_loss_mix = self._ai_search2_rank_loss_mix(ai_index),
                search2_rank_loss_apply_type = self._ai_search2_rank_loss_apply_type(ai_index),
                search3_rank_loss_mix = self._ai_search3_rank_loss_mix(ai_index),
                w1_initializers = self._ai_w1_initializers(ai_index),
            ))
        return ais

    def _ai_search2_value_loss_type(self, ai_index):
        loss_types = getattr(self.config, 'search2_value_loss_types', None)
        if loss_types is None:
            return 'myloss2'
        return loss_types[ai_index]

    def _ai_search2_value_loss_margin(self, ai_index):
        margins = getattr(self.config, 'search2_value_loss_margins', None)
        if margins is None:
            return 0.2
        return margins[ai_index]

    def _ai_search2_rank_loss_mix(self, ai_index):
        mixes = getattr(self.config, 'search2_rank_loss_mixes', None)
        if mixes is None:
            return 0.0
        return mixes[ai_index]

    def _ai_search2_rank_loss_apply_type(self, ai_index):
        apply_types = getattr(self.config, 'search2_rank_loss_apply_types', None)
        if apply_types is None:
            return 'distance'
        return apply_types[ai_index]

    def _ai_search3_rank_loss_mix(self, ai_index):
        mixes = getattr(self.config, 'search3_rank_loss_mixes', None)
        if mixes is None:
            return 0.0
        return mixes[ai_index]

    def _ai_activation(self, ai_index):
        activations = getattr(self.config, 'activations', None)
        if activations is None:
            return 'ReLU'
        return activations[ai_index]

    def _ai_residual_enabled(self, ai_index, search_mode):
        residuals = getattr(self.config, 'residuals', None)
        if residuals is not None:
            return bool(residuals[ai_index])
        return search_mode == 'search3'

    def _ai_original_transformer_attention_enabled(self, ai_index):
        flags = getattr(self.config, 'original_transformer_attention', None)
        if flags is None:
            return False
        return bool(flags[ai_index])

    def _ai_original_transformer_attention_dim(self, ai_index):
        dims = getattr(self.config, 'original_transformer_attention_dims', None)
        if dims is None:
            return 64
        return int(dims[ai_index])

    def _ai_original_transformer_attention_token_mode(self, ai_index):
        modes = getattr(self.config, 'original_transformer_attention_token_modes', None)
        if modes is None:
            return 'hidden'
        return modes[ai_index]

    def _ai_original_piece_attention_backward_chunk_size(self, ai_index):
        sizes = getattr(self.config, 'original_piece_attention_backward_chunk_sizes', None)
        if sizes is None:
            return 32
        return int(sizes[ai_index])

    def _ai_original_train_batch_size(self, ai_index):
        sizes = getattr(self.config, 'original_train_batch_sizes', None)
        if sizes is None:
            return None
        return int(sizes[ai_index])

    def _ai_original_train_state_batch_size(self, ai_index):
        sizes = getattr(self.config, 'original_train_state_batch_sizes', None)
        if sizes is None:
            return None
        return int(sizes[ai_index])

    def _ai_original_train_max_batches(self, ai_index):
        sizes = getattr(self.config, 'original_train_max_batches', None)
        if sizes is None:
            return None
        return int(sizes[ai_index])

    def _ai_original_train_recent_ratio(self, ai_index):
        ratios = getattr(self.config, 'original_train_recent_ratios', None)
        if ratios is None:
            return None
        return float(ratios[ai_index])

    def _ai_w1_initializers(self, ai_index):
        initializers = getattr(self.config, 'w1_initializers', None)
        if initializers is None:
            return None
        return initializers[ai_index]

    def _init_ai_state(self, cube_size, priority_list):
        """AI 配列数に依存する基本状態と myval AI を初期化する。"""
        self.AInum = len(self.AIs)
        self.level = 1 * np.ones((self.AInum,self.stage_num),dtype = 'i')
        self.success = np.zeros((self.AInum,),dtype = 'i')
        self.move_keys = self.cube.move_keys
        self.my_scramble = []
        self.AI_idx = 0
        self._init_myval_ai(cube_size)

        if priority_list is not None:
            self.priority_list = self._resolve_priority_list(priority_list)
        else:
            self.priority_list = self._default_priority_list()

    def _init_myval_ai(self, cube_size):
        """myval 用の補助 AI を既定パラメータで初期化する。"""
        self.myval_AI = Rubiks_3_AI([2],cube_size = cube_size,cube = self.cube,Batch_Normalize = True)
        self.myval_AI.params['BNg1'][:] = 1
        self.myval_AI.params['BNb1'][:] = 0
        self.myval_AI.params['BNgV'][:] = 1
        self.myval_AI.params['BNbV'][:] = 0
        self.myval_AI.params['W1'][:] = 0
        self.myval_AI.params['B1'][:] = 0
        self.myval_AI.params['WO_V'][:] = 1
        self._sync_value_target_gamma()
        self.myval_AI.params['W1'][0] = self.cube.makedata()
        self.myval_AI.params['W1'][0,-192:] *= 1
        self.myval_AI.params['WM_V'] *= 0
        for i in range(self.myval_AI.params['WM_V'].shape[0]):
            self.myval_AI.params['WM_V'][i,i] = 1
        self.myval_AI.myval = True
        self.myval_AI.mark_params_dirty()

    def _build_search3_progress_flags(self, search3_progress):
        """AI ごとの Search3 progress 表示フラグ列を返す。"""
        if search3_progress is None:
            ai_modes = self.config.ai_search_modes or self.default_ai_search_modes()
            return [True] * len(ai_modes)
        return list(search3_progress)

    def _default_priority_list(self):
        """puzzle type ごとの既定 priority list を返す。"""
        if self.puzzle_type in ('group', 'symmetric_group', 'linear_group'):
            return [list(self.cube.group_val.keys())] * self.AInum
        if self.puzzle_type == 'megaminx':
            return [['Corner', 'MidEdge']] * self.AInum
        if self.puzzle_type == 'pyraminx':
            return [['Corner', 'Edge', 'Center']] * self.AInum
        if self.puzzle_type == 'master_pyraminx':
            return [['Corner', 'Edge', 'MidEdge', 'Center']] * self.AInum
        if self.puzzle_type == 'skewb':
            return [['Corner', 'Center']] * self.AInum
        if self.puzzle_type == 'fto':
            return [['Corner', 'Edge', 'CenterA', 'CenterB']] * self.AInum
        if self.puzzle_type == 'cto':
            return [['Corner', 'Edge', 'Center']] * self.AInum

        return [[
            'CoreCenter', 'ObliqueCenter-A', 'PlusCenter-Layer2',
            'XCenter-Layer2', 'ObliqueCenter-B', 'PlusCenter-Layer3',
            'XCenter-Layer3', 'Wing-Layer2', 'Wing-Layer3',
            'Corner', 'MidEdge',
        ]] * self.AInum

    def _resolve_priority_list(self, priority_list):
        """Filter configured priority groups to those supported by the current puzzle."""
        available_groups = set(getattr(self.cube, 'group_val', {}).keys())
        if not available_groups:
            return self._default_priority_list()

        resolved = []
        for group_row in priority_list:
            filtered_row = [group for group in group_row if group in available_groups]
            if not filtered_row:
                filtered_row = self._default_priority_list()[0].copy()
            resolved.append(filtered_row)
        return resolved

    def _link_ai_data(self):
        """全 AI と myval AI を cube / datas に接続する。"""
        for i in range(self.AInum):
            if i > 0:
                self.AIs[i].datas = self.AIs[0].datas
            self.AIs[i].cube = self.cube

        self.myval_AI.cube = self.cube
        self.myval_AI.datas = self.AIs[0].datas

    def _apply_ai_runtime_settings(self,
                                   lrs = None,
                                   wdlrs = None,
                                   skip_search = None,
                                   weight_decay = None,
                                   adam = None,
                                   lr_vs = None,
                                   lr_hs = None,
                                   out_cs = None,
                                   search3_cs = None,
                                   search2_max_frontiers = None,
                                   search2_torch_batch_sizes = None,
                                   search2_value_loss_types = None,
                                   search2_value_loss_margins = None,
                                   search2_rank_loss_mixes = None,
                                   search2_rank_loss_apply_types = None,
                                   search3_rank_loss_mixes = None,
                                   torch_training_devices = None,
                                   use_torch = None,
                                   use_torch_predict = None,
                                   use_torch_training = None,
                                   update_scales = None,
                                   transform_idx = None,
                                   flip_inside_idx = None):
        """AI ごとの学習・探索設定と transform 設定を適用する。"""
        for i in range(self.AInum):
            if lrs is not None:
                self.AIs[i].lr = lrs[i]
            if wdlrs is not None:
                self.AIs[i].wdlr = wdlrs[i]
            if skip_search is not None:
                self.AIs[i].skip_search = skip_search[i]
            if weight_decay is not None:
                self.AIs[i].weight_decay = weight_decay[i]
            if adam is not None:
                self.AIs[i].adam = adam[i]

            if lr_vs is not None:
                self.AIs[i].lr_v = lr_vs[i]
            else:
                self.AIs[i].lr_v = 0.99

            if lr_hs is not None:
                self.AIs[i].lr_h = lr_hs[i]
            else:
                self.AIs[i].lr_h = 0.99

            if out_cs is not None:
                self.AIs[i].out_C = out_cs[i]
            else:
                self.AIs[i].out_C = 1.0

            if search3_cs is not None:
                self.AIs[i].search3_C = search3_cs[i]
            else:
                self.AIs[i].search3_C = 0.05
            if search2_max_frontiers is not None:
                self.AIs[i].search2_max_frontier = int(search2_max_frontiers[i])
            if search2_torch_batch_sizes is not None:
                self.AIs[i].search2_torch_batch_size = int(search2_torch_batch_sizes[i])
            if search2_value_loss_types is not None:
                self.AIs[i].set_search2_value_loss_type(search2_value_loss_types[i])
            if search2_value_loss_margins is not None:
                self.AIs[i].set_search2_value_loss_margin(search2_value_loss_margins[i])
            if search2_rank_loss_mixes is not None:
                self.AIs[i].set_search2_rank_loss_mix(search2_rank_loss_mixes[i])
            if search2_rank_loss_apply_types is not None:
                self.AIs[i].set_search2_rank_loss_apply_type(search2_rank_loss_apply_types[i])
            if search3_rank_loss_mixes is not None:
                self.AIs[i].set_search3_rank_loss_mix(search3_rank_loss_mixes[i])
            if torch_training_devices is not None:
                self.AIs[i].torch_training_device = str(torch_training_devices[i])
            if use_torch is not None:
                self.AIs[i].use_torch = bool(use_torch[i])
            if use_torch_predict is not None:
                self.AIs[i].use_torch_predict = bool(use_torch_predict[i])
            if use_torch_training is not None:
                self.AIs[i].use_torch_training = bool(use_torch_training[i])

            self._apply_update_scales(self.AIs[i], None if update_scales is None else update_scales[i])

        if transform_idx is not None:
            self.transform_idx = transform_idx
        else:
            self.transform_idx = [0] * self.AInum

        if flip_inside_idx is not None:
            self.flip_inside_idx = flip_inside_idx
        else:
            self.flip_inside_idx = [False] * self.AInum

    def _apply_update_scales(self, ai, scales):
        """Apply shared/policy/value update scales to a single AI."""
        if scales is None:
            shared_scale, policy_scale, value_scale = (1.0, 1.0, 1.0)
        else:
            shared_scale, policy_scale, value_scale = scales
        ai.update_scale_shared = float(shared_scale)
        ai.update_scale_policy = float(policy_scale)
        ai.update_scale_value = float(value_scale)

    def _init_myperms_metadata(self):
        """myperms の参照用メタデータと逆引き辞書を構築する。"""
        self.myval_AI.myperms = self.cube.myperms
        self.myperms_keys = sorted(list(self.cube.myperms.keys()))
        self.myperms_vals = set(self.cube.myperms.values())
        self.myperms_len = len(self.cube.myperms)
        self.myperms_col = self._build_myperms_col()

    def _build_myperms_col(self):
        """適用後 state から myperm 名を引く逆引き辞書を作る。"""
        myperms_col = {}
        for key in self.cube.myperms.keys():
            snapshot = self.cube._snapshot() if hasattr(self.cube, '_snapshot') else self.cube.state.copy()
            try:
                for m in self.cube.invert_moves(self.cube.myperms[key]):
                    self.cube.make_move(m)

                state_key = reduce(lambda x,y: x+y,self.cube.state)
                if state_key not in myperms_col:
                    myperms_col[state_key] = key

                for m in self.cube.myperms[key]:
                    self.cube.make_move(m)
            except Exception:
                if hasattr(self.cube, '_restore'):
                    self.cube._restore(snapshot)
                else:
                    self.cube.state = snapshot

        return myperms_col

    def _append_bootstrap_datas(self, bootstrap_datas = None):
        """追加の初期学習データを登録し、全 AI の index を張り直す。"""
        if bootstrap_datas is None:
            bootstrap_datas = build_default_bootstrap_datas(self.cube_size)

        for moves in bootstrap_datas:
            d = data(
                moves,
                self.cube.invert_moves(moves),
                None,
                source_search_mode = 'bootstrap',
                source_search2_value_loss_type = 'bootstrap',
            )
            d.succeeded = True
            self.AIs[0].datas.append(d)

        for i in range(self.AInum):
            self.AIs[i].indices = list(range(len(self.AIs[0].datas)))

    def _init_managers(self):
        """manager 群と solve state を生成する。"""
        self.param_manager = ParamManager(self)
        self.learn_manager = LearnManager(self)
        self.last_perms_reporter = LastPermsReporter(self)
        self.debug_analysis_manager = DebugAnalysisManager(self)
        self.solve_session_manager = SolveSessionManager(self)
        self.search_data_manager = SearchDataManager(self)
        self.myperm_manager = MyPermManager(self)
        self.solve_state = SolveSessionState()

    def _build_control_panel(self):
        """上部 control panel を生成して alias を束縛する。"""
        self.control_panel = ControlPanel(self,self)
        self.control_buttons = self.control_panel
        self.control_panel.grid(row = 0,column = 0,columnspan = 4,sticky = 'ew')
        self._bind_control_panel_aliases()
        return self.control_panel.font

    def _build_move_pad(self, font):
        """手動 move 用の別ウィンドウを生成する。"""
        self.move_window = Tk.Toplevel(self)
        self.move_window.title('Manual Moves')
        self.move_window.protocol('WM_DELETE_WINDOW',self.hide_move_pad)
        self.move_window.withdraw()
        self.move_buttons = Tk.Frame(self.move_window,relief = Tk.RIDGE,bd = 4)
        self.move_buttons.pack(fill = 'both',expand = True)
        self.close_move_pad_button = Tk.Button(
            self.move_buttons,
            text = 'Close',
            font = font,
            padx = 1,
            pady = 1,
            command = self.hide_move_pad,
        )
        if self.puzzle_type == 'square1':
            self._build_square1_move_pad(font)
            return

        self.close_move_pad_button.grid(row = 0,column = 0,columnspan = len(self.move_keys) // 9 + 1,sticky = 'ew')

        self.movebuttons = {}
        for i, move_key in enumerate(self.move_keys):
            self.movebuttons[move_key] = MoveButton(self.move_buttons,move_key,self.cube,font,self)
            self.movebuttons[move_key].grid(row = i % 9 + 1,column = i // 9)

    def _build_square1_move_pad(self, font):
        """Square-1用の手動move入力をコンパクトに配置する。"""
        self.close_move_pad_button.grid(row = 0,column = 0,columnspan = 4,sticky = 'ew')
        self.square1_u_var = Tk.StringVar(value = '0')
        self.square1_d_var = Tk.StringVar(value = '0')
        self.square1_slash_var = Tk.IntVar(value = 1)

        Tk.Label(self.move_buttons,text = 'U',font = font).grid(row = 1,column = 0,sticky = 'e')
        Tk.Label(self.move_buttons,text = 'D',font = font).grid(row = 2,column = 0,sticky = 'e')
        u_spin = Tk.Spinbox(self.move_buttons,from_ = -5,to = 6,width = 4,font = font,textvariable = self.square1_u_var,command = self._update_square1_manual_status)
        d_spin = Tk.Spinbox(self.move_buttons,from_ = -5,to = 6,width = 4,font = font,textvariable = self.square1_d_var,command = self._update_square1_manual_status)
        u_spin.grid(row = 1,column = 1,sticky = 'ew')
        d_spin.grid(row = 2,column = 1,sticky = 'ew')

        slash_check = Tk.Checkbutton(self.move_buttons,text = '/',font = font,variable = self.square1_slash_var,command = self._update_square1_manual_status)
        slash_check.grid(row = 1,column = 2,rowspan = 2,sticky = 'nsew')
        apply_button = Tk.Button(self.move_buttons,text = 'Apply',font = font,padx = 1,pady = 1,command = self._apply_square1_manual_move)
        apply_button.grid(row = 1,column = 3,rowspan = 2,sticky = 'nsew')

        slash_button = Tk.Button(self.move_buttons,text = '/',font = font,padx = 1,pady = 1,command = lambda:self._set_and_apply_square1_move(0, 0, True))
        slash_button.grid(row = 3,column = 0,columnspan = 2,sticky = 'ew')
        rotate_button = Tk.Button(self.move_buttons,text = 'Rotate only',font = font,padx = 1,pady = 1,command = lambda:self._set_square1_slash(False))
        rotate_button.grid(row = 3,column = 2,columnspan = 2,sticky = 'ew')

        self.square1_status_label = Tk.Label(self.move_buttons,text = '',font = font)
        self.square1_status_label.grid(row = 4,column = 0,columnspan = 4,sticky = 'ew')
        for column in range(4):
            self.move_buttons.grid_columnconfigure(column,weight = 1)

        controls = [u_spin, d_spin, slash_check, apply_button, slash_button, rotate_button]
        proxy = MoveControlProxy(controls)
        self.movebuttons = {move_key:proxy for move_key in self.move_keys}
        self._update_square1_manual_status()

    def _square1_manual_move_from_vars(self):
        u = int(self.square1_u_var.get())
        d = int(self.square1_d_var.get())
        slash = "/" if self.square1_slash_var.get() else None
        return self.cube.normalize_move_key((u, d, slash))

    def _apply_square1_manual_move(self):
        try:
            move = self._square1_manual_move_from_vars()
            if not self.cube.is_legal_move(move):
                self._update_square1_manual_status()
                return
            self.cube.make_move(move)
        except Exception as error:
            self.square1_status_label.configure(text = str(error))
            return
        self.SV.set_color(self.cube.state)
        self._update_square1_manual_status()

    def _set_and_apply_square1_move(self, u, d, slash):
        self.square1_u_var.set(str(u))
        self.square1_d_var.set(str(d))
        self.square1_slash_var.set(1 if slash else 0)
        self._apply_square1_manual_move()

    def _set_square1_slash(self, slash):
        self.square1_slash_var.set(1 if slash else 0)
        self._update_square1_manual_status()

    def _update_square1_manual_status(self):
        try:
            move = self._square1_manual_move_from_vars()
            status = self.cube.format_move(move)
            if not self.cube.is_legal_move(move):
                status += '  illegal'
        except Exception:
            status = 'invalid'
        self.square1_status_label.configure(text = status)

    def _build_viewers(self, cube_size):
        """state/move/prob/success viewer を生成して配置する。"""
        self._build_grad_viewer_panels()
        if self.puzzle_type in ('group', 'symmetric_group', 'linear_group'):
            viewer_class = GroupStateViewer
            self.SV = viewer_class(self, self.cube)
            self.grad_viewer_positive = viewer_class(self.grad_viewer_positive_panel, self.cube, mini_mode = True)
            self.grad_viewer_negative = viewer_class(self.grad_viewer_negative_panel, self.cube, mini_mode = True)
        elif self.puzzle_type == 'megaminx':
            viewer_class = MegaminxStateViewer
            self.SV = viewer_class(self)
            self.grad_viewer_positive = viewer_class(self.grad_viewer_positive_panel, mini_mode = True)
            self.grad_viewer_negative = viewer_class(self.grad_viewer_negative_panel, mini_mode = True)
        elif self.puzzle_type in ('pyraminx', 'master_pyraminx'):
            viewer_class = PyraminxStateViewer
            self.SV = viewer_class(self, cube_size)
            self.grad_viewer_positive = viewer_class(self.grad_viewer_positive_panel, cube_size, mini_mode = True)
            self.grad_viewer_negative = viewer_class(self.grad_viewer_negative_panel, cube_size, mini_mode = True)
        elif self.puzzle_type == 'skewb':
            viewer_class = SkewbStateViewer
            self.SV = viewer_class(self)
            self.grad_viewer_positive = viewer_class(self.grad_viewer_positive_panel, mini_mode = True)
            self.grad_viewer_negative = viewer_class(self.grad_viewer_negative_panel, mini_mode = True)
        elif self.puzzle_type == 'square1':
            viewer_class = Square1StateViewer
            self.SV = viewer_class(self)
            self.grad_viewer_positive = viewer_class(self.grad_viewer_positive_panel, mini_mode = True)
            self.grad_viewer_negative = viewer_class(self.grad_viewer_negative_panel, mini_mode = True)
        elif self.puzzle_type == 'fto':
            viewer_class = FtoStateViewer
            self.SV = viewer_class(self)
            self.grad_viewer_positive = viewer_class(self.grad_viewer_positive_panel, mini_mode = True)
            self.grad_viewer_negative = viewer_class(self.grad_viewer_negative_panel, mini_mode = True)
        elif self.puzzle_type == 'cto':
            viewer_class = CtoStateViewer
            self.SV = viewer_class(self)
            self.grad_viewer_positive = viewer_class(self.grad_viewer_positive_panel, mini_mode = True)
            self.grad_viewer_negative = viewer_class(self.grad_viewer_negative_panel, mini_mode = True)
        else:
            viewer_class = StateViewer
            self.SV = viewer_class(self,cube_size)
            self.grad_viewer_positive = viewer_class(self.grad_viewer_positive_panel,cube_size,mini_mode = True)
            self.grad_viewer_negative = viewer_class(self.grad_viewer_negative_panel,cube_size,mini_mode = True)

        self.SV.grid(row = 1,column = 0,columnspan = 2)
        self.grad_viewer_positive_panel.grid(row = 2,column = 0,sticky = 'nsew')
        self.grad_viewer_negative_panel.grid(row = 2,column = 1,sticky = 'nsew')
        self.grad_viewer_positive.grid(row = 1,column = 0)
        self.grad_viewer_negative.grid(row = 1,column = 0)
        self.MV = MoveViewer(self)
        self.MV.grid(row = 1,column = 2,columnspan = 2)

        self.PV = ProbViewer(self,self._display_move_keys(self.move_keys))
        self.PV.grid(row = 2,column = 2,sticky = 'nw')
        self.success_viewer = SuccessViewer(self,self.AInum)
        self.success_viewer.grid(row = 2,column = 3,sticky = 'nsew')
        self.success_viewer.put_summary(self.success,self.N,self.AI_idx)
        self.log_viewer = LogViewer(self)
        self.log_viewer.grid(row = 3,column = 2,columnspan = 2,sticky = 'nsew')

    def _build_grad_viewer_panels(self):
        """GradViewerをPositive/Negativeラベルとrange表示付きで配置する。"""
        self.grad_viewer_positive_panel = Tk.Frame(self, relief = Tk.RIDGE, bd = 2, bg = '#303030')
        self.grad_viewer_negative_panel = Tk.Frame(self, relief = Tk.RIDGE, bd = 2, bg = '#303030')
        self.grad_viewer_positive_title = Tk.Label(
            self.grad_viewer_positive_panel,
            text = 'Positive / high values',
            font = ('Century Gothic', 11, 'bold'),
            fg = '#FFB0B0',
            bg = '#303030',
        )
        self.grad_viewer_negative_title = Tk.Label(
            self.grad_viewer_negative_panel,
            text = 'Negative / low values',
            font = ('Century Gothic', 11, 'bold'),
            fg = '#B0C8FF',
            bg = '#303030',
        )
        self.grad_viewer_positive_range = Tk.Label(
            self.grad_viewer_positive_panel,
            text = 'range: -',
            font = ('Menlo', 9),
            fg = '#F0F0F0',
            bg = '#303030',
            justify = Tk.LEFT,
        )
        self.grad_viewer_negative_range = Tk.Label(
            self.grad_viewer_negative_panel,
            text = 'range: -',
            font = ('Menlo', 9),
            fg = '#F0F0F0',
            bg = '#303030',
            justify = Tk.LEFT,
        )
        self.grad_viewer_info_label = Tk.Label(
            self,
            text = '',
            font = ('Menlo', 9),
            fg = '#F0F0F0',
            bg = '#303030',
            anchor = 'w',
        )
        self.grad_viewer_positive_title.grid(row = 0,column = 0,sticky = 'ew')
        self.grad_viewer_negative_title.grid(row = 0,column = 0,sticky = 'ew')
        self.grad_viewer_positive_range.grid(row = 2,column = 0,sticky = 'ew')
        self.grad_viewer_negative_range.grid(row = 2,column = 0,sticky = 'ew')
        self.grad_viewer_positive_panel.grid_columnconfigure(0, weight = 1)
        self.grad_viewer_negative_panel.grid_columnconfigure(0, weight = 1)

    def set_grad_viewer_info(self, info_text, positive_range_text, negative_range_text):
        """GradViewerの対象AIと選択値rangeを表示する。"""
        if hasattr(self, 'grad_viewer_info_label'):
            self.grad_viewer_info_label.configure(text = info_text)
            self.grad_viewer_info_label.grid(row = 3,column = 0,columnspan = 2,sticky = 'new')
        if hasattr(self, 'grad_viewer_positive_range'):
            self.grad_viewer_positive_range.configure(text = positive_range_text)
        if hasattr(self, 'grad_viewer_negative_range'):
            self.grad_viewer_negative_range.configure(text = negative_range_text)

    def _init_runtime_state(self):
        """UI 起動直後の runtime 状態を初期化する。"""
        self.stop = False
        self.last_perms = {}
        self.last_perms_changed_number = {}
        self.solve_state.reset_session()
        self.data_len = len(self.AIs[0].datas)
        self.data_search3_len = self.current_search3_data_len()

    def current_search3_data_len(self):
        """Return the largest Search3 dataset length among AIs."""
        if not self.AIs:
            return 0
        return max(len(ai.datas_search3) for ai in self.AIs)

    def _display_move_keys(self, move_keys):
        """Return move labels in puzzle-specific display notation when available."""
        if hasattr(self.cube, 'format_moves'):
            return tuple(self.cube.format_moves(move_keys))
        return tuple(move_keys)

    def display_move_rows(self, move_rows):
        """Return move rows in puzzle-specific display notation when available."""
        if not hasattr(self.cube, 'format_moves'):
            return move_rows
        return [tuple(self.cube.format_moves(moves)) for moves in move_rows]

    def display_move_sequence(self, moves):
        """Return one move sequence in puzzle-specific display notation when available."""
        if hasattr(self.cube, 'format_moves'):
            return tuple(self.cube.format_moves(moves))
        return tuple(moves)

    def _normalize_move_sequence(self, moves):
        """Normalize a move sequence for puzzle-specific internal storage when supported."""
        if hasattr(self.cube, 'normalize_move_key'):
            return tuple(self.cube.normalize_move_key(move) for move in moves)
        return tuple(moves)

    def append_log(self, message):
        """GUI 上のログビューアへ 1 行追記する。"""
        if isinstance(message, (tuple, list)) and hasattr(self.cube, 'format_moves'):
            message = self.cube.format_moves(message)
        self.log_viewer.append_line(message)
        self.update()

    def initialize(self,N):
        #for key in ['W1','W2','W3','W4','W5','W6','W7','W8','WM_P','WM_V']:
        for key in ['W1']:
            if key in self.AIs[N].params:
                Z = np.linalg.svd(self.AIs[N].params[key])
                Z[1][8:] = 0
                A = np.zeros_like(self.AIs[N].params[key],dtype = 'f')
                size = Z[1].shape[0]
                A[:size,:size] = np.diag(Z[1])
                self.AIs[N].params[key][:] = Z[0] @ A @ Z[2]
                self.AIs[N].v[key] *= 0
                

    def toggle_move_pad(self):
        if self.move_window.state() == 'withdrawn':
            self.show_move_pad()
        else:
            self.hide_move_pad()

    def show_move_pad(self):
        self.move_window.deiconify()
        self.move_window.lift()
        self.open_move_pad_button.configure(text = 'hide manual moves')

    def hide_move_pad(self):
        self.move_window.withdraw()
        self.open_move_pad_button.configure(text = 'manual moves')

    def _bind_control_panel_aliases(self):
        alias_names = [
            'reset_button',
            'stopper_button',
            'my_solve_button',
            'loadparams_all_button',
            'saveparams_all_button',
            'tools_button',
            'param_index_var',
            'param_index_entry',
            'loadparams_selected_button',
            'saveparams_selected_button',
            'sum_and_var_button',
            'level_var',
            'level_entry',
            'set_level_button',
            'show_counter_button',
            'grad_index_var',
            'grad_index_entry',
            'grad_mode_var',
            'grad_mode_menu',
            'grad_layer_var',
            'grad_layer_entry',
            'show_debug_viewer_button',
        ]
        for alias_name in alias_names:
            setattr(self, alias_name, getattr(self.control_panel, alias_name))

    def _selected_param_indices(self):
        return self.param_manager.selected_indices(self.param_index_var.get())

    def loadparams_selected(self, keylis = None):
        self.param_manager.load_selected(self.param_index_var.get(), keylis = keylis)

    def saveparams_selected(self, keylis = None):
        self.param_manager.save_selected(self.param_index_var.get(), keylis = keylis)

    def sum_and_var_from_entry(self):
        for index in self._selected_param_indices():
            self.sum_and_var(index)

    def show_counter_from_entry(self):
        text = self.level_var.get().strip()
        if text == '':
            return
        try:
            counter_index = int(text)
        except ValueError:
            return
        self.show_counter(counter_index)

    def set_value_target_gamma(self, gamma):
        try:
            self.value_target_gamma = float(gamma)
        except ValueError:
            return False
        self._sync_value_target_gamma()
        return True

    def _sync_value_target_gamma(self):
        for ai in self.AIs:
            ai.value_target_gamma = self.value_target_gamma
        self.myval_AI.value_target_gamma = self.value_target_gamma

    def set_level_from_entry(self):
        text = self.level_var.get().strip()
        if text == '':
            return
        try:
            level = int(text)
        except ValueError:
            return
        self.set_level(level)

    def show_debug_viewer_from_entry(self):
        self.debug_analysis_manager.update_viewer_settings(
            self.grad_index_var.get(),
            self.grad_mode_var.get(),
            self.grad_layer_var.get(),
        )
        ai_index = self._debug_viewer_ai_index()
        self.debug_analysis_manager.show_current_viewer(ai_index,self.cube_size ** 2)

    def _debug_viewer_ai_index(self):
        """停止中だけidx欄で指定したAIをGradViewer対象にする。"""
        if not self.stop:
            return self.AI_idx
        selected_indices = self.param_manager.selected_indices(self.param_index_var.get())
        if len(selected_indices) == 0:
            return self.AI_idx
        return selected_indices[0]

    def set_level(self,N):
        #self.load_lists()
        self.level[:] = N

    def myperms_col_key(self,M):
        return self.myperm_manager.myperms_col_key(M)
        

    def sum_and_var(self,i):
        self.debug_analysis_manager.sum_and_var(i)


    def save_lists(self):
        f = open('Datas.binaryfile','wb')
        pickle.dump(self.AIs[0].datas,f)
        f.close()


    def load_lists(self):
        f = open('Datas.binaryfile','rb')
        L = pickle.load(f)
        f.close()

        for i in range(self.AInum):
            self.AIs[i].datas = L
            self.AIs[i].indices = list(range(len(L)))

    def max_val(self,T0 = (),T1 = (),head = '',Top = True,Num = 1):
        self.debug_analysis_manager.max_val(T0 = T0,T1 = T1,head = head,Top = Top,Num = Num)

    def lpk(self):
        self.last_perms_reporter.lpk()
                

    def lp_show(self,key,N = None):
        self.last_perms_reporter.lp_show(key,N = N)

    def show_counter(self,N):
        self.last_perms_reporter.show_counter(N)

    def search_myperms(self,T0,T1,head):
        return self.myperm_manager.search_myperms(T0,T1,head)

    def normalize(self,i):
        self.debug_analysis_manager.normalize(i)
                
                    
    def re_activate(self,i):
        self.debug_analysis_manager.re_activate(i)
                    



    def set_color(self,S):
        self.SV.set_color(S)
    
    def reset(self):
        self.cube.reset()
        self.set_color(self.cube.state)

    def make_myperm(self):
        self.myperm_manager.open_apply_dialog()
    
    def lp_show_by_button(self):
        Frame = Tk.Toplevel(self)
        Frame.title('show lp')
        E_key = Tk.Entry(master = Frame,width = 20)
        E_key.grid(row = 0,column = 0)
        E_length = Tk.Entry(master = Frame,width = 20)
        E_length.grid(row = 1,column = 0)
        B = LpShowKeyButton(Frame,self,E_key,E_length)
        B.grid(row = 2,column = 0)

    def open_param_editor(self):
        ParamEditorDialog(self)

    def open_tools_dialog(self):
        ToolsDialog(self)

    def show_analysis_scores(self, title, rows):
        if not hasattr(self, 'analysis_scores_dialog') or not self.analysis_scores_dialog.winfo_exists():
            self.analysis_scores_dialog = AnalysisScoresDialog(self)
        self.analysis_scores_dialog.set_content(title, rows)
        self.analysis_scores_dialog.deiconify()
        self.analysis_scores_dialog.lift()

    def show_analysis_text(self, title, content):
        if not hasattr(self, 'analysis_scores_dialog') or not self.analysis_scores_dialog.winfo_exists():
            self.analysis_scores_dialog = AnalysisScoresDialog(self)
        self.analysis_scores_dialog.set_text_content(title, content)
        self.analysis_scores_dialog.deiconify()
        self.analysis_scores_dialog.lift()

    def analyze_transformer_attention(self):
        self.debug_analysis_manager.show_transformer_attention_analysis(self.AI_idx)

    def analyze_transformer_embedding(self):
        self.debug_analysis_manager.show_transformer_embedding_analysis(self.AI_idx)

    def open_w1_embedding_map(self):
        if not hasattr(self, 'w1_embedding_dialog') or not self.w1_embedding_dialog.winfo_exists():
            self.w1_embedding_dialog = W1EmbeddingDialog(self)
        else:
            self.w1_embedding_dialog.deiconify()
            self.w1_embedding_dialog.lift()

    def open_dataset_inspector(self):
        if not hasattr(self, 'dataset_inspector_dialog') or not self.dataset_inspector_dialog.winfo_exists():
            self.dataset_inspector_dialog = DatasetInspectorDialog(self)
        else:
            self.dataset_inspector_dialog.refresh()
            self.dataset_inspector_dialog.deiconify()
            self.dataset_inspector_dialog.lift()


    def my_solve(self):
        self.solve_session_manager.my_solve()

    def _schedule_next_solve(self):
        if self.cube.is_perfect():
            self.after(10,self.my_solve)
        else:
            self.after(10,self.my_solve)

    def learn(self):
        self.learn_manager.learn_all()
        
    def AIlearn(self,Indices):
        self.learn_manager.learn_indices(Indices)
    
    def stopper(self):
        self.stop = not self.stop
        if self.stop:
            self.stopper_button.configure(text = 'restart')
            self.my_solve_button.configure(state = Tk.NORMAL)
            for k in self.move_keys:
                self.movebuttons[k].configure(state = Tk.NORMAL)
        else:
            self.stopper_button.configure(text = 'stop')
                    
    def loadparams(self,N,keylis = None):
        self.param_manager.load(N, keylis = keylis)

    def saveparams(self,N,keylis = None):
        self.param_manager.save(N, keylis = keylis)

    def loadparams_all(self,keylis = None):
        self.param_manager.load_all(keylis = keylis)
    
    def saveparams_all(self,keylis = None):
        self.param_manager.save_all(keylis = keylis)

    def myfunc(self):
        self.last_perms_reporter.myfunc()

    def myfunc2(self,N = 0):
        self.last_perms_reporter.myfunc2(N = N)

    def myviewer(self,AInum,i,N = 1,SVD = False,Grad = False,IG = False,Contrast = False,Occ = False,PieceOcc = False,PolicyOcc = False,PiecePolicyOcc = False,AttnIn = False,AttnOut = False,AttnCentral = False,EmbNorm = False,EmbPC1 = False,layer = "WO_V"):
        self.debug_analysis_manager.myviewer(AInum,i,N = N,SVD = SVD,Grad = Grad,IG = IG,Contrast = Contrast,Occ = Occ,PieceOcc = PieceOcc,PolicyOcc = PolicyOcc,PiecePolicyOcc = PiecePolicyOcc,AttnIn = AttnIn,AttnOut = AttnOut,AttnCentral = AttnCentral,EmbNorm = EmbNorm,EmbPC1 = EmbPC1,layer = layer)
