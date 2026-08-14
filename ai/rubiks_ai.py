"""Rubiks AI model and its local layer/loss implementations."""

import gc
import math
import random
from collections import Counter, OrderedDict

import numpy as np

try:
    import torch
    import torch.nn.functional as F_torch
except Exception:
    torch = None
    F_torch = None

from cube.search2_engine import Search2Engine
from cube.search3_engine import Search3Engine
from cube.rubiks_cube import Rubiks_3
from ai.layers import Affine, Batch_Normalization, Hard_Sigmoid, ReLU, ResidualBlock, Sigmoid, SiLU, Transformer_SelfAttention, PieceTokenSelfAttention
from ai.losses import BCEWithLogits, MyLoss2, MyLoss2Pairwise, Myloss, Q_loss, Soft_Target_Cross_Entropy, Softmax_Cross_Entropy



class Rubiks_3_AI:
    def __init__(self,Mid,cube_size = 3,Activation = 'silu',cube = None,Batch_Normalize = False,search_mode = 'search2',residual = False,use_transformer_attention = False,transformer_attention_dim = 64,transformer_attention_token_mode = 'hidden',piece_attention_backward_chunk_size = 32,train_batch_size = None,train_state_batch_size = None,train_max_batches = None,train_recent_ratio = None,search2_value_loss_type = 'myloss2',search2_value_loss_margin = 0.2,search2_rank_loss_mix = 0.0,search2_rank_loss_apply_type = 'distance',search3_rank_loss_mix = 0.0,w1_initializers = None):
        if cube == None:
            self.cube = Rubiks_3(size = cube_size)
        else:
            self.cube = cube
        #self.ips = 36 * self.cube.surface_num
        self.Batch_Normalize = Batch_Normalize
        self.residual = residual
        self.use_transformer_attention = bool(use_transformer_attention)
        self.use_torch = False
        self.use_torch_predict = self.use_transformer_attention
        self.use_torch_training = self.use_transformer_attention
        self.requested_Mid = list(Mid)
        self.transformer_attention_dim = int(transformer_attention_dim)
        self.transformer_attention_mps = max(1,int(transformer_attention_dim))
        self.transformer_attention_token_mode = transformer_attention_token_mode
        self.piece_attention_backward_chunk_size = int(piece_attention_backward_chunk_size)
        self.use_piece_tokens = self.use_transformer_attention and transformer_attention_token_mode in ('piece','pieces','piece_tokens')
        self.train_batch_size = int(train_batch_size) if train_batch_size is not None else (24 if self.use_piece_tokens else 100)
        self.train_state_batch_size = int(train_state_batch_size) if train_state_batch_size is not None else (16 if self.use_piece_tokens else 0)
        self.train_max_batches = int(train_max_batches) if train_max_batches is not None else (8 if self.use_piece_tokens else 0)
        self.train_recent_ratio = float(train_recent_ratio) if train_recent_ratio is not None else (0.5 if self.use_piece_tokens else 0.0)
        self.torch_training_device = 'cpu' if self.use_transformer_attention else 'auto'
        self.activation = Activation
        self.ips = self.cube.ips
        self.ops = self.cube.move_len
        self.Mid = self.requested_Mid
        Mid = self.Mid

        self.skip_search = False
        self.skip_difference = 10.0
        self.search_mode = search_mode
        self.search2_value_loss_type = self._normalize_search2_value_loss_type(search2_value_loss_type)
        self.search2_value_loss_margin = float(search2_value_loss_margin)
        self.search2_rank_loss_mix = float(search2_rank_loss_mix)
        self.search2_rank_loss_apply_type = self._normalize_search2_rank_loss_apply_type(search2_rank_loss_apply_type)
        self.search3_rank_loss_mix = float(search3_rank_loss_mix)
        self._search2_value_has_rank_loss = False
        self._search3_value_has_rank_loss = False
        self._last_search2_value_loss_components = self._empty_search2_value_loss_components()
        self.w1_initializers = w1_initializers
        self.value_target_gamma = (1/2) ** (1/20)
        




        self.myval = False
        self.weight_decay = False
        self.adam = True
        self.cube_size = cube_size
        self.surface_num = self.cube.surface_num







        self.layers = OrderedDict()
        self.trunk_blocks = []
        self.params = {}
        
        
        
        self.affines = ['Aff1']
        self.attentions = []
        self.BNs = []
        self.activation_betas = []
        
        if self.use_piece_tokens:
            self.layers['Aff1'] = PieceTokenSelfAttention(self.ips,self.transformer_attention_mps,Mid[0],self._piece_token_feature_indices())
            self.layers['Aff1'].backward_chunk_size = self.piece_attention_backward_chunk_size
            self.attentions.append('Aff1')
        elif self.use_transformer_attention:
            self.layers['Aff1'] = Affine(self.ips,self.transformer_attention_mps)
        else:
            self.layers['Aff1'] = Affine(self.ips,Mid[0])
        if self.use_transformer_attention and not self.use_piece_tokens:
            self.layers['Att1'] = Transformer_SelfAttention(self.transformer_attention_mps,self.transformer_attention_mps,Mid[0])
            self.attentions.append('Att1')
        if self.Batch_Normalize:
            self.layers['BN1'] = Batch_Normalization(Mid[0])
            self.BNs.append('BN1')
        
        self.layers['Act1'] = self._create_trunk_activation(1)
        self._append_trunk_block(1)
        
        self.params['W1'] = self.layers['Aff1'].W
        self.params['B1'] = self.layers['Aff1'].B
        if self.use_transformer_attention:
            attention_layer = self.layers['Aff1'] if self.use_piece_tokens else self.layers['Att1']
            self.params['WQ1'] = attention_layer.WQ
            self.params['WK1'] = attention_layer.WK
            self.params['WV1'] = attention_layer.WV

        if self.Batch_Normalize:
            self.params['BNg1'] = self.layers['BN1'].g
            self.params['BNb1'] = self.layers['BN1'].b
            self.params['BNm1'] = self.layers['BN1'].m
            self.params['BNs1'] = self.layers['BN1'].s
        self._register_activation_params('1', self.layers['Act1'])

        
        for i in range(1,len(Mid)):
            self.layers['Aff' + str(i+1)] = Affine(Mid[i-1],Mid[i])
            if self.Batch_Normalize:
                self.layers['BN' + str(i+1)] =  Batch_Normalization(Mid[i])
            self.layers['Act' + str(i+1)] = self._create_trunk_activation(i+1)
            self._append_trunk_block(i+1)

            self.params['W' + str(i + 1)] = self.layers['Aff' + str(i + 1)].W
            self.params['B' + str(i + 1)] = self.layers['Aff' + str(i + 1)].B
            self._register_activation_params(str(i + 1), self.layers['Act' + str(i + 1)])
            self.affines.append('Aff' + str(i+1))
            if self.Batch_Normalize:
                self.params['BNg' + str(i+1)] = self.layers['BN' + str(i+1)].g
                self.params['BNb' + str(i+1)] = self.layers['BN' + str(i+1)].b
                self.params['BNm' + str(i+1)] = self.layers['BN' + str(i+1)].m
                self.params['BNs' + str(i+1)] = self.layers['BN' + str(i+1)].s
            
                self.BNs.append('BN' + str(i+1))


        self.policy_mid = Affine(Mid[-1],Mid[-1] // 2)
        self.value_mid = Affine(Mid[-1],Mid[-1] // 2)

        if self.Batch_Normalize:
            self.policy_BN = Batch_Normalization(Mid[-1] // 2)
            self.value_BN = Batch_Normalization(Mid[-1] // 2)

        self.policy_act = self._create_head_activation()
        self.value_act = self._create_head_activation()
        
        
        self.policy_layer = Affine(Mid[-1] // 2,self.ops)
        self.value_layer = Affine(Mid[-1] // 2,1)

        self.params['WM_P'] = self.policy_mid.W
        self.params['BM_P'] = self.policy_mid.B

        self.params['WM_V'] = self.value_mid.W
        self.params['BM_V'] = self.value_mid.B
        self._register_activation_params('P', self.policy_act)
        self._register_activation_params('V', self.value_act)


        if self.Batch_Normalize:
            self.params['BNgP'] = self.policy_BN.g
            self.params['BNbP'] = self.policy_BN.b
            self.params['BNmP'] = self.policy_BN.m
            self.params['BNsP'] = self.policy_BN.s

            self.params['BNgV'] = self.value_BN.g
            self.params['BNbV'] = self.value_BN.b
            self.params['BNmV'] = self.value_BN.m
            self.params['BNsV'] = self.value_BN.s

        self.params['WO_P'] = self.policy_layer.W
        self.params['BO_P'] = self.policy_layer.B

        self.params['WO_V'] = self.value_layer.W
        self.params['BO_V'] = self.value_layer.B




        if self.residual:
            self.params['W1'] *= 1
            self.params['B1'][::2] = 1.0

        if self.use_transformer_attention:
            self.params['WQ1'] *= 0.2
            self.params['WK1'] *= 0.2
            self.params['WV1'] *= 0.2

        if self.search_mode == "search3":
            self.params['WO_V'] *= 0.05


        self._apply_w1_initializers(w1_initializers)



        self.v = {}
        for key in self.params.keys():
            self.v[key] = np.zeros_like(self.params[key],dtype = 'f')

        self.h = {}
        for key in self.params.keys():
            self.h[key] = np.zeros_like(self.params[key],dtype = 'f')        

    
            
        self.lr = 1.0e-3
        self.wdlr = 1.0e-6
        self.update_scale_shared = 1.0
        self.update_scale_policy = 1.0
        self.update_scale_value = 1.0
        self.lr_C = 1
        self.out_C = 1.0
        self.lr_v = 0.99
        self.lr_h = 0.99

        self.losslayer = Softmax_Cross_Entropy()
        self.losslayer2 = self._create_search2_value_loss()
        self.search2_rank_loss = Myloss()
        self.losslayer3 = BCEWithLogits()
        self.search3_rank_loss = Myloss()
        self.losslayer4 = Soft_Target_Cross_Entropy()
        self.search3_value_sigmoid = Sigmoid()




        self.scramble_num = 3
        self.scramble_num_min = 2
        self.solve_num = 30
        self.datas = []
        self.indices = []
        self.datas_search3 = []
        self.indices_search3 = []


        self.myperms = {}
        self.nodes = {}
        self.search_num2 = 10000
        self.search2_max_frontier = 30000
        self.search2_torch_batch_size = 32 if self.use_piece_tokens else 100
        self.release_search_memory_each_step = False
        self.search_num3 = 1000
        self.search_repeat3 = 10
        self.search_batch3 = 40
        self.search_depth3 = 200
        self.search3_C = 0.05
        self.search3_max_node_cache = 5000
        self.search3_max_prediction_cache = 5000
        self.perfect_val = 1.0
        self._torch_params_cache = None
        self._torch_params_dirty = True
        self._torch_params_device = None
        self._search3_policy_transform_cache = {}
        self.search2_engine = Search2Engine(self)
        self.search3_engine = Search3Engine(self)
        self.set_perfect_val()

    def set_search2_value_loss_type(self, loss_type):
        """Set the Search2 value loss implementation used by NumPy and Torch training."""
        self.search2_value_loss_type = self._normalize_search2_value_loss_type(loss_type)
        self.losslayer2 = self._create_search2_value_loss()

    def set_search2_value_loss_margin(self, margin):
        """Set the margin used by pairwise Search2 value training."""
        self.search2_value_loss_margin = float(margin)
        self.losslayer2 = self._create_search2_value_loss()

    def set_search2_rank_loss_mix(self, mix):
        """Set the auxiliary Search2 rank-loss coefficient."""
        self.search2_rank_loss_mix = max(0.0,float(mix))

    def set_search2_rank_loss_apply_type(self, apply_type):
        """Set which Search2 value-loss modes receive the auxiliary rank loss."""
        self.search2_rank_loss_apply_type = self._normalize_search2_rank_loss_apply_type(apply_type)

    def set_search3_rank_loss_mix(self, mix):
        """Set the auxiliary Search3 rank-loss coefficient."""
        self.search3_rank_loss_mix = max(0.0,float(mix))

    def _create_trunk_activation(self, index):
        """Create the configured hidden activation for one trunk layer."""
        activation = self._normalized_activation_name()
        if activation == 'silu':
            return SiLU(beta = 0)
        if activation == 'relu':
            return ReLU()
        if activation == 'hard_sigmoid':
            return Hard_Sigmoid()
        if activation == 'sigmoid':
            return Sigmoid()
        return ReLU()

    def _create_head_activation(self):
        """Create policy/value hidden-head activation matching the configured activation."""
        activation = self._normalized_activation_name()
        if activation == 'silu':
            return SiLU(beta = 0)
        if activation == 'hard_sigmoid':
            return Hard_Sigmoid()
        if activation == 'sigmoid':
            return Sigmoid()
        return ReLU()

    def _normalized_activation_name(self):
        activation = str(getattr(self,'activation','ReLU')).strip().lower().replace('-','_')
        aliases = {
            'relu': 'relu',
            'hard_sigmoid': 'hard_sigmoid',
            'hardsigmoid': 'hard_sigmoid',
            'sigmoid': 'sigmoid',
            'silu': 'silu',
            'swish': 'silu',
            'silu_beta': 'silu',
            'swish_beta': 'silu',
        }
        return aliases.get(activation, activation)

    def _register_activation_params(self, suffix, layer):
        """Expose trainable activation parameters through params/v/h."""
        if not isinstance(layer, SiLU):
            return
        key = 'ActBeta' + str(suffix)
        self.params[key] = layer.beta
        self.activation_betas.append((key, layer))

    def _apply_w1_initializers(self, initializers):
        """Apply optional rule-based offsets to selected W1 columns at construction time."""
        self._w1_initializer_summary = []
        if not initializers:
            return
        if isinstance(initializers, dict):
            initializer_list = [initializers]
        else:
            initializer_list = list(initializers)
        records = self._w1_feature_records()
        if not records:
            raise ValueError('w1_initializers require piece-feature metadata for W1 columns.')
        for initializer in initializer_list:
            if initializer is None:
                continue
            self._apply_single_w1_initializer(initializer, records)

    def _apply_single_w1_initializer(self, initializer, records):
        if not isinstance(initializer, dict):
            raise TypeError(f'w1 initializer must be a dict, got {type(initializer).__name__}')
        selector = initializer.get('selector')
        if selector is None:
            selector = {
                key: value
                for key,value in initializer.items()
                if key not in ('vector','scale','mode','basis','constant','random_normal','std')
            }
        columns = [
            record['feature_index']
            for record in records
            if self._w1_record_matches_selector(record, selector)
        ]
        if not columns:
            self._w1_initializer_summary.append({'matched': 0, 'selector': selector})
            return
        vector = self._w1_initializer_vector(initializer)
        mode = str(initializer.get('mode','add')).lower()
        if mode in ('add','plus','+='):
            self.params['W1'][:,columns] += vector.reshape(-1,1)
        elif mode in ('set','replace','='):
            self.params['W1'][:,columns] = vector.reshape(-1,1)
        else:
            raise ValueError(f'unknown W1 initializer mode: {initializer.get("mode")}')
        self._w1_initializer_summary.append({
            'matched': len(columns),
            'selector': selector,
            'mode': mode,
            'columns': tuple(columns),
        })

    def _w1_initializer_vector(self, initializer):
        rows = int(self.params['W1'].shape[0])
        scale = float(initializer.get('scale',1.0))
        if 'vector' in initializer:
            vector_value = initializer['vector']
            if isinstance(vector_value, str):
                normalized = vector_value.lower().replace('-','_')
                if normalized in ('random','random_normal','normal'):
                    std = float(initializer.get('std',1.0))
                    return (np.random.randn(rows).astype('f') * std * scale).astype('f')
                if normalized in ('ones','one'):
                    return np.full(rows,scale,dtype = 'f')
                if normalized in ('zeros','zero'):
                    return np.zeros(rows,dtype = 'f')
                raise ValueError(f'unknown W1 initializer vector string: {vector_value}')
            vector = np.asarray(vector_value,dtype = 'f').reshape(-1)
            if vector.size == 1:
                return np.full(rows,float(vector[0]) * scale,dtype = 'f')
            if vector.size != rows:
                raise ValueError(f'W1 initializer vector length must be {rows}, got {vector.size}.')
            return (vector * scale).astype('f')
        if 'basis' in initializer:
            vector = np.zeros(rows,dtype = 'f')
            basis = initializer['basis']
            if isinstance(basis, (int, np.integer)):
                basis_indices = [int(basis)]
            else:
                basis_indices = [int(index) for index in basis]
            for basis_index in basis_indices:
                if basis_index < 0 or basis_index >= rows:
                    raise ValueError(f'W1 initializer basis index out of range: {basis_index}')
                vector[basis_index] = scale
            return vector
        if 'constant' in initializer:
            return np.full(rows,float(initializer['constant']) * scale,dtype = 'f')
        if 'random_normal' in initializer:
            std = float(initializer.get('random_normal'))
            return (np.random.randn(rows).astype('f') * std * scale).astype('f')
        raise ValueError('W1 initializer requires vector, basis, constant, or random_normal.')

    def _w1_record_matches_selector(self, record, selector):
        selector = selector or {}
        if not self._selector_bool_matches(record, selector, 'correct'):
            return False
        if not self._selector_string_matches(record['piece_type'], selector, ('piece_type','type')):
            return False
        if not self._selector_string_matches(record['solve_group'], selector, ('solve_group','group')):
            return False
        if not self._selector_position_matches(record['position_label'], selector):
            return False
        if not self._selector_feature_index_matches(record['feature_index'], selector):
            return False
        if not self._selector_colors_match(record['colors'], selector):
            return False
        return True

    def _selector_bool_matches(self, record, selector, key):
        if key not in selector:
            return True
        return bool(record.get(key,False)) == bool(selector[key])

    def _selector_string_matches(self, value, selector, keys):
        selected = None
        for key in keys:
            if key in selector:
                selected = selector[key]
                break
        if selected is None:
            return True
        if isinstance(selected, str):
            candidates = [selected]
        else:
            candidates = list(selected)
        normalized_value = self._normalize_selector_text(value)
        return any(self._normalize_selector_text(candidate) == normalized_value for candidate in candidates)

    def _selector_position_matches(self, position_label, selector):
        searchable_position = self._position_search_text(position_label)
        if 'position' in selector:
            selected = selector['position']
            candidates = [selected] if isinstance(selected, str) else list(selected)
            normalized_position = self._normalize_selector_text(position_label)
            if not any(
                self._normalize_selector_text(candidate) == normalized_position
                or self._normalize_selector_text(candidate) in searchable_position
                for candidate in candidates
            ):
                return False
        if 'position_contains' in selector:
            selected = selector['position_contains']
            fragments = [selected] if isinstance(selected, str) else list(selected)
            if not all(self._normalize_selector_text(fragment) in searchable_position for fragment in fragments):
                return False
        return True

    def _position_search_text(self, position_label):
        normalized_label = self._normalize_selector_text(position_label)
        compact_aliases = []
        text = str(position_label)
        if '(' in text and ')' in text:
            inside = text.split('(',1)[1].rsplit(')',1)[0]
            parts = [part.split(':',1)[0].strip() for part in inside.split(',') if part.strip()]
            if parts:
                compact_aliases.append(''.join(parts))
            if len(parts) >= 2:
                compact_aliases.append(''.join([parts[1],parts[0]] + parts[2:]))
        return normalized_label + ''.join(self._normalize_selector_text(alias) for alias in compact_aliases)

    def _selector_feature_index_matches(self, feature_index, selector):
        selected = selector.get('feature_indices', selector.get('columns', None))
        if selected is None:
            return True
        if isinstance(selected, (int, np.integer)):
            return int(feature_index) == int(selected)
        return int(feature_index) in {int(value) for value in selected}

    def _selector_colors_match(self, record_colors, selector):
        record_sequence = tuple(self._normalize_color_code(color) for color in record_colors)
        record_set = set(record_sequence)
        if 'colors' in selector:
            selected_values = self._selector_color_values(selector['colors'])
            selected_set = {self._normalize_color_code(color) for color in selected_values}
            if not selected_set.issubset(record_set):
                return False
        if 'exact_colors' in selector:
            selected_values = self._selector_color_values(selector['exact_colors'])
            selected_set = {self._normalize_color_code(color) for color in selected_values}
            if selected_set != record_set:
                return False
        if 'color_sequence' in selector:
            selected_sequence = tuple(self._normalize_color_code(color) for color in self._selector_color_values(selector['color_sequence']))
            if selected_sequence != record_sequence:
                return False
        return True

    def _selector_color_values(self, selected):
        if not isinstance(selected, str):
            return list(selected)
        lowered = selected.strip().lower()
        if lowered in ('red','orange','yellow','white','green','blue','masked'):
            return [selected]
        if ',' in selected:
            return [value.strip() for value in selected.split(',') if value.strip()]
        return list(selected.strip())

    def _w1_feature_records(self):
        if self._has_rubiks_piece_feature_layout():
            return self._rubiks_w1_feature_records()
        if hasattr(self.cube,'feature_index_to_piece_color'):
            return self._generic_w1_feature_records()
        return []

    def _rubiks_w1_feature_records(self):
        records = []
        perfect_data = np.asarray(self.cube.perfect_data).reshape(-1)
        solve_groups_by_feature = self._w1_solve_groups_by_feature()
        offset = 0
        for piece_index,piece in enumerate(self.cube.center_index):
            position_label = self._w1_piece_position_label('Center',piece)
            for local_index,color in enumerate(self.cube.colors):
                feature_index = offset + local_index
                records.append(self._w1_feature_record(
                    feature_index,'Center',position_label,(color,),perfect_data,solve_groups_by_feature,piece_index,
                ))
            offset += 6
        for piece_index,piece in enumerate(self.cube.edge_index):
            position_label = self._w1_piece_position_label('Edge',piece)
            for local_index,color in enumerate(self.cube.edge_colors):
                feature_index = offset + local_index
                records.append(self._w1_feature_record(
                    feature_index,'Edge',position_label,tuple(color),perfect_data,solve_groups_by_feature,piece_index,
                ))
            offset += 24
        for piece_index,piece in enumerate(self.cube.corner_index):
            position_label = self._w1_piece_position_label('Corner',piece)
            for local_index,color in enumerate(self.cube.corner_colors):
                feature_index = offset + local_index
                records.append(self._w1_feature_record(
                    feature_index,'Corner',position_label,tuple(color),perfect_data,solve_groups_by_feature,piece_index,
                ))
            offset += 24
        return records

    def _generic_w1_feature_records(self):
        perfect_data = np.asarray(getattr(self.cube,'perfect_data',np.zeros(self.ips))).reshape(-1)
        solve_groups_by_feature = self._w1_solve_groups_by_feature()
        records = []
        for feature_index,piece_color in self.cube.feature_index_to_piece_color.items():
            if feature_index < 0 or feature_index >= self.ips:
                continue
            piece,color = piece_color
            piece_type = self._generic_piece_type(piece)
            position_label = self._w1_piece_position_label(piece_type,piece)
            records.append(self._w1_feature_record(
                int(feature_index),
                piece_type,
                position_label,
                tuple(str(value) for value in color),
                perfect_data,
                solve_groups_by_feature,
                0,
            ))
        return records

    def _w1_feature_record(self, feature_index, piece_type, position_label, colors, perfect_data, solve_groups_by_feature, piece_index):
        return {
            'feature_index': int(feature_index),
            'piece_type': str(piece_type),
            'position_label': str(position_label),
            'colors': tuple(str(color) for color in colors),
            'correct': bool(feature_index < perfect_data.size and perfect_data[feature_index] > 0.5),
            'solve_group': solve_groups_by_feature.get(int(feature_index),''),
            'piece_index': int(piece_index),
        }

    def _w1_piece_position_label(self, piece_type, piece):
        if hasattr(self.cube,'piece_display_name'):
            return self.cube.piece_display_name(piece_type,piece)
        return f'{piece_type}-{tuple(piece)}'

    def _generic_piece_type(self, piece):
        for group_name,pieces in getattr(self.cube,'group_pieces',{}).items():
            if piece in pieces:
                return str(group_name)
        return 'Piece'

    def _w1_solve_groups_by_feature(self):
        groups = {}
        for group_name,group_vector in getattr(self.cube,'group_val',{}).items():
            name = str(group_name)
            if len(name) <= 1:
                continue
            mask = np.asarray(group_vector).reshape(-1) > 0
            for feature_index in np.flatnonzero(mask):
                groups.setdefault(int(feature_index),name)
        return groups

    def _normalize_selector_text(self, value):
        return str(value).lower().replace(' ','').replace('_','').replace('-','')

    def _normalize_color_code(self, value):
        text = str(value).strip()
        aliases = {
            'red': 'R',
            'orange': 'O',
            'yellow': 'Y',
            'white': 'W',
            'green': 'G',
            'blue': 'B',
            'masked': 'X',
        }
        lowered = text.lower()
        if lowered in aliases:
            return aliases[lowered]
        return text[:1].upper()

    def _normalize_search2_value_loss_type(self, loss_type):
        normalized = str(loss_type or 'myloss2').replace('_','').replace('-','').lower()
        aliases = {
            'myloss': 'myloss',
            'loss1': 'myloss',
            '1': 'myloss',
            'myloss2': 'myloss2',
            'myloss2distance': 'myloss2',
            'distance': 'myloss2',
            'loss2': 'myloss2',
            '2': 'myloss2',
            'myloss2pairwise': 'myloss2_pairwise',
            'pairwise': 'myloss2_pairwise',
            'margin': 'myloss2_pairwise',
            'loss2pairwise': 'myloss2_pairwise',
            '2pairwise': 'myloss2_pairwise',
        }
        if normalized not in aliases:
            raise ValueError(f'unknown Search2 value loss type: {loss_type}')
        return aliases[normalized]

    def _normalize_search2_rank_loss_apply_type(self, apply_type):
        normalized = str(apply_type or 'distance').replace('_','').replace('-','').lower()
        aliases = {
            'none': 'none',
            'off': 'none',
            'false': 'none',
            '0': 'none',
            'distance': 'distance',
            'myloss2': 'distance',
            'softtarget': 'distance',
            'pairwise': 'pairwise',
            'myloss2pairwise': 'pairwise',
            'margin': 'pairwise',
            'all': 'all',
            'both': 'all',
            'true': 'all',
            '1': 'all',
        }
        if normalized not in aliases:
            raise ValueError(f'unknown Search2 rank loss apply type: {apply_type}')
        return aliases[normalized]

    def _create_search2_value_loss(self):
        if self.search2_value_loss_type == 'myloss':
            return Myloss()
        if self.search2_value_loss_type == 'myloss2_pairwise':
            return MyLoss2Pairwise(self.search2_value_loss_margin)
        return MyLoss2()
        
        
        


    def set_perfect_val(self):
        self._mark_torch_params_dirty()
        trunk_out = self._forward_trunk(self.cube.perfect_data.reshape(-1,1),loss = False,retain_cache = False)
        value_out = self._predict_value_head(trunk_out,loss = False)
        self.perfect_val = float(value_out[0][0])
        self.clear_training_cache(collect = False)

    def _mark_torch_params_dirty(self):
        self._torch_params_dirty = True
        self._torch_params_cache = None
        self._torch_params_device = None

    def mark_params_dirty(self):
        self._mark_torch_params_dirty()
        if hasattr(self,'search3_engine'):
            self.search3_engine.reset_tree()
        if torch is not None and torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def clear_training_cache(self, collect = True):
        for layer in self.layers.values():
            self._clear_layer_cache(layer)
        extra_layers = [
            self.policy_mid,
            self.value_mid,
            self.policy_act,
            self.value_act,
            self.policy_layer,
            self.value_layer,
            self.losslayer,
            self.losslayer2,
            self.search2_rank_loss,
            self.losslayer3,
            self.search3_rank_loss,
            self.losslayer4,
            self.search3_value_sigmoid,
        ]
        if self.Batch_Normalize:
            extra_layers += [self.policy_BN,self.value_BN]
        for layer in extra_layers:
            self._clear_layer_cache(layer)
        if collect:
            gc.collect()

    def _clear_layer_cache(self, layer):
        for attr_name in ['x','out','y','t','x_bar','train_m','train_s','tokens','Q','K','V','M','s','attended','attended_sum']:
            if hasattr(layer,attr_name):
                setattr(layer,attr_name,None)

            
        
    def predict(self,x,policy = True,value = False,loss = False,retain_cache = None):
        if self._can_use_torch_predict(loss,retain_cache):
            return self._torch_predict_select(x,policy = policy,value = value)

        trunk_out = self._forward_trunk(x, loss = loss, retain_cache = retain_cache)

        if policy and not value:
            return self._predict_policy_head(trunk_out, loss = loss)

        if value and not policy:
            return self._predict_value_head(trunk_out, loss = loss)

        policy_out = self._predict_policy_head(trunk_out, loss = loss)
        value_out = self._predict_value_head(trunk_out, loss = loss)
        return np.r_[policy_out, value_out]

    def _can_use_torch_predict(self, loss = False, retain_cache = None):
        """Return True when inference can use the Torch forward path."""
        use_torch_predict = bool(getattr(self,'use_torch',False) or getattr(self,'use_torch_predict',False))
        if not use_torch_predict or torch is None or self.Batch_Normalize or loss:
            return False
        return retain_cache is not True

    def _torch_predict_select(self, x, policy = True, value = False):
        out = self._torch_predict(x)
        if policy and not value:
            return out[:-1]
        if value and not policy:
            return out[-1:]
        return out

    def myval_predict(self, x=None, state_rows=None):
        """myval 用の軽量 value 予測を返す。

        x が渡された場合は既存 myval ネットワークと同じ solved 特徴ベクトル内積を使う。
        state_rows だけの場合は makedata() を作らず state から同じ一致数を直接数える。
        """
        if state_rows is not None:
            return self.myval_predict_fast(state_rows).reshape(1, -1)
        if x is None:
            x = self.cube.makedata().reshape(-1, 1)
        value = self.params['W1'][:1] @ x
        if 'BO_V' in self.params:
            value = value + self.params['BO_V'].reshape(-1, 1)[:1]
        return value.astype('f')

    def myval_predict_fast(self, state_rows):
        """Rubik's myval を state 配列から直接計算する高速版。"""
        state_rows = np.asarray(state_rows)
        if state_rows.ndim == 1:
            state_rows = state_rows.reshape(1, -1)
        if not self._can_use_direct_myval_score():
            X = self._makedata_batch_from_states(state_rows)
            return (self.params['W1'][:1] @ X).reshape(-1).astype('f')

        scores = np.zeros(state_rows.shape[0], dtype='f')
        solved_state = self.cube.state_0

        if getattr(self.cube, 'center_num', 0) > 0:
            center_indices = []
            for face_index in range(6):
                start = 4 * (self.cube.size - 1) + self.cube.surface_num * face_index
                end = self.cube.surface_num * (face_index + 1)
                center_indices.extend(range(start, end))
            if len(center_indices) > 0:
                center_indices = np.array(center_indices, dtype=int)
                scores += np.sum(state_rows[:, center_indices] == solved_state[center_indices].reshape(1, -1), axis=1)

        for piece_indices in getattr(self.cube, 'edge_index', []):
            indices = np.array(piece_indices, dtype=int)
            scores += np.all(state_rows[:, indices] == solved_state[indices].reshape(1, -1), axis=1)

        for piece_indices in getattr(self.cube, 'corner_index', []):
            indices = np.array(piece_indices, dtype=int)
            scores += np.all(state_rows[:, indices] == solved_state[indices].reshape(1, -1), axis=1)

        if 'BO_V' in self.params:
            scores += float(self.params['BO_V'].reshape(-1)[0])
        return scores.astype('f')

    def _can_use_direct_myval_score(self):
        return all(hasattr(self.cube, attr) for attr in ('state_0', 'edge_index', 'corner_index', 'surface_num', 'center_num'))

    def _makedata_batch_from_states(self, state_rows):
        original_state = self.cube.state.copy()
        X = np.empty((self.ips, state_rows.shape[0]), dtype='f')
        try:
            for index, state_row in enumerate(state_rows):
                self.cube.state[:] = state_row
                X[:, index] = self.cube.makedata()
        finally:
            self.cube.state[:] = original_state
        return X

    def _piece_token_feature_indices(self):
        """Return feature-index blocks used as piece tokens for makedata input."""
        if self._has_generic_piece_feature_layout():
            return self._generic_piece_token_feature_indices()
        if self._looks_like_megaminx_feature_layout():
            return self._megaminx_piece_token_feature_indices()
        if self._has_rubiks_piece_feature_layout():
            return self._rubiks_piece_token_feature_indices()
        return [np.arange(self.ips,dtype = int)]

    def _has_generic_piece_feature_layout(self):
        return (
            hasattr(self.cube,'piece_feature_offsets')
            and hasattr(self.cube,'group_pieces')
        )

    def _generic_piece_token_feature_indices(self):
        feature_indices = []
        for pieces in self.cube.group_pieces.values():
            for piece in pieces:
                offset, feature_size = self.cube.piece_feature_offsets[piece]
                feature_indices.append(np.arange(offset,offset + feature_size,dtype = int))
        return feature_indices

    def _looks_like_megaminx_feature_layout(self):
        return (
            hasattr(self.cube,'corner_index')
            and hasattr(self.cube,'edge_index')
            and hasattr(self.cube,'corner_colors')
            and hasattr(self.cube,'edge_colors')
            and len(getattr(self.cube,'center_index',[])) == 0
            and len(getattr(self.cube,'corner_colors',[])) == 60
        )

    def _megaminx_piece_token_feature_indices(self):
        feature_indices = []
        offset = 0
        for _ in self.cube.corner_index:
            feature_indices.append(np.arange(offset,offset + 60,dtype = int))
            offset += 60
        for _ in self.cube.edge_index:
            feature_indices.append(np.arange(offset,offset + 60,dtype = int))
            offset += 60
        return feature_indices

    def _has_rubiks_piece_feature_layout(self):
        return all(hasattr(self.cube,attr) for attr in ('center_index','edge_index','corner_index'))

    def _rubiks_piece_token_feature_indices(self):
        feature_indices = []
        offset = 0
        for _ in self.cube.center_index:
            feature_indices.append(np.arange(offset,offset + 6,dtype = int))
            offset += 6
        for _ in self.cube.edge_index:
            feature_indices.append(np.arange(offset,offset + 24,dtype = int))
            offset += 24
        for _ in self.cube.corner_index:
            feature_indices.append(np.arange(offset,offset + 24,dtype = int))
            offset += 24
        return feature_indices

    def _forward_trunk(self, x, loss = False, retain_cache = None):
        """共有 trunk を順伝播して中間表現を返す。"""
        if retain_cache is None:
            retain_cache = loss
        out = x.copy()
        if self.residual and self.use_transformer_attention:
            return self._forward_trunk_with_attention_residual(out, loss = loss, retain_cache = retain_cache)
        if self.residual:
            for block in self.trunk_blocks:
                out = block.forward(out, loss = loss)
            return out

        for key in self.layers.keys():
            if loss and key[:2] == 'BN':
                out = self.layers[key].forward(out,True)
            elif isinstance(self.layers[key],PieceTokenSelfAttention):
                out = self.layers[key].forward(out,cache = retain_cache)
            else:
                out = self.layers[key].forward(out)
        return out

    def _forward_trunk_with_attention_residual(self, out, loss = False, retain_cache = None):
        """attention を含む trunk を residual block 単位で順伝播する。"""
        if retain_cache is None:
            retain_cache = loss
        self._residual_skip_flags = {}
        for index in range(1,len(self.Mid) + 1):
            residual = out
            aff_layer = self.layers['Aff' + str(index)]
            if isinstance(aff_layer,PieceTokenSelfAttention):
                out = aff_layer.forward(out,cache = retain_cache)
            else:
                out = aff_layer.forward(out)
            attention_key = 'Att' + str(index)
            if attention_key in self.layers:
                out = self.layers[attention_key].forward(out)
            bn_key = 'BN' + str(index)
            if self.Batch_Normalize and bn_key in self.layers:
                out = self.layers[bn_key].forward(out,loss)
            out = self.layers['Act' + str(index)].forward(out)
            use_skip = (out.shape == residual.shape)
            self._residual_skip_flags[index] = use_skip
            if use_skip:
                out = out + residual
        return out

    def _predict_policy_head(self, trunk_out, loss = False):
        """policy head を順伝播して policy 出力を返す。"""
        out = self.policy_mid.forward(trunk_out)
        if self.Batch_Normalize:
            out = self.policy_BN.forward(out,loss)
        out = self.policy_act.forward(out)
        return self.policy_layer.forward(out)

    def _predict_value_head(self, trunk_out, loss = False):
        """value head を順伝播して value 出力を返す。"""
        out = self.value_mid.forward(trunk_out)
        if self.Batch_Normalize:
            out = self.value_BN.forward(out,loss)
        out = self.value_act.forward(out)
        return self.value_layer.forward(out)

    def grad(self,x,layer = "WO_V",index = 0):
        self.predict(x,policy = True,value = True,loss = False,retain_cache = True)

        if layer == "WO_V":
            dO = self._grad_from_value_head(x.shape[1])

        elif layer == "WO_P":
            dO = self._grad_from_policy_head(x.shape[1],index)

        else:
            dO = self._grad_from_hidden_layer(layer,index)
                

        return dO

    def _grad_from_value_head(self, batch_size):
        """value head 出力から trunk 入力まで勾配を戻す。"""
        dO = np.ones((1,batch_size),dtype = 'f')
        dO = self.value_layer.backward(dO)
        dO = self.value_act.backward(dO)
        dO = self.value_mid.backward(dO)
        return self._backprop_through_trunk_layers(dO)

    def _grad_from_policy_head(self, batch_size, index):
        """policy head の指定 index から trunk 入力まで勾配を戻す。"""
        dO = np.zeros((self.ops,batch_size),dtype = "f")
        dO[index] = 1
        dO = self.policy_layer.backward(dO)
        dO = self.policy_act.backward(dO)
        dO = self.policy_mid.backward(dO)
        return self._backprop_through_trunk_layers(dO)

    def _grad_from_hidden_layer(self, layer, index):
        """指定した trunk layer の出力から入力側へ勾配を戻す。"""
        reverse_key = list(self.layers)
        reverse_key.reverse()
        backward = False
        for key in reverse_key:
            if key == layer:
                dO = np.zeros(self.layers[key].out.shape,dtype = 'f')
                dO[index] = 1
                backward = True

            if backward:
                dO = self.layers[key].backward(dO)
        return dO

    def _backprop_through_trunk_layers(self, dO):
        """共有 trunk layer 群を逆順に backward する。"""
        if self.residual and self.use_transformer_attention:
            for index in reversed(range(1,len(self.Mid) + 1)):
                skip_grad = dO.copy() if getattr(self,'_residual_skip_flags',{}).get(index,False) else 0
                dO = self.layers['Act' + str(index)].backward(dO)
                bn_key = 'BN' + str(index)
                if self.Batch_Normalize and bn_key in self.layers:
                    dO = self.layers[bn_key].backward(dO)
                attention_key = 'Att' + str(index)
                if attention_key in self.layers:
                    dO = self.layers[attention_key].backward(dO)
                dO = self.layers['Aff' + str(index)].backward(dO)
                dO = dO + skip_grad
            return dO
        if self.residual:
            for block in reversed(self.trunk_blocks):
                dO = block.backward(dO)
            return dO
        reverse_key = list(self.layers)
        reverse_key.reverse()
        for key in reverse_key:
            dO = self.layers[key].backward(dO)
        return dO
        
    def integrated_grad(self,x,steps = 200,layer = "WO_V",index = 0):
        interpolated_inputs = self._integrated_grad_inputs(x,steps)
        gradients = self.grad(interpolated_inputs,layer = layer,index = index)
        return np.average(gradients,axis = 1)

    def _integrated_grad_inputs(self, x, steps):
        """完全状態と入力状態の間を結ぶ補間入力列を作る。"""
        interpolated_inputs = np.zeros((self.ips,steps),dtype = 'f')
        for i in range(steps):
            interpolated_inputs[:,i] = i / steps * self.cube.perfect_data + (1 - i / steps) * x
        return interpolated_inputs

    

    def loss(self,d_Lis,transformation = 0,flip_inside = False):
        loss_inputs = self._build_loss_inputs(
            d_Lis,
            transformation = transformation,
            flip_inside = flip_inside,
        )
        out = self._predict_loss_outputs(loss_inputs['x'])
        return self._compute_search2_losses(
            out,
            loss_inputs['args'],
            loss_inputs['indices'],
            loss_inputs['value_columns'],
            loss_inputs['value_indices'],
            loss_inputs['value_steps_to_goal'],
        )

    def _build_loss_inputs(self, d_lis, transformation = 0, flip_inside = False):
        """Search2 学習用の入力テンソルと index 情報を構築する。"""
        indices, total_steps = self._build_loss_indices(d_lis)
        args = np.zeros(total_steps - len(d_lis),dtype = 'i')
        x = np.zeros((self.ips,total_steps))
        self._fill_loss_tensors(d_lis, transformation, flip_inside, args, x)
        value_columns,value_indices = self._build_search2_value_loss_selection(d_lis,indices)
        value_steps_to_goal = self._build_search2_value_steps_to_goal(d_lis,indices,value_columns)
        return {
            'indices': indices,
            'args': args,
            'x': x,
            'value_columns': value_columns,
            'value_indices': value_indices,
            'value_steps_to_goal': value_steps_to_goal,
        }

    def _predict_loss_outputs(self, x):
        """loss 計算用に policy/value 出力をまとめて推論する。"""
        return self.predict(x,policy = True,value = True,loss = True,retain_cache = True)

    def _compute_search2_losses(self, out, args, indices, value_columns = None, value_indices = None, value_steps_to_goal = None):
        """Search2 用の policy loss と value loss を計算する。"""
        policy_loss = self.losslayer.forward(out[:-1],args,np.zeros(0),indices)
        self._search2_value_output_size = out.shape[1]
        if value_columns is None:
            value_columns = np.arange(out.shape[1],dtype = 'i')
        self._search2_value_columns = np.asarray(value_columns,dtype = 'i')
        self._search2_value_has_rank_loss = False
        if len(self._search2_value_columns) == 0:
            self._search2_value_indices = [0]
            self._record_search2_value_loss_components('none',0.0,0.0,0.0,False)
            return (policy_loss,0.0)
        if value_indices is None:
            value_indices = indices
        self._search2_value_indices = value_indices
        if self.search2_value_loss_type == 'myloss2':
            base_loss = self.losslayer2.forward(
                out[-1:,self._search2_value_columns],
                value_indices,
                value_steps_to_goal,
                self.value_target_gamma,
            )
            base_name = 'myloss2_distance'
        else:
            base_loss = self.losslayer2.forward(out[-1:,self._search2_value_columns],value_indices)
            base_name = self.search2_value_loss_type
        value_loss = base_loss
        rank_loss = 0.0
        rank_scaled_loss = 0.0
        rank_applied = False
        if self._uses_search2_rank_mix() and self.search2_rank_loss_mix > 0.0:
            rank_loss = self.search2_rank_loss.forward(
                out[-1:,self._search2_value_columns],
                value_indices,
            )
            rank_scaled_loss = self.search2_rank_loss_mix * rank_loss
            value_loss += rank_scaled_loss
            rank_applied = True
            self._search2_value_has_rank_loss = True
        self._record_search2_value_loss_components(base_name,base_loss,rank_loss,rank_scaled_loss,rank_applied)
        return (policy_loss,value_loss)

    def _uses_search2_rank_mix(self):
        if self.search2_rank_loss_apply_type == 'none':
            return False
        if self.search2_rank_loss_apply_type == 'all':
            return self.search2_value_loss_type in ('myloss2','myloss2_pairwise')
        if self.search2_rank_loss_apply_type == 'pairwise':
            return self.search2_value_loss_type == 'myloss2_pairwise'
        return self.search2_value_loss_type == 'myloss2'

    def _empty_search2_value_loss_components(self):
        return {
            'loss_type': getattr(self,'search2_value_loss_type',''),
            'base_name': '',
            'base': 0.0,
            'rank_raw': 0.0,
            'rank_scaled': 0.0,
            'total': 0.0,
            'rank_mix': float(getattr(self,'search2_rank_loss_mix',0.0)),
            'rank_apply_type': getattr(self,'search2_rank_loss_apply_type',''),
            'rank_applied': False,
            'margin': float(getattr(self,'search2_value_loss_margin',0.0)),
        }

    def _record_search2_value_loss_components(self, base_name, base_loss, rank_loss, rank_scaled_loss, rank_applied):
        total = float(base_loss) + float(rank_scaled_loss)
        self._last_search2_value_loss_components = {
            'loss_type': self.search2_value_loss_type,
            'base_name': str(base_name),
            'base': float(base_loss),
            'rank_raw': float(rank_loss),
            'rank_scaled': float(rank_scaled_loss),
            'total': total,
            'rank_mix': float(self.search2_rank_loss_mix),
            'rank_apply_type': self.search2_rank_loss_apply_type,
            'rank_applied': bool(rank_applied),
            'margin': float(self.search2_value_loss_margin),
        }

    def _build_loss_indices(self, d_Lis):
        """Search2 loss 用に各サンプルの区切り index を作る。"""
        Indices = [0]
        N = 0
        for d in d_Lis:
            N += len(d.moves) + 1
            Indices.append(N)
        return Indices, N

    def _build_search2_value_loss_selection(self, d_lis, indices):
        """Return output columns/index boundaries used by the Search2 value loss."""
        columns = []
        value_indices = [0]
        total = 0
        for data_index,data_item in enumerate(d_lis):
            start = indices[data_index]
            end = indices[data_index + 1]
            if not self._uses_search2_value_loss_for_sample(data_item):
                continue
            columns.extend(range(start,end))
            total += end - start
            value_indices.append(total)
        return np.asarray(columns,dtype = 'i'), value_indices

    def _build_search2_value_steps_to_goal(self, d_lis, indices, value_columns):
        if len(value_columns) == 0:
            return np.zeros(0,dtype = 'f')
        steps_by_column = np.zeros(indices[-1],dtype = 'f')
        for data_index,data_item in enumerate(d_lis):
            start = indices[data_index]
            end = indices[data_index + 1]
            steps_by_column[start:end] = self._steps_to_goal_for_search2_data(data_item,end - start)
        return steps_by_column[value_columns]

    def _steps_to_goal_for_search2_data(self, data_item, expected_count):
        steps = getattr(data_item,'steps_to_goal',None)
        if steps is None:
            steps = getattr(data_item,'remaining_steps',None)
        if steps is None or len(steps) != expected_count:
            steps = tuple(range(expected_count - 1,-1,-1))
        return np.asarray(steps,dtype = 'f')

    def _uses_search2_value_loss_for_sample(self, data_item):
        """MyLoss skips value training for data solved by MyLoss2; MyLoss2 uses all data."""
        if self.search2_value_loss_type == 'myloss2':
            return True
        source_loss_type = getattr(data_item,'source_search2_value_loss_type',None)
        if source_loss_type is None:
            return True
        try:
            return self._normalize_search2_value_loss_type(source_loss_type) != 'myloss2'
        except ValueError:
            return True

    def _fill_loss_tensors(self, d_Lis, transformation, flip_inside, args, x):
        """Search2 学習データ列をまとめて入力 tensor に展開する。"""
        arg_index = 0
        state_index = 0
        for data_item in d_Lis:
            arg_index, state_index = self._append_loss_item_tensors(
                data_item,
                transformation,
                flip_inside,
                args,
                x,
                arg_index,
                state_index,
            )

    def _append_loss_item_tensors(
        self,
        data_item,
        transformation,
        flip_inside,
        args,
        x,
        arg_index,
        state_index,
    ):
        """Search2 データ 1 件分の入力 tensor を追記する。"""
        scramble, moves = self._transform_loss_moves(data_item, transformation, flip_inside)
        self._apply_scramble_for_loss(scramble)
        self._write_loss_move_args(moves,args,arg_index)
        x[:,state_index] = self.cube.makedata()
        next_state_index = self._apply_moves_and_collect(moves, x, state_index)
        return arg_index + len(moves), next_state_index

    def _write_loss_move_args(self, moves, args, start_index):
        """Search2 の move 列を policy 教師ラベル列へ書き込む。"""
        args[start_index:start_index + len(moves)] = np.array(
            [self.cube.key_to_num[m] for m in moves]
        )

    def _transform_loss_moves(self, d, transformation, flip_inside):
        """Search2 データ 1 件分の scramble / move を対称変換する。"""
        scramble = self.cube.transform(d.scramble,transformation,flip_inside)
        moves = self.cube.transform(d.moves,transformation,flip_inside)
        return scramble, moves

    def _apply_scramble_for_loss(self, scramble):
        """loss 計算前に cube を指定 scramble 状態へ戻す。"""
        self.cube.reset()
        self.cube.scramble(0,scramble)

    def _apply_moves_and_collect(self, moves, x, start_index):
        """move を順に適用しながら途中局面の特徴量を追記する。"""
        i = 1
        for m in moves:
            self.cube.make_move(m)
            x[:,start_index + i] = self.cube.makedata()
            i += 1
        return start_index + len(moves) + 1

    def learn(self,transformation = 0,flip_inside = False, progress_callback = None):
        training_result = self._run_training_epochs(
            indices = self.indices,
            data_source = self.datas,
            train_batch = self._train_batch,
            state_count_fn = lambda data_item:max(1,len(data_item.moves) + 1),
            transformation = transformation,
            flip_inside = flip_inside,
            progress_callback = progress_callback,
        )
        err, err2, new_indices, original_len, epoch_num, l1_max = training_result

        self.indices = new_indices
        self.set_perfect_val()
        if original_len > 0:
            self.lr_C = min(1,err/original_len/10)

        if epoch_num == 0:
            return (0,0,0,0,0)

        self._finalize_training(progress_callback = progress_callback)
        return (err / epoch_num,err2 / epoch_num,len(self.indices),original_len)

    def loss_search3(self,d_Lis,transformation = 0,flip_inside = False):
        search3_inputs = self._build_search3_loss_inputs(
            d_Lis,
            transformation = transformation,
            flip_inside = flip_inside,
        )
        out = self._predict_loss_outputs(search3_inputs['x'])
        losses = self._compute_search3_losses(
            out,
            search3_inputs['policy_targets'],
            search3_inputs['value_targets'],
            search3_inputs['sample_weights'],
            search3_inputs['value_indices'],
            search3_inputs['policy_weights'],
        )
        self._last_search3_debug_summary = self._build_search3_debug_summary(out, search3_inputs)
        return losses

    def _build_search3_loss_inputs(self, d_lis, transformation = 0, flip_inside = False):
        """Search3 学習用の入力テンソル群を構築する。"""
        x, policy_targets, value_targets, sample_weights, policy_weights, step_metadata, value_indices = self._build_search3_tensors(
            d_lis,
            transformation = transformation,
            flip_inside = flip_inside,
        )
        return {
            'x': x,
            'policy_targets': policy_targets,
            'value_targets': value_targets,
            'sample_weights': sample_weights,
            'policy_weights': policy_weights,
            'step_metadata': step_metadata,
            'value_indices': value_indices,
        }

    def _compute_search3_losses(self, out, policy_targets, value_targets, sample_weights, value_indices = None, policy_weights = None):
        """Search3 用の policy loss と value loss を計算する。"""
        if policy_weights is None:
            policy_weights = sample_weights
        policy_loss = self.losslayer4.forward(out[:-1],policy_targets,policy_weights)
        value_loss = self.losslayer3.forward(out[-1:],value_targets,sample_weights)
        self._search3_value_has_rank_loss = False
        if self.search3_rank_loss_mix > 0.0 and value_indices is not None and len(value_indices) > 1:
            value_loss += self.search3_rank_loss_mix * self.search3_rank_loss.forward(out[-1:],value_indices)
            self._search3_value_has_rank_loss = True
        return (policy_loss,value_loss)

    def _build_search3_debug_summary(self, out, search3_inputs):
        """Original Search3 の policy/value 予測と target の差分を短いログにまとめる。"""
        targets = search3_inputs['value_targets'].reshape(-1)
        weights = search3_inputs['sample_weights'].reshape(-1)
        policy_weights = search3_inputs.get('policy_weights',search3_inputs['sample_weights']).reshape(-1)
        policy_targets = search3_inputs['policy_targets']
        metadata = search3_inputs.get('step_metadata',[])
        if targets.size == 0:
            return 'Search3 value debug: empty batch'

        raw_values = np.clip(out[-1:].reshape(-1),-60.0,60.0)
        predictions = 1.0 / (1.0 + np.exp(-raw_values))
        policy_probs = self._softmax_columns(out[:-1])
        policy_ce = -np.sum(policy_targets * np.log(policy_probs + 1.0e-7),axis = 0)
        active_policy = policy_weights > 0.0
        target_moves = np.argmax(policy_targets,axis = 0)
        pred_moves = np.argmax(policy_probs,axis = 0)
        target_probs = policy_probs[target_moves,np.arange(policy_probs.shape[1])]
        top_hits = (target_moves == pred_moves)
        target_entropy = -np.sum(policy_targets * np.log(policy_targets + 1.0e-7),axis = 0)
        diff = predictions - targets
        abs_diff = np.abs(diff)
        denom = max(float(np.sum(weights)),1.0e-8)
        weighted_mse = float(np.sum(weights * diff ** 2) / denom)
        weighted_mae = float(np.sum(weights * abs_diff) / denom)
        mode_counts = Counter(meta.get('search_mode','') for meta in metadata)
        end_reason_counts = Counter(meta.get('end_reason',None) for meta in metadata)
        source_successes = sum(1 for meta in metadata if meta.get('source_succeeded',False))

        lines = [
            (
                'Search3 value debug: '
                f'steps={targets.size} modes={dict(mode_counts)} end_reasons={dict(end_reason_counts)} '
                f'source_succeeded={source_successes}/{len(metadata)} '
                f'target={float(np.min(targets)):.4f}/{float(np.mean(targets)):.4f}/{float(np.max(targets)):.4f} '
                f'pred={float(np.min(predictions)):.4f}/{float(np.mean(predictions)):.4f}/{float(np.max(predictions)):.4f} '
                f'diff_mean={float(np.mean(diff)):.4f} mae={weighted_mae:.4f} mse={weighted_mse:.4f} '
                f'bad>|.10|={int(np.sum(abs_diff > 0.10))} bad>|.20|={int(np.sum(abs_diff > 0.20))}'
            ),
            (
                'Search3 policy debug: '
                f'top1={self._masked_mean(top_hits,active_policy):.3f} '
                f'target_prob={self._masked_min_mean_max_text(target_probs,active_policy)} '
                f'ce={self._masked_min_mean_max_text(policy_ce,active_policy)} '
                f'target_entropy={self._masked_mean(target_entropy,active_policy):.4f}'
            )
        ]

        worst_indices = np.argsort(abs_diff)[-3:][::-1]
        for rank, step_index in enumerate(worst_indices,1):
            meta = metadata[step_index] if step_index < len(metadata) else {}
            stored_value = meta.get('stored_value')
            stored_text = 'None' if stored_value is None else f'{float(stored_value):.4f}'
            lines.append(
                '  worst#{rank}: mode={mode} reason={reason} source_ok={source_ok} solve_ok={solve_ok} '
                'move={move} step={step}/{count} '
                'target={target:.4f} pred={pred:.4f} diff={diff:.4f} '
                'stored={stored} scramble_len={scramble_len} key={key} group={group}'.format(
                    rank = rank,
                    mode = meta.get('search_mode',''),
                    reason = meta.get('end_reason',None),
                    source_ok = meta.get('source_succeeded',False),
                    solve_ok = meta.get('solve_succeeded',False),
                    move = meta.get('move_label',''),
                    step = int(meta.get('move_index',0)),
                    count = int(meta.get('move_count',0)),
                    target = float(targets[step_index]),
                    pred = float(predictions[step_index]),
                    diff = float(diff[step_index]),
                    stored = stored_text,
                    scramble_len = int(meta.get('scramble_count',0)),
                    key = meta.get('perfect_key',None),
                    group = meta.get('top_group',None),
                )
            )
        policy_rank_values = np.where(active_policy,policy_ce,-np.inf)
        worst_policy_indices = np.argsort(policy_rank_values)[-3:][::-1]
        for rank, step_index in enumerate(worst_policy_indices,1):
            if not active_policy[step_index]:
                continue
            meta = metadata[step_index] if step_index < len(metadata) else {}
            lines.append(
                '  policy_worst#{rank}: mode={mode} reason={reason} source_ok={source_ok} solve_ok={solve_ok} '
                'move={move} target_move={target_move} '
                'pred_move={pred_move} target_prob={target_prob:.4f} ce={ce:.4f} '
                'target_entropy={entropy:.4f} step={step}/{count} key={key} group={group}'.format(
                    rank = rank,
                    mode = meta.get('search_mode',''),
                    reason = meta.get('end_reason',None),
                    source_ok = meta.get('source_succeeded',False),
                    solve_ok = meta.get('solve_succeeded',False),
                    move = meta.get('move_label',''),
                    target_move = self.cube.move_keys[int(target_moves[step_index])],
                    pred_move = self.cube.move_keys[int(pred_moves[step_index])],
                    target_prob = float(target_probs[step_index]),
                    ce = float(policy_ce[step_index]),
                    entropy = float(target_entropy[step_index]),
                    step = int(meta.get('move_index',0)),
                    count = int(meta.get('move_count',0)),
                    key = meta.get('perfect_key',None),
                    group = meta.get('top_group',None),
                )
            )
        return '\n'.join(lines)

    def _masked_mean(self, values, mask):
        values = np.asarray(values)
        mask = np.asarray(mask,dtype = bool)
        if not np.any(mask):
            return 0.0
        return float(np.mean(values[mask]))

    def _masked_min_mean_max_text(self, values, mask):
        values = np.asarray(values)
        mask = np.asarray(mask,dtype = bool)
        if not np.any(mask):
            return '0.0000/0.0000/0.0000'
        selected = values[mask]
        return f'{float(np.min(selected)):.4f}/{float(np.mean(selected)):.4f}/{float(np.max(selected)):.4f}'

    def _softmax_columns(self, x):
        shifted = x - np.max(x,axis = 0,keepdims = True)
        exp_x = np.exp(shifted)
        return exp_x / np.sum(exp_x,axis = 0,keepdims = True)

    def _build_search3_tensors(self, d_Lis, transformation = 0, flip_inside = False):
        """Search3 学習用の state / policy / value tensor をまとめて作る。"""
        total_steps = self._search3_total_steps(d_Lis)
        x = np.zeros((self.ips,total_steps),dtype = 'f')
        policy_targets = np.zeros((self.ops,total_steps),dtype = 'f')
        value_targets = np.zeros((1,total_steps),dtype = 'f')
        sample_weights = np.ones((1,total_steps),dtype = 'f')
        policy_weights = np.ones((1,total_steps),dtype = 'f')
        step_metadata = []
        value_indices = [0]
        step_index = 0

        for data_item in d_Lis:
            item_start = step_index
            step_index = self._append_search3_item_tensors(
                data_item,
                x,
                policy_targets,
                value_targets,
                sample_weights,
                policy_weights,
                step_metadata,
                step_index,
                transformation = transformation,
                flip_inside = flip_inside,
            )
            if step_index > item_start:
                value_indices.append(step_index)

        return x, policy_targets, value_targets, sample_weights, policy_weights, step_metadata, value_indices

    def _append_search3_item_tensors(
        self,
        data_item,
        x,
        policy_targets,
        value_targets,
        sample_weights,
        policy_weights,
        step_metadata,
        step_index,
        transformation = 0,
        flip_inside = False,
    ):
        """1 件分の Search3 学習データをテンソルへ追記する。"""
        scramble, moves = self._transform_search3_item(data_item,transformation,flip_inside)
        if len(moves) == 0:
            return step_index

        rewards = self._search3_rewards_array(data_item)
        self._reset_cube_to_search3_scramble(scramble)

        for move_index, move_label in enumerate(moves):
            self._write_search3_step_tensors(
                data_item,
                moves,
                rewards,
                move_index,
                x,
                policy_targets,
                value_targets,
                sample_weights,
                policy_weights,
                step_index,
                transformation = transformation,
                flip_inside = flip_inside,
            )
            step_metadata.append(self._search3_step_metadata(data_item, move_index, move_label))
            self.cube.make_move(move_label)
            step_index += 1

        self._write_search3_terminal_tensors(
            data_item,
            rewards,
            len(moves),
            x,
            policy_targets,
            value_targets,
            sample_weights,
            policy_weights,
            step_index,
        )
        step_metadata.append(self._search3_step_metadata(data_item, len(moves), '<terminal>'))
        step_index += 1

        return step_index

    def _search3_step_metadata(self, data_item, move_index, move_label):
        stored_trace = getattr(data_item,'value_trace',[])
        stored_value = None
        if move_index < len(stored_trace):
            stored_value = stored_trace[move_index]
        return {
            'search_mode': getattr(data_item,'search_mode',''),
            'move_index': move_index,
            'move_label': move_label,
            'move_count': len(getattr(data_item,'moves',())),
            'terminal': move_index >= len(getattr(data_item,'moves',())),
            'scramble_count': len(getattr(data_item,'scramble',())),
            'stored_value': stored_value,
            'perfect_key': getattr(data_item,'perfect_key',None),
            'top_group': getattr(data_item,'top_group',None),
            'end_reason': getattr(data_item,'end_reason',None),
            'source_succeeded': getattr(data_item,'source_succeeded',getattr(data_item,'succeeded',False)),
            'solve_succeeded': getattr(data_item,'solve_succeeded',getattr(data_item,'succeeded',False)),
        }

    def _transform_search3_item(self, data_item, transformation, flip_inside):
        """Search3 データ 1 件分の scramble と move を対称変換する。"""
        scramble = self.cube.transform(data_item.scramble,transformation,flip_inside)
        moves = self.cube.transform(data_item.moves,transformation,flip_inside)
        return scramble, moves

    def _search3_rewards_array(self, data_item):
        """Search3 value target 計算に使う rewards 配列を整形する。"""
        return np.array(data_item.rewards,dtype = 'f').reshape(-1)

    def _reset_cube_to_search3_scramble(self, scramble):
        """Search3 学習データの開始局面へ cube を戻す。"""
        self.cube.reset()
        self.cube.scramble(0,scramble)

    def _write_search3_step_tensors(
        self,
        data_item,
        moves,
        rewards,
        move_index,
        x,
        policy_targets,
        value_targets,
        sample_weights,
        policy_weights,
        step_index,
        transformation = 0,
        flip_inside = False,
    ):
        """Search3 の 1 手分の特徴量と target を書き込む。"""
        x[:,step_index] = self.cube.makedata()
        policy_targets[:,step_index] = self._search3_policy_target(
            data_item,
            moves,
            move_index,
            transformation,
            flip_inside,
        )
        value_targets[0,step_index] = self._search3_value_target(
            data_item,
            rewards,
            move_index,
            len(moves),
        )
        sample_weights[0,step_index] = self._search3_step_weight(data_item,move_index)
        policy_weights[0,step_index] = sample_weights[0,step_index]

    def _write_search3_terminal_tensors(
        self,
        data_item,
        rewards,
        move_count,
        x,
        policy_targets,
        value_targets,
        sample_weights,
        policy_weights,
        step_index,
    ):
        """Search3 の終端状態を value 専用 target として書き込む。"""
        x[:,step_index] = self.cube.makedata()
        policy_targets[:,step_index] = self._uniform_search3_policy_target()
        value_targets[0,step_index] = self._search3_value_target(
            data_item,
            rewards,
            move_count,
            move_count,
        )
        sample_weights[0,step_index] = self._search3_terminal_step_weight(data_item)
        policy_weights[0,step_index] = 0.0

    def _search3_total_steps(self, d_Lis):
        """Search3 データ全体で必要な step 数を数える。"""
        total_steps = 0
        for data_item in d_Lis:
            move_count = len(data_item.moves)
            if move_count > 0:
                total_steps += move_count + 1
        return total_steps

    def _normalize_search3_policy_target(self, policy_target):
        """policy target を確率分布に正規化する。"""
        if policy_target is None:
            return self._uniform_search3_policy_target()
        normalized_target = np.array(policy_target,dtype = 'f').reshape(-1)
        total = np.sum(normalized_target)
        if total <= 0:
            return self._uniform_search3_policy_target()
        return normalized_target / total

    def _uniform_search3_policy_target(self):
        """Search3 の policy target が無いときの一様分布を返す。"""
        return np.ones(self.ops,dtype = 'f') / self.ops

    def _search3_policy_target(self, data_item, moves, move_index, transformation, flip_inside):
        if self._use_soft_search3_policy_target(
            data_item,
            move_index,
            transformation,
            flip_inside,
        ):
            return self._transform_search3_policy_target(
                data_item.policy_target,
                transformation,
                flip_inside,
            )
        return self._one_hot_search3_policy(moves[move_index])

    def _use_soft_search3_policy_target(self, data_item, move_index, transformation, flip_inside):
        """soft target をそのまま使う条件か判定する。"""
        return (
            data_item.search_mode == 'search3'
            and move_index == 0
            and data_item.policy_target is not None
        )

    def _transform_search3_policy_target(self, policy_target, transformation, flip_inside):
        """対称変換後の move 空間に合わせて soft policy target を並べ替える。"""
        normalized_target = self._normalize_search3_policy_target(policy_target)
        if transformation == 0 and not flip_inside:
            return normalized_target

        transformed_target = np.zeros_like(normalized_target)
        move_index_map = self._search3_policy_move_index_map(transformation,flip_inside)
        for original_index, transformed_index in move_index_map.items():
            transformed_target[transformed_index] = normalized_target[original_index]
        return transformed_target

    def _search3_policy_move_index_map(self, transformation, flip_inside):
        """move index を対称変換後の index へ写す対応表を返す。"""
        cache_key = (transformation, bool(flip_inside))
        if cache_key in self._search3_policy_transform_cache:
            return self._search3_policy_transform_cache[cache_key]

        move_index_map = {}
        for move_label in self.cube.move_keys:
            transformed_move = self.cube.transform((move_label,),transformation,flip_inside)
            transformed_label = transformed_move[0]
            move_index_map[self.cube.key_to_num[move_label]] = self.cube.key_to_num[transformed_label]

        self._search3_policy_transform_cache[cache_key] = move_index_map
        return move_index_map

    def _one_hot_search3_policy(self, move_label):
        """指定手を one-hot の policy target に変換する。"""
        policy_target = np.zeros((self.ops,),dtype = 'f')
        policy_target[self.cube.key_to_num[move_label]] = 1.0
        return policy_target

    def _search3_value_target(self, data_item, rewards, move_index, move_count):
        explicit_target = self._explicit_search3_value_target(data_item,move_index,move_count)
        if explicit_target is not None:
            return explicit_target
        return self._fallback_search3_value_target(data_item,rewards,move_index,move_count)

    def _explicit_search3_value_target(self, data_item, move_index, move_count):
        """data_item に value_targets があればそこから target を取る。"""
        value_targets = getattr(data_item,'value_targets',None)
        if value_targets is None:
            return None
        normalized_targets = np.array(value_targets,dtype = 'f').reshape(-1)
        if move_index < normalized_targets.size:
            return normalized_targets[move_index]
        if move_index == move_count and normalized_targets.size == move_count and normalized_targets.size > 0:
            return self._terminal_target_from_previous(normalized_targets[-1])
        if normalized_targets.size > 0:
            return normalized_targets[-1]
        return None

    def _terminal_target_from_previous(self, previous_target):
        """旧Search3データの終端targetを、最後の手前targetから推定する。"""
        gamma = max(float(getattr(self,'value_target_gamma',1.0)),1.0e-8)
        return min(1.0,float(previous_target) / gamma)

    def _fallback_search3_value_target(self, data_item, rewards, move_index, move_count):
        """value_targets が無い場合の reward/value_trace/gamma fallback を返す。"""
        if move_index == move_count:
            terminal_target = self._fallback_search3_terminal_value_target(data_item,rewards,move_count)
            if terminal_target is not None:
                return terminal_target

        reward_target = self._reward_based_search3_value_target(rewards,move_index)
        if reward_target is not None:
            return reward_target

        bootstrap_target = self._bootstrap_search3_value_target(data_item)
        if bootstrap_target is not None:
            return bootstrap_target

        return self._discounted_search3_value_target(move_count,move_index)

    def _fallback_search3_terminal_value_target(self, data_item, rewards, move_count):
        if move_count < rewards.size:
            return rewards[move_count]
        if rewards.size == move_count and rewards.size > 0:
            return self._terminal_target_from_previous(rewards[-1])
        value_trace = getattr(data_item,'value_trace',[])
        if len(value_trace) > move_count:
            return value_trace[move_count]
        if len(value_trace) > 0:
            return value_trace[-1]
        best_value = getattr(data_item,'best_value',None)
        if best_value is not None:
            return best_value
        return None

    def _reward_based_search3_value_target(self, rewards, move_index):
        """reward 列があれば、その step に対応する target を返す。"""
        if move_index < rewards.size:
            return rewards[move_index]
        if rewards.size > 0:
            return rewards[-1]
        return None

    def _bootstrap_search3_value_target(self, data_item):
        """value_trace から bootstrap 用 target を取り出す。"""
        if len(data_item.value_trace) > 0:
            return data_item.value_trace[-1]
        return None

    def _discounted_search3_value_target(self, move_count, move_index):
        """明示 target が無い場合の gamma fallback を返す。"""
        return self.value_target_gamma ** max(move_count - move_index, 0)

    def _search3_step_weight(self, data_item, move_index):
        """Search3 学習の 1 手分 sample weight を返す。"""
        sample_weight = getattr(data_item,'sample_weight',1.0)
        if data_item.search_mode == 'search3' and move_index > 0:
            return sample_weight * 0.7
        return sample_weight

    def _search3_terminal_step_weight(self, data_item):
        """終端状態はpolicyを持たないので、valueのアンカーとして通常weightを使う。"""
        return getattr(data_item,'sample_weight',1.0)

    def learn_search3(self,transformation = 0,flip_inside = False, progress_callback = None):
        training_result = self._run_training_epochs(
            indices = self.indices_search3,
            data_source = self.datas_search3,
            train_batch = self._train_batch_search3,
            state_count_fn = lambda data_item:max(1,len(data_item.moves) + 1),
            transformation = transformation,
            flip_inside = flip_inside,
            progress_callback = progress_callback,
        )
        err, err2, new_indices, original_len, epoch_num, l1_max = training_result

        self.indices_search3 = new_indices
        if epoch_num == 0:
            return (0,0,0,0)

        self.set_perfect_val()
        if original_len > 0:
            self.lr_C = min(1,err/original_len/10)

        self._finalize_training(progress_callback = progress_callback)
        return (err / epoch_num,err2 / epoch_num,len(self.indices_search3),original_len,l1_max)

    def _run_training_epochs(self, indices, data_source, train_batch, state_count_fn, transformation, flip_inside, progress_callback = None):
        """index 列を batch 学習して、残す index と誤差集計を返す。"""
        batch_size = max(1,int(getattr(self,'train_batch_size',100)))
        state_batch_size = max(0,int(getattr(self,'train_state_batch_size',0)))
        pack_state_batch_size = self._training_pack_state_batch_size(state_batch_size)
        if int(getattr(self,'train_max_batches',0)) > 0:
            batches,remainder_indices = self._build_sampled_training_batches(
                indices,
                data_source,
                batch_size,
                pack_state_batch_size,
                state_count_fn,
            )
        else:
            batches,remainder_indices = self._build_training_batches(
                indices,
                data_source,
                batch_size,
                pack_state_batch_size,
                state_count_fn,
            )
        epoch_state = self._init_training_epoch_state(indices,len(batches))
        self._report_training_sample(progress_callback)

        for epoch_index, batch_indices in self._iter_training_batches(batches):
            epoch_state['epoch_index'] = epoch_index
            epoch_state['progress_callback'] = progress_callback
            self._report_training_progress(
                progress_callback,
                epoch_index,
                len(batches),
                batch_indices,
                data_source,
                state_count_fn,
            )
            batch_data = [data_source[n] for n in batch_indices]
            losses, epoch_state = train_batch(
                batch_data,
                batch_indices,
                transformation,
                flip_inside,
                epoch_state,
            )
            epoch_state['err'] += losses[0]
            epoch_state['err2'] += losses[1]
            if self._should_keep_training_batch(losses):
                epoch_state['new_indices'] += batch_indices

        epoch_state['new_indices'] += remainder_indices
        self._report_final_search2_value_debug(epoch_state)
        return (
            epoch_state['err'],
            epoch_state['err2'],
            epoch_state['new_indices'],
            epoch_state['original_len'],
            epoch_state['epoch_num'],
            epoch_state['l1_max'],
        )

    def _training_pack_state_batch_size(self, state_batch_size):
        """Torch piece-token training uses state_batch_size as a micro-batch limit."""
        if self._uses_torch_microbatch_training():
            return 0
        return state_batch_size

    def _uses_torch_microbatch_training(self):
        return (
            self._can_use_torch_training()
            and getattr(self,'use_piece_tokens',False)
            and int(getattr(self,'train_state_batch_size',0)) > 0
        )

    def _report_training_progress(self, progress_callback, batch_index, batch_count, batch_indices, data_source, state_count_fn):
        """学習進捗を callback 経由で通知する。"""
        if progress_callback is None:
            return
        max_batches = int(getattr(self,'train_max_batches',0))
        if max_batches > 0:
            state_count = sum(int(state_count_fn(data_source[data_index])) for data_index in batch_indices)
            progress_callback(
                f'batch {batch_index + 1}/{batch_count} '
                f'data={len(batch_indices)} states={state_count} '
                f'index={min(batch_indices)}-{max(batch_indices)}'
            )
            return
        if batch_index % 20 == 0:
            progress_callback(f'batch {batch_index + 1}/{batch_count}')

    def _report_training_sample(self, progress_callback):
        """サンプル学習時の選択batch数をログに出す。"""
        max_batches = int(getattr(self,'train_max_batches',0))
        summary = getattr(self,'_last_training_sample_summary',None)
        if progress_callback is None or max_batches <= 0 or not summary:
            return
        progress_callback(
            'sampled: '
            f'batches={summary["selected_batches"]}/{summary["original_batches"]} '
            f'data={summary["selected_items"]}/{summary["original_items"]} '
            f'states={summary["selected_states"]} '
            f'recent_batches={summary["recent_batches"]} '
            f'random_batches={summary["random_batches"]} '
            f'kept_for_later={summary["remainder_items"]} '
            f'index={summary["selected_index_min"]}-{summary["selected_index_max"]}'
        )

    def _maybe_report_search3_value_debug(self, epoch_state, value_loss):
        """Search3 value debug を最初の数 batch と以後一定間隔で GUI log に流す。"""
        progress_callback = epoch_state.get('progress_callback')
        if progress_callback is None:
            return
        epoch_index = int(epoch_state.get('epoch_index',0))
        if epoch_index >= 3 and epoch_index % 20 != 0:
            return
        summary = getattr(self,'_last_search3_debug_summary',None)
        if summary is None:
            return
        progress_callback(f'{summary}\n  batch_value_loss_per_item={float(value_loss):.6f}')

    def _report_final_search2_value_debug(self, epoch_state):
        """Search2 value loss の内訳を学習終了時に GUI log へ流す。"""
        progress_callback = epoch_state.get('progress_callback')
        if progress_callback is None:
            return
        epoch_components = epoch_state.get('search2_value_loss_components')
        if not epoch_components:
            return
        progress_callback(
            self._format_search2_value_loss_components(
                epoch_components,
                item_count = int(epoch_components.get('items',1) or 1),
                label = 'final_avg',
            )
        )

    def _format_search2_value_loss_components(self, components, item_count = 1, label = 'batch'):
        denom = max(1,int(item_count))
        base_name = components.get('base_name','value')
        return (
            f'Search2 value loss debug ({label}): '
            f'type={components.get("loss_type","")} '
            f'margin={float(components.get("margin",0.0)):.4f} '
            f'rank_mix={float(components.get("rank_mix",0.0)):.4f} '
            f'rank_apply={components.get("rank_apply_type","")} '
            f'rank_applied={bool(components.get("rank_applied",False))}\n'
            f'  {base_name}={float(components.get("base",0.0)) / denom:.6f} '
            f'myloss_rank_raw={float(components.get("rank_raw",0.0)) / denom:.6f} '
            f'myloss_rank_mix={float(components.get("rank_scaled",0.0)) / denom:.6f} '
            f'total={float(components.get("total",0.0)) / denom:.6f}'
        )

    def _accumulate_search2_value_loss_components(self, epoch_state, components, item_count):
        epoch_state['search2_value_loss_components'] = self._merged_search2_value_loss_components(
            epoch_state.get('search2_value_loss_components'),
            components,
            item_count,
        )

    def _merged_search2_value_loss_components(self, current, components, item_count):
        if components is None:
            return current
        if current is None:
            current = {
                'loss_type': components.get('loss_type',''),
                'base_name': components.get('base_name',''),
                'base': 0.0,
                'rank_raw': 0.0,
                'rank_scaled': 0.0,
                'total': 0.0,
                'rank_mix': float(components.get('rank_mix',0.0)),
                'rank_apply_type': components.get('rank_apply_type',''),
                'rank_applied': False,
                'margin': float(components.get('margin',0.0)),
                'items': 0,
            }
        for key in ('base','rank_raw','rank_scaled','total'):
            current[key] = float(current.get(key,0.0)) + float(components.get(key,0.0))
        current['items'] = int(current.get('items',0)) + int(item_count)
        current['loss_type'] = components.get('loss_type',current.get('loss_type',''))
        current['base_name'] = components.get('base_name',current.get('base_name',''))
        current['rank_mix'] = float(components.get('rank_mix',current.get('rank_mix',0.0)))
        current['rank_apply_type'] = components.get('rank_apply_type',current.get('rank_apply_type',''))
        current['rank_applied'] = bool(current.get('rank_applied',False) or components.get('rank_applied',False))
        current['margin'] = float(components.get('margin',current.get('margin',0.0)))
        return current

    def _build_training_batches(self, indices, data_source, batch_size, state_batch_size, state_count_fn):
        """item数と必要ならstate数の上限に従って学習batchを作る。"""
        shuffled_indices = list(indices)
        random.shuffle(shuffled_indices)
        self._last_training_sample_summary = None
        return self._pack_training_batches(
            shuffled_indices,
            data_source,
            batch_size,
            state_batch_size,
            state_count_fn,
        )

    def _pack_training_batches(self, ordered_indices, data_source, batch_size, state_batch_size, state_count_fn, max_batches = None):
        """指定順のindexをbatchへ詰める。max_batches指定時は残りをremainderへ返す。"""
        if state_batch_size <= 0:
            epoch_num = len(ordered_indices) // batch_size
            if max_batches is not None:
                epoch_num = min(epoch_num,max_batches)
            batches = [
                ordered_indices[epoch_index * batch_size:(epoch_index + 1) * batch_size]
                for epoch_index in range(epoch_num)
            ]
            return batches,ordered_indices[batch_size * epoch_num:]

        batches = []
        current = []
        current_states = 0
        for order_index,data_index in enumerate(ordered_indices):
            if max_batches is not None and len(batches) >= max_batches:
                return batches,current + ordered_indices[order_index:]
            data_item = data_source[data_index]
            item_states = int(state_count_fn(data_item))
            exceeds_item_limit = len(current) >= batch_size
            exceeds_state_limit = current and current_states + item_states > state_batch_size
            if exceeds_item_limit or exceeds_state_limit:
                batches.append(current)
                if max_batches is not None and len(batches) >= max_batches:
                    return batches,ordered_indices[order_index:]
                current = []
                current_states = 0
            current.append(data_index)
            current_states += item_states
        if current:
            batches.append(current)
        return batches,[]

    def _build_sampled_training_batches(self, indices, data_source, batch_size, state_batch_size, state_count_fn):
        """Transformer向けに、新しいdataとランダムdataを分けてsampleする。"""
        all_indices = list(indices)
        max_batches = int(getattr(self,'train_max_batches',0))
        recent_ratio = min(max(float(getattr(self,'train_recent_ratio',0.0)),0.0),1.0)
        recent_count = min(max_batches,int(round(max_batches * recent_ratio)))
        random_count = max_batches - recent_count
        original_batches,_ = self._pack_training_batches(
            all_indices,
            data_source,
            batch_size,
            state_batch_size,
            state_count_fn,
        )

        recent_order = sorted(all_indices,reverse = True)
        recent_batches,_ = self._pack_training_batches(
            recent_order,
            data_source,
            batch_size,
            state_batch_size,
            state_count_fn,
            max_batches = recent_count,
        )
        recent_items = set(self._flatten_batches(recent_batches))

        random_order = [data_index for data_index in all_indices if data_index not in recent_items]
        random.shuffle(random_order)
        random_batches,_ = self._pack_training_batches(
            random_order,
            data_source,
            batch_size,
            state_batch_size,
            state_count_fn,
            max_batches = random_count,
        )

        selected_batches = recent_batches + random_batches
        selected_items = set(self._flatten_batches(selected_batches))
        remainder_indices = [data_index for data_index in all_indices if data_index not in selected_items]
        self._last_training_sample_summary = self._training_sample_summary(
            original_batch_count = len(original_batches),
            original_item_count = len(all_indices),
            selected_batches = selected_batches,
            recent_batch_count = len(recent_batches),
            random_batch_count = len(random_batches),
            remainder_indices = remainder_indices,
            data_source = data_source,
            state_count_fn = state_count_fn,
        )
        return selected_batches,remainder_indices

    def _flatten_batches(self, batches):
        items = []
        for batch in batches:
            items += batch
        return items

    def _training_sample_summary(self, original_batch_count, original_item_count, selected_batches, recent_batch_count, random_batch_count, remainder_indices, data_source, state_count_fn):
        selected_items = self._flatten_batches(selected_batches)
        selected_states = sum(int(state_count_fn(data_source[data_index])) for data_index in selected_items)
        if selected_items:
            selected_index_min = min(selected_items)
            selected_index_max = max(selected_items)
        else:
            selected_index_min = -1
            selected_index_max = -1
        return {
            'original_batches': original_batch_count,
            'selected_batches': len(selected_batches),
            'recent_batches': recent_batch_count,
            'random_batches': random_batch_count,
            'original_items': original_item_count,
            'selected_items': len(selected_items),
            'selected_states': selected_states,
            'remainder_items': len(remainder_indices),
            'selected_index_min': selected_index_min,
            'selected_index_max': selected_index_max,
        }

    def _init_training_epoch_state(self, indices, epoch_num):
        """学習 epoch の集計用状態を初期化する。"""
        return {
            'indices': list(indices),
            'original_len': len(indices),
            'epoch_num': epoch_num,
            'err': 0.0,
            'err2': 0.0,
            'new_indices': [],
            'l0_max': 0.0,
            'l1_max': 0.0,
            'l1_indices': [],
            'epoch_index': 0,
            'progress_callback': None,
            'last_batch_item_count': 0,
            'search2_value_loss_components': None,
        }

    def _iter_training_batches(self, batches):
        """学習対象 index を batch 単位で順に返す。"""
        for epoch_index,batch_indices in enumerate(batches):
            yield epoch_index,batch_indices

    def _should_keep_training_batch(self, losses):
        """誤差が大きい batch を次 epoch に残すか判定する。"""
        return losses[0] > 0.01 or losses[1] > 0.0001

    def _finalize_training(self, progress_callback = None):
        """学習後の BN 更新、ログ出力、cache 破棄をまとめて行う。"""
        if self.Batch_Normalize:
            self._refresh_bn_stats()
        self._log_weight_stats(progress_callback = progress_callback)
        self.mark_params_dirty()
        self.clear_training_cache()

    def _train_batch_search3(self, d_lis, batch_indices, transformation, flip_inside, epoch_state):
        if self._can_use_torch_training():
            return self._train_batch_search3_torch(d_lis,batch_indices,transformation,flip_inside,epoch_state)

        L = self.loss_search3(d_lis,transformation = transformation,flip_inside = flip_inside)
        L0 = L[0] / len(d_lis)
        L1 = L[1] / len(d_lis)
        epoch_state = self._update_training_maxima(L0,L1,batch_indices,epoch_state)
        self._maybe_report_search3_value_debug(epoch_state,L1)

        dO = self._backprop_policy_search3()
        dO2 = self._backprop_value_search3()
        dO += dO2
        self._backprop_trunk(dO)
        self._update_output_momentum()
        self._update_affine_momentum()
        self._update_attention_momentum()
        self._update_bn_momentum()
        self._update_activation_momentum()
        self._apply_param_updates()
        self.clear_training_cache(collect = False)
        return (L0,L1), epoch_state

    def _train_batch(self, d_lis, batch_indices, transformation, flip_inside, epoch_state):
        if self._can_use_torch_training():
            return self._train_batch_torch(d_lis,batch_indices,transformation,flip_inside,epoch_state)

        L = self.loss(d_lis,transformation = transformation,flip_inside = flip_inside)
        L0 = L[0] / len(d_lis)
        L1 = L[1] / len(d_lis)
        epoch_state = self._update_training_maxima(L0,L1,batch_indices,epoch_state)
        epoch_state['last_batch_item_count'] = len(d_lis)
        self._accumulate_search2_value_loss_components(epoch_state,self._last_search2_value_loss_components,len(d_lis))

        dO = self._backprop_policy()
        dO2 = self._backprop_value()
        dO += dO2
        self._backprop_trunk(dO)
        self._update_output_momentum()
        self._update_affine_momentum()
        self._update_attention_momentum()
        self._update_bn_momentum()
        self._update_activation_momentum()
        self._apply_param_updates()
        self.clear_training_cache(collect = False)
        return (L0,L1), epoch_state

    def _can_use_torch_training(self):
        """Torch autograd training path is used when BatchNorm is not active."""
        use_torch_training = bool(getattr(self,'use_torch',False) or getattr(self,'use_torch_training',False))
        return use_torch_training and torch is not None and not self.Batch_Normalize

    def _train_batch_torch(self, d_lis, batch_indices, transformation, flip_inside, epoch_state):
        loss_sum = [0.0,0.0]
        component_sum = None
        grad_by_key = {}
        for micro_d_lis in self._iter_torch_training_microbatches(
            d_lis,
            lambda data_item:max(1,len(data_item.moves) + 1),
        ):
            loss_inputs = self._build_loss_inputs(
                micro_d_lis,
                transformation = transformation,
                flip_inside = flip_inside,
            )
            micro_losses,micro_grads = self._torch_search2_losses_and_grads(loss_inputs)
            loss_sum[0] += micro_losses[0]
            loss_sum[1] += micro_losses[1]
            component_sum = self._merged_search2_value_loss_components(
                component_sum,
                self._last_search2_value_loss_components,
                len(micro_d_lis),
            )
            self._accumulate_torch_grad_dict(grad_by_key,micro_grads)
            del loss_inputs, micro_grads
        L0 = loss_sum[0] / len(d_lis)
        L1 = loss_sum[1] / len(d_lis)
        if component_sum is not None:
            self._last_search2_value_loss_components = component_sum
        epoch_state = self._update_training_maxima(L0,L1,batch_indices,epoch_state)
        epoch_state['last_batch_item_count'] = len(d_lis)
        self._accumulate_search2_value_loss_components(epoch_state,self._last_search2_value_loss_components,len(d_lis))
        self._apply_torch_grad_updates(grad_by_key)
        del grad_by_key
        self.clear_training_cache(collect = False)
        self._release_torch_training_memory()
        return (L0,L1), epoch_state

    def _train_batch_search3_torch(self, d_lis, batch_indices, transformation, flip_inside, epoch_state):
        loss_sum = [0.0,0.0]
        grad_by_key = {}
        debug_inputs = None
        debug_out = None
        for micro_d_lis in self._iter_torch_training_microbatches(
            d_lis,
            lambda data_item:max(1,len(data_item.moves) + 1),
        ):
            search3_inputs = self._build_search3_loss_inputs(
                micro_d_lis,
                transformation = transformation,
                flip_inside = flip_inside,
            )
            micro_losses,micro_grads,out = self._torch_search3_losses_and_grads(search3_inputs)
            loss_sum[0] += micro_losses[0]
            loss_sum[1] += micro_losses[1]
            self._accumulate_torch_grad_dict(grad_by_key,micro_grads)
            debug_inputs = search3_inputs
            debug_out = out
            del micro_grads
        L0 = loss_sum[0] / len(d_lis)
        L1 = loss_sum[1] / len(d_lis)
        epoch_state = self._update_training_maxima(L0,L1,batch_indices,epoch_state)
        if debug_out is not None and debug_inputs is not None:
            self._last_search3_debug_summary = self._build_search3_debug_summary(debug_out, debug_inputs)
        self._maybe_report_search3_value_debug(epoch_state,L1)
        self._apply_torch_grad_updates(grad_by_key)
        del debug_inputs, debug_out, grad_by_key
        self.clear_training_cache(collect = False)
        self._release_torch_training_memory()
        return (L0,L1), epoch_state

    def _torch_trainable_params(self, device):
        params = {}
        for key,value in self.params.items():
            param = torch.as_tensor(value, dtype = torch.float32, device = device).detach().clone()
            if not self._is_non_trainable_param(key):
                param.requires_grad_(True)
            params[key] = param
        return params

    def _release_torch_training_memory(self):
        if torch is None:
            return
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    def torch_memory_status(self):
        if torch is None:
            return {}
        status = {}
        if torch.backends.mps.is_available():
            if hasattr(torch.mps,'current_allocated_memory'):
                status['mps_current_mb'] = float(torch.mps.current_allocated_memory()) / (1024.0 * 1024.0)
            if hasattr(torch.mps,'driver_allocated_memory'):
                status['mps_driver_mb'] = float(torch.mps.driver_allocated_memory()) / (1024.0 * 1024.0)
        if torch.cuda.is_available():
            status['cuda_allocated_mb'] = float(torch.cuda.memory_allocated()) / (1024.0 * 1024.0)
            status['cuda_reserved_mb'] = float(torch.cuda.memory_reserved()) / (1024.0 * 1024.0)
        return status

    def _iter_torch_training_microbatches(self, d_lis, state_count_fn):
        """Yield data chunks that keep the Torch autograd graph small."""
        state_limit = max(0,int(getattr(self,'train_state_batch_size',0)))
        if not self._uses_torch_microbatch_training() or state_limit <= 0:
            yield d_lis
            return

        current = []
        current_states = 0
        for data_item in d_lis:
            item_states = int(state_count_fn(data_item))
            if current and current_states + item_states > state_limit:
                yield current
                current = []
                current_states = 0
            current.append(data_item)
            current_states += item_states
        if current:
            yield current

    def _accumulate_torch_grad_dict(self, grad_accumulator, grad_by_key):
        for key,grad in grad_by_key.items():
            if key not in grad_accumulator:
                grad_accumulator[key] = grad.copy()
            else:
                grad_accumulator[key] += grad

    def _torch_search2_losses_and_grads(self, loss_inputs):
        device = self._torch_training_device()
        params = self._torch_trainable_params(device)
        x = torch.as_tensor(loss_inputs['x'], dtype = torch.float32, device = device)
        out = self._torch_forward_policy_value(x, params)
        policy_loss = self._torch_search2_policy_loss(out[:-1],loss_inputs['args'],loss_inputs['indices'],device)
        value_loss = self._torch_search2_value_loss(
            out[-1],
            loss_inputs['value_indices'],
            loss_inputs['value_columns'],
            loss_inputs['value_steps_to_goal'],
        )
        total_loss = policy_loss + value_loss
        total_loss.backward()
        return (
            (float(policy_loss.detach().cpu().item()),float(value_loss.detach().cpu().item())),
            self._torch_grad_dict(params),
        )

    def _torch_search2_policy_loss(self, logits, args, indices, device):
        if len(args) == 0:
            return torch.zeros((), dtype = torch.float32, device = device)
        total_loss = torch.zeros((), dtype = logits.dtype, device = device)
        has_loss = False
        target_start = 0
        target_tensor = torch.as_tensor(args, dtype = torch.long, device = device)
        for index in range(len(indices) - 1):
            start = indices[index]
            end = indices[index + 1] - 1
            if end <= start:
                continue
            count = end - start
            total_loss = total_loss + F_torch.cross_entropy(
                logits[:,start:end].transpose(0,1),
                target_tensor[target_start:target_start + count],
                reduction = 'sum',
            )
            has_loss = True
            target_start += count
        if not has_loss:
            return torch.zeros((), dtype = torch.float32, device = device)
        return total_loss

    def _torch_search2_value_loss(self, values, indices, value_columns = None, value_steps_to_goal = None):
        if value_columns is not None:
            if len(value_columns) == 0:
                self._record_search2_value_loss_components('none',0.0,0.0,0.0,False)
                return torch.zeros((), dtype = values.dtype, device = values.device)
            column_tensor = torch.as_tensor(value_columns, dtype = torch.long, device = values.device)
            values = values.index_select(0,column_tensor)
        if self.search2_value_loss_type == 'myloss':
            value_loss = self._torch_search2_value_loss_softmax(values,indices)
            self._record_search2_value_loss_components(
                'myloss',
                self._torch_loss_float(value_loss),
                0.0,
                0.0,
                False,
            )
            return value_loss
        if self.search2_value_loss_type == 'myloss2_pairwise':
            value_loss = self._torch_search2_value_loss_pairwise(values,indices)
            base_name = 'myloss2_pairwise'
        else:
            value_loss = self._torch_search2_value_loss_distance(values,indices,value_steps_to_goal)
            base_name = 'myloss2_distance'
        base_loss = value_loss
        rank_loss = torch.zeros((), dtype = values.dtype, device = values.device)
        rank_scaled_loss = torch.zeros((), dtype = values.dtype, device = values.device)
        rank_applied = False
        if self.search2_rank_loss_mix > 0.0 and self._uses_search2_rank_mix():
            rank_loss = self._torch_search2_value_loss_softmax(values,indices)
            rank_scaled_loss = self.search2_rank_loss_mix * rank_loss
            value_loss = value_loss + rank_scaled_loss
            rank_applied = True
        self._record_search2_value_loss_components(
            base_name,
            self._torch_loss_float(base_loss),
            self._torch_loss_float(rank_loss),
            self._torch_loss_float(rank_scaled_loss),
            rank_applied,
        )
        return value_loss

    def _torch_loss_float(self, loss):
        return float(loss.detach().cpu().item())

    def _torch_search2_value_loss_softmax(self, values, indices):
        total_loss = torch.zeros((), dtype = values.dtype, device = values.device)
        has_loss = False
        for index in range(len(indices) - 1):
            start = indices[index]
            end = indices[index + 1]
            if end <= start:
                continue
            total_loss = total_loss - F_torch.log_softmax(values[start:end],dim = 0)[-1]
            has_loss = True
        if not has_loss:
            return torch.zeros((), dtype = values.dtype, device = values.device)
        return total_loss

    def _torch_search2_value_loss_distance(self, values, indices, value_steps_to_goal = None):
        if values.numel() == 0:
            return torch.zeros((), dtype = values.dtype, device = values.device)
        if value_steps_to_goal is None:
            value_steps_to_goal = torch.zeros_like(values)
        else:
            value_steps_to_goal = torch.as_tensor(value_steps_to_goal, dtype = values.dtype, device = values.device)
        gamma = torch.as_tensor(self.value_target_gamma, dtype = values.dtype, device = values.device)
        total_loss = torch.zeros((), dtype = values.dtype, device = values.device)
        has_loss = False
        for index in range(len(indices) - 1):
            start = indices[index]
            end = indices[index + 1]
            if end <= start:
                continue
            targets = torch.pow(gamma,value_steps_to_goal[start:end])
            targets = targets / torch.sum(targets)
            total_loss = total_loss - torch.sum(targets * F_torch.log_softmax(values[start:end],dim = 0))
            has_loss = True
        if not has_loss:
            return torch.zeros((), dtype = values.dtype, device = values.device)
        return total_loss

    def _torch_search2_value_loss_pairwise(self, values, indices):
        total_loss = torch.zeros((), dtype = values.dtype, device = values.device)
        has_loss = False
        margin = torch.as_tensor(self.search2_value_loss_margin, dtype = values.dtype, device = values.device)
        for index in range(len(indices) - 1):
            start = indices[index]
            end = indices[index + 1]
            if end - start <= 1:
                continue
            diffs = values[start + 1:end] - values[start:end - 1]
            total_loss = total_loss + torch.sum(F_torch.softplus(margin - diffs))
            has_loss = True
        if not has_loss:
            return torch.zeros((), dtype = values.dtype, device = values.device)
        return total_loss

    def _torch_search3_losses_and_grads(self, search3_inputs):
        device = self._torch_training_device()
        params = self._torch_trainable_params(device)
        x = torch.as_tensor(search3_inputs['x'], dtype = torch.float32, device = device)
        policy_targets = torch.as_tensor(search3_inputs['policy_targets'], dtype = torch.float32, device = device)
        value_targets = torch.as_tensor(search3_inputs['value_targets'], dtype = torch.float32, device = device)
        sample_weights = torch.as_tensor(search3_inputs['sample_weights'], dtype = torch.float32, device = device)
        policy_weights = torch.as_tensor(search3_inputs['policy_weights'], dtype = torch.float32, device = device)
        out = self._torch_forward_policy_value(x, params)
        policy_log_probs = F_torch.log_softmax(out[:-1],dim = 0)
        policy_loss = -torch.sum(policy_weights * policy_targets * policy_log_probs)
        value_loss = F_torch.binary_cross_entropy_with_logits(
            out[-1:],
            value_targets,
            weight = sample_weights,
            reduction = 'sum',
        )
        if self.search3_rank_loss_mix > 0.0 and len(search3_inputs['value_indices']) > 1:
            value_loss = value_loss + self.search3_rank_loss_mix * self._torch_search2_value_loss_softmax(
                out[-1],
                search3_inputs['value_indices'],
            )
        total_loss = policy_loss + value_loss
        total_loss.backward()
        return (
            (float(policy_loss.detach().cpu().item()),float(value_loss.detach().cpu().item())),
            self._torch_grad_dict(params),
            out.detach().cpu().numpy(),
        )

    def _torch_grad_dict(self, params):
        grad_by_key = {}
        for key,param in params.items():
            if self._is_non_trainable_param(key) or param.grad is None:
                continue
            grad_by_key[key] = param.grad.detach().cpu().numpy().astype('f')
        return grad_by_key

    def _apply_torch_grad_updates(self, grad_by_key):
        if self.weight_decay:
            for key in self._weight_decay_keys():
                if key in self.params:
                    self._apply_weight_decay(key)
        for key,grad in grad_by_key.items():
            if key in self.v and not self._is_non_trainable_param(key):
                self._accumulate_weight_optimizer_state(key,grad)
        self._apply_param_updates()

    def _weight_decay_keys(self):
        keys = ['WO_P','WO_V','WM_P']
        for key in self.affines:
            keys.append('W' + key[-1])
        for key in self.attentions:
            suffix = key[3:]
            keys += ['WQ' + suffix,'WK' + suffix,'WV' + suffix]
        return keys

    def _is_non_trainable_param(self, key):
        if key == 'BO_V':
            return not self._is_value_output_bias_trainable()
        return key.startswith('BNm') or key.startswith('BNs')

    def _is_value_output_bias_trainable(self):
        """Search3 value BCE can need a global logit shift; keep Search2 behavior unchanged."""
        return getattr(self,'search_mode',None) == 'search3'

    def _update_training_maxima(self, l0, l1, batch_indices, epoch_state):
        """最大損失 batch の記録を更新する。"""
        if l0 > epoch_state['l0_max']:
            epoch_state['l0_max'] = l0
        if l1 > epoch_state['l1_max']:
            epoch_state['l1_max'] = l1
            epoch_state['l1_indices'] = batch_indices.copy()
        return epoch_state

    def _backprop_policy(self):
        dO = self.losslayer.backward()
        dO = self.policy_layer.backward(dO)
        dO = self.policy_act.backward(dO)
        if self.Batch_Normalize:
            dO = self.policy_BN.backward(dO)
        dO = self.policy_mid.backward(dO)
        return dO

    def _backprop_value(self):
        dO2 = self._search2_value_loss_backward()
        dO2 = self.value_layer.backward(dO2)
        dO2 = self.value_act.backward(dO2)
        if self.Batch_Normalize:
            dO2 = self.value_BN.backward(dO2)
        dO2 = self.value_mid.backward(dO2)
        return dO2

    def _search2_value_loss_backward(self):
        output_size = int(getattr(self,'_search2_value_output_size',0) or 0)
        columns = getattr(self,'_search2_value_columns',None)
        if columns is None:
            return self.losslayer2.backward()
        d_selected = self.losslayer2.backward() if len(columns) > 0 else np.zeros((1,0),dtype = 'f')
        if (
            len(columns) > 0
            and self._uses_search2_rank_mix()
            and self._search2_value_has_rank_loss
        ):
            d_selected = d_selected + self.search2_rank_loss_mix * self.search2_rank_loss.backward()
        d_full = np.zeros((1,output_size),dtype = d_selected.dtype if d_selected.size > 0 else 'f')
        if len(columns) > 0:
            d_full[:,columns] = d_selected
        return d_full

    def _backprop_policy_search3(self):
        dO = self.losslayer4.backward()
        dO = self.policy_layer.backward(dO)
        dO = self.policy_act.backward(dO)
        if self.Batch_Normalize:
            dO = self.policy_BN.backward(dO)
        dO = self.policy_mid.backward(dO)
        return dO

    def _backprop_value_search3(self):
        dO2 = self.losslayer3.backward()
        if self._search3_value_has_rank_loss:
            dO2 = dO2 + self.search3_rank_loss_mix * self.search3_rank_loss.backward()
        dO2 = self.value_layer.backward(dO2)
        dO2 = self.value_act.backward(dO2)
        if self.Batch_Normalize:
            dO2 = self.value_BN.backward(dO2)
        dO2 = self.value_mid.backward(dO2)
        return dO2

    def _backprop_trunk(self, dO):
        if self.residual and self.use_transformer_attention:
            self._backprop_trunk_with_attention_residual(dO)
            return
        if self.residual:
            for block in reversed(self.trunk_blocks):
                dO = block.backward(dO)
            return

        reverse_key = list(self.layers)
        reverse_key.reverse()
        for key in reverse_key:
            dO = self.layers[key].backward(dO)

    def _backprop_trunk_with_attention_residual(self, dO):
        """attention を含む residual trunk を逆伝播する。"""
        for index in reversed(range(1,len(self.Mid) + 1)):
            skip_grad = dO.copy() if getattr(self,'_residual_skip_flags',{}).get(index,False) else 0
            dO = self.layers['Act' + str(index)].backward(dO)
            bn_key = 'BN' + str(index)
            if self.Batch_Normalize and bn_key in self.layers:
                dO = self.layers[bn_key].backward(dO)
            attention_key = 'Att' + str(index)
            if attention_key in self.layers:
                dO = self.layers[attention_key].backward(dO)
            dO = self.layers['Aff' + str(index)].backward(dO)
            dO = dO + skip_grad

    def _update_output_momentum(self):
        self._accumulate_affine_optimizer_state(self.policy_layer,'WO_P','BO_P')
        self._accumulate_affine_optimizer_state(
            self.value_layer,
            'WO_V',
            'BO_V',
            update_bias = self._is_value_output_bias_trainable(),
        )
        self._accumulate_affine_optimizer_state(self.policy_mid,'WM_P','BM_P')
        self._accumulate_affine_optimizer_state(self.value_mid,'WM_V','BM_V')

        if self.Batch_Normalize:
            self._accumulate_bn_optimizer_state(self.policy_BN,'BNgP','BNbP')
            self._accumulate_bn_optimizer_state(self.value_BN,'BNgV','BNbV')

    def _update_affine_momentum(self):
        #if self.weight_decay:
        #    self._apply_weight_decay('WO_P')
        #    self._apply_weight_decay('WO_V')
        #    self._apply_weight_decay('WM_P')

        for key in self.affines:
            if self.weight_decay:
                self._apply_weight_decay('W' + key[-1])
            self._accumulate_affine_optimizer_state(
                self.layers[key],
                'W' + key[-1],
                'B' + key[-1],
            )

    def _update_attention_momentum(self):
        """Transformer_SelfAttention の Q/K/V optimizer 状態を更新する。"""
        for key in self.attentions:
            suffix = key[3:]
            if self.weight_decay:
                self._apply_weight_decay('WQ' + suffix)
                self._apply_weight_decay('WK' + suffix)
                self._apply_weight_decay('WV' + suffix)
            layer = self.layers[key]
            self._accumulate_weight_optimizer_state('WQ' + suffix,layer.dWQ)
            self._accumulate_weight_optimizer_state('WK' + suffix,layer.dWK)
            self._accumulate_weight_optimizer_state('WV' + suffix,layer.dWV)

    def _update_bn_momentum(self):
        for key in self.BNs:
            self._accumulate_bn_optimizer_state(
                self.layers[key],
                'BNg' + key[-1],
                'BNb' + key[-1],
            )

    def _update_activation_momentum(self):
        for key,layer in self.activation_betas:
            self._accumulate_weight_optimizer_state(key,layer.dbeta)

    def _apply_param_updates(self):
        for key in self.v.keys():
            self.params[key] -= self._param_update_scale(key) * self._optimizer_step(key)
            self.v[key] *= self.lr_v
            self.h[key] *= self.lr_h
        self._mark_torch_params_dirty()

    def _param_update_scale(self, key):
        """Return the configured post-optimizer scale for this parameter group."""
        if key.endswith('P'):
            return self.update_scale_policy
        if key.endswith('V'):
            return self.update_scale_value
        return self.update_scale_shared

    def _accumulate_affine_optimizer_state(self, layer, weight_key, bias_key, update_bias = True):
        """Affine layer の勾配を optimizer 状態へ加算する。"""
        self._accumulate_weight_optimizer_state(weight_key,layer.dW)
        if update_bias:
            self._accumulate_weight_optimizer_state(bias_key,layer.dB)

    def _accumulate_bn_optimizer_state(self, layer, gamma_key, beta_key):
        """BatchNorm layer の勾配を optimizer 状態へ加算する。"""
        self._accumulate_weight_optimizer_state(gamma_key,layer.dg)
        self._accumulate_weight_optimizer_state(beta_key,layer.db)

    def _accumulate_weight_optimizer_state(self, key, grad):
        """1 パラメータ分の一次/二次モーメントを更新する。"""
        self.v[key] += self.lr * grad
        self.h[key] += grad ** 2

    def _apply_weight_decay(self, key):
        """対象 weight に weight decay を適用する。"""
        self.params[key] -= self.wdlr * self.params[key]

    def _optimizer_step(self, key):
        """現在の optimizer 設定に応じた 1 step 分の更新量を返す。"""
        if self.adam:
            if key[:2] in ['WQ','WK']:
                return self.v[key] / (np.sqrt(self.h[key]) + 1.0e-8)
            else:
                return self.v[key] / (np.sqrt(self.h[key]) + 1)
        return self.v[key]

    def _refresh_bn_stats(self):
        for key in self.BNs:
            self.layers[key].set_ms()
        self.policy_BN.set_ms()
        self.value_BN.set_ms()

    def _log_weight_stats(self, progress_callback = None):
        for key in self.v.keys():
            if key[0] == "W":
                if self.adam:
                    message = (
                        key + ":" + str(np.log10(np.average(self.v[key] ** 2 / (self.h[key] + 1))))
                        + ", " + str(np.max(self.h[key]))
                    )
                else:
                    message = (
                        key + ":" + str(np.log10(np.average(self.v[key] ** 2)))
                        + ", " + str(np.max(self.h[key]))
                    )
                if progress_callback is not None:
                    progress_callback(message)
        

    def search2(self,N):
        return self.search2_engine.search2(N)
    
    def search3(self,N):
        return self.search3_engine.search3(N,C = self.search3_C)

    def search(self, progress_callback = None):
        if self.search_mode == 'search2':
            result = self.search2(self.search_num2)
            if progress_callback is not None:
                progress_callback(result)
            return result
        elif self.search_mode == 'search3':
            return self._search3_with_repeats(progress_callback = progress_callback)
        else:
            raise ValueError(self.search_mode)

    def _search3_with_repeats(self, progress_callback = None):
        last_result = None
        attempt_results = []
        for attempt_index in range(self.search_repeat3):
            last_result = self.search3(self.search_num3)
            last_result.attempt_index = attempt_index + 1
            attempt_results.append(last_result)
            if progress_callback is not None:
                progress_callback(last_result)
            if last_result.end_reason == 'solved':
                last_result.attempt_results = attempt_results
                return last_result
        if last_result is not None:
            last_result.attempt_results = attempt_results
        return last_result

    def _predict_search2(self, X):
        if not self._can_use_torch_predict(loss = False, retain_cache = False):
            return self.predict(X,policy = True,value = True)
        return self._torch_predict_batched(X,self.search2_torch_batch_size)

    def _torch_predict(self, X):
        return self._torch_predict_batched(X,0)

    def _torch_predict_batched(self, X, max_batch_size = 0):
        max_batch_size = int(max_batch_size or 0)
        if max_batch_size <= 0 or X.shape[1] <= max_batch_size:
            return self._torch_predict_direct(X)
        outputs = []
        for start in range(0,X.shape[1],max_batch_size):
            end = min(start + max_batch_size,X.shape[1])
            outputs.append(self._torch_predict_direct(X[:,start:end]))
        return np.concatenate(outputs,axis = 1)

    def _torch_predict_direct(self, X):
        device = self._torch_device()
        params = self._torch_params_from_numpy(device)
        with torch.no_grad():
            X_t = torch.as_tensor(X, dtype = torch.float32, device = device)
            out = self._torch_forward_policy_value(X_t, params)
        return out.cpu().numpy()

    def _torch_device(self):
        if hasattr(self, "_torch_device_cache"):
            return self._torch_device_cache
        device_name = "mps" if torch.backends.mps.is_available() else "cpu"
        self._torch_device_cache = torch.device(device_name)
        return self._torch_device_cache

    def _torch_training_device(self):
        requested_device = str(getattr(self,'torch_training_device','auto')).lower()
        if requested_device in ('auto','infer','inference'):
            return self._torch_device()
        if requested_device == 'mps' and torch.backends.mps.is_available():
            return torch.device('mps')
        if requested_device == 'cuda' and torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')

    def _torch_params_from_numpy(self, device):
        if self._torch_params_cache is not None and not self._torch_params_dirty and self._torch_params_device == device:
            return self._torch_params_cache
        params = {
            key: torch.as_tensor(value, dtype = torch.float32, device = device)
            for key, value in self.params.items()
        }
        self._torch_params_cache = params
        self._torch_params_device = device
        self._torch_params_dirty = False
        return params

    def _torch_forward_policy_value(self, X, params):
        out = X
        for i in range(1, len(self.Mid) + 1):
            residual = out
            W = params['W' + str(i)]
            B = params['B' + str(i)]
            if i == 1 and getattr(self,'use_piece_tokens',False) and 'WQ1' in params:
                out = self._torch_piece_token_attention(X, params, suffix = '1')
            else:
                out = W @ out + B.unsqueeze(1)
            if i == 1 and 'WQ1' in params and not getattr(self,'use_piece_tokens',False):
                out = self._torch_self_attention(out, params, suffix = '1')
            out = self._torch_activation(out, params = params, suffix = str(i))
            if self.residual and out.shape == residual.shape:
                out = out + residual

        p = self._torch_forward_head(out, params, mid_key = 'WM_P', mid_bias_key = 'BM_P', out_key = 'WO_P', out_bias_key = 'BO_P')
        v = self._torch_forward_head(out, params, mid_key = 'WM_V', mid_bias_key = 'BM_V', out_key = 'WO_V', out_bias_key = 'BO_V')

        return torch.cat([p, v], dim = 0)

    def _torch_self_attention(self, out, params, suffix = '1'):
        """Torch 推論用の単層 self-attention。"""
        q = params['WQ' + suffix] @ out
        k = params['WK' + suffix] @ out
        v = out
        scale = 1.0 / math.sqrt(max(q.shape[0],1))
        score = scale * torch.einsum('ib,jb->bij',q,k)
        attention = torch.softmax(score,dim = -1)
        attended = torch.einsum('bij,jb->ib',attention,v)
        return params['WV' + suffix] @ attended

    def _torch_piece_token_attention(self, X, params, suffix = '1'):
        tokens = []
        W = params['W' + suffix]
        for index_tensor in self._torch_piece_feature_index_tensors(suffix,X.device):
            tokens.append(W.index_select(1,index_tensor) @ X.index_select(0,index_tensor))
        token_tensor = torch.stack(tokens, dim = 0)
        q = torch.einsum('od,pdb->pob',params['WQ' + suffix],token_tensor)
        k = torch.einsum('od,pdb->pob',params['WK' + suffix],token_tensor)
        v = token_tensor
        scale = 1.0 / math.sqrt(max(q.shape[1],1))
        score = scale * torch.einsum('pdb,qdb->bpq',q,k)
        attention = torch.softmax(score,dim = -1)
        attended = torch.einsum('bpq,qdb->pdb',attention,v)
        return params['WV' + suffix] @ torch.sum(attended,dim = 0) + params['B' + suffix].unsqueeze(1)

    def _torch_piece_feature_index_tensors(self, suffix, device):
        cache_key = (suffix,str(device))
        if not hasattr(self,'_torch_piece_index_cache'):
            self._torch_piece_index_cache = {}
        if cache_key not in self._torch_piece_index_cache:
            self._torch_piece_index_cache[cache_key] = [
                torch.as_tensor(feature_indices, dtype = torch.long, device = device)
                for feature_indices in self.layers['Aff' + suffix].piece_feature_indices
            ]
        return self._torch_piece_index_cache[cache_key]

    def _torch_forward_head(self, trunk_out, params, mid_key, mid_bias_key, out_key, out_bias_key):
        """policy/value head を 1 本分だけ順伝播する。"""
        head_out = params[mid_key] @ trunk_out + params[mid_bias_key].unsqueeze(1)
        head_out = self._torch_head_act(head_out, params, out_key[-1])
        return params[out_key] @ head_out + params[out_bias_key].unsqueeze(1)

    def _torch_head_act(self, x, params, suffix):
        return self._torch_activation(x, params = params, suffix = suffix)

    def _torch_activation(self, x, params = None, suffix = None):
        activation = self._normalized_activation_name()
        if activation == 'silu':
            beta = None
            if params is not None and suffix is not None:
                beta = params.get('ActBeta' + str(suffix))
            if beta is None:
                beta = torch.as_tensor([1.0], dtype = x.dtype, device = x.device)
            return x * torch.sigmoid(beta.reshape(1,1) * x)
        if activation == 'relu':
            return F_torch.relu(x)
        elif activation == 'hard_sigmoid':
            return F_torch.hardsigmoid(x)
        else:
            return torch.sigmoid(x)
    def _append_trunk_block(self, index):
        if 'Att' + str(index) in self.layers:
            return
        bn = self.layers.get('BN' + str(index))
        self.trunk_blocks.append(
            ResidualBlock(
                self.layers['Aff' + str(index)],
                self.layers['Act' + str(index)],
                bn,
            )
        )
