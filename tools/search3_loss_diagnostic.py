"""Diagnose Search3 policy/value loss geometry on generated successful samples."""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from dataclasses import dataclass

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0,ROOT_DIR)

from ai.rubiks_ai import Rubiks_3_AI
from ai.losses import BCEWithLogits, Myloss, Soft_Target_Cross_Entropy, softmax
from model.search_result import data_search3


@dataclass
class Search3Diagnostic:
    name: str
    sample_count: int
    step_count: int
    target_min: float
    target_mean: float
    target_max: float
    target_std: float
    logit_min: float
    logit_mean: float
    logit_max: float
    pred_min: float
    pred_mean: float
    pred_max: float
    pred_std: float
    value_bce: float
    rank_raw: float
    rank_scaled: float
    value_total: float
    policy_loss: float
    value_grad_norm: float
    rank_grad_norm: float
    scaled_rank_grad_norm: float
    combined_value_grad_norm: float
    value_rank_cosine: float
    value_rank_sign_conflict: float
    value_grad_mean: float
    value_grad_sum: float
    all_raise_fraction: float
    policy_grad_norm: float
    policy_to_value_grad_ratio: float
    target_contrast_mean: float
    target_contrast_max: float
    pred_contrast_mean: float
    pred_contrast_max: float
    final_minus_start_target_mean: float
    final_minus_start_pred_mean: float
    bo_v_trainable: bool


def parse_args():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('--cube-size', type = int, default = 3)
    parser.add_argument('--samples', type = int, default = 64)
    parser.add_argument('--min-len', type = int, default = 2)
    parser.add_argument('--max-len', type = int, default = 8)
    parser.add_argument('--mid', type = int, default = 64)
    parser.add_argument('--layers', type = int, default = 2)
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--rank-mix', type = float, default = 0.25)
    parser.add_argument('--target-mode', choices = ('gamma','linear','binary-final'), default = 'gamma')
    parser.add_argument('--single', action = 'store_true')
    return parser.parse_args()


def build_ai(args, rank_mix):
    return Rubiks_3_AI(
        [int(args.mid)] * int(args.layers),
        cube_size = args.cube_size,
        search_mode = 'search3',
        search3_rank_loss_mix = rank_mix,
    )


def generated_search3_data(cube, sample_count, min_len, max_len, rng, gamma, target_mode):
    move_pool = face_move_pool(cube)
    samples = []
    for _ in range(sample_count):
        length = rng.randint(min_len,max_len)
        scramble = tuple(rng.choice(move_pool) for _ in range(length))
        moves = tuple(cube.invert_moves(scramble))
        value_targets = search3_value_targets(length,gamma,target_mode)
        policy_target = one_hot_policy(cube,moves[0]) if moves else None
        value_trace = value_targets.tolist()
        samples.append(data_search3(
            scramble,
            moves,
            value_targets.copy(),
            value_trace[0],
            value_trace,
            value_trace[-1],
            {'source': 'diagnostic'},
            policy_target = policy_target,
            search_mode = 'search3',
            sample_weight = 1.0,
            value_targets = value_targets,
            root_value_raw = value_trace[0],
            value_trace_raw = value_trace,
            best_value_raw = value_trace[-1],
            perfect_key = 'diagnostic',
            top_group = None,
            end_reason = 'diagnostic',
            source_succeeded = True,
            solve_succeeded = True,
        ))
    return samples


def face_move_pool(cube):
    faces = set('URFDLB')
    moves = [
        move
        for move in cube.move_keys
        if len(move) >= 3 and move[0] == ' ' and move[1] in faces
    ]
    return moves or list(cube.move_keys)


def search3_value_targets(length, gamma, mode):
    if mode == 'binary-final':
        targets = np.zeros(length + 1,dtype = 'f')
        targets[-1] = 1.0
        return targets
    if mode == 'linear':
        if length <= 1:
            return np.ones(length + 1,dtype = 'f')
        return np.linspace(0.1,1.0,length + 1,dtype = 'f')
    return np.asarray([gamma ** max(length - i,0) for i in range(length + 1)],dtype = 'f')


def one_hot_policy(cube, move):
    target = np.zeros((cube.move_len,),dtype = 'f')
    target[cube.key_to_num[move]] = 1.0
    return target


def run_diagnostic(args, rank_mix, target_mode, name):
    np.random.seed(args.seed)
    random.seed(args.seed)
    rng = random.Random(args.seed)
    ai = build_ai(args,rank_mix)
    samples = generated_search3_data(
        ai.cube,
        args.samples,
        args.min_len,
        args.max_len,
        rng,
        ai.value_target_gamma,
        target_mode,
    )
    inputs = ai._build_search3_loss_inputs(samples)
    out = ai._predict_loss_outputs(inputs['x'])
    policy_targets = inputs['policy_targets']
    value_targets = inputs['value_targets']
    weights = inputs['sample_weights']
    policy_weights = inputs['policy_weights']
    value_indices = inputs['value_indices']
    policy_loss_layer = Soft_Target_Cross_Entropy()
    policy_loss = policy_loss_layer.forward(out[:-1],policy_targets,policy_weights)
    policy_grad = policy_loss_layer.backward()
    bce = BCEWithLogits()
    value_bce = bce.forward(out[-1:],value_targets,weights)
    value_grad = bce.backward()
    rank_raw = 0.0
    rank_grad = np.zeros_like(value_grad)
    if rank_mix > 0.0 and len(value_indices) > 1:
        rank = Myloss()
        rank_raw = rank.forward(out[-1:],value_indices)
        rank_grad = rank.backward()
    scaled_rank_grad = rank_mix * rank_grad
    combined_value_grad = value_grad + scaled_rank_grad
    predictions = sigmoid(out[-1:])
    target_contrasts = sequence_contrasts(value_targets,value_indices)
    pred_contrasts = sequence_contrasts(predictions,value_indices)
    return Search3Diagnostic(
        name = name,
        sample_count = len(samples),
        step_count = int(value_targets.size),
        target_min = safe_min(value_targets),
        target_mean = safe_mean(value_targets),
        target_max = safe_max(value_targets),
        target_std = safe_std(value_targets),
        logit_min = safe_min(out[-1:]),
        logit_mean = safe_mean(out[-1:]),
        logit_max = safe_max(out[-1:]),
        pred_min = safe_min(predictions),
        pred_mean = safe_mean(predictions),
        pred_max = safe_max(predictions),
        pred_std = safe_std(predictions),
        value_bce = float(value_bce),
        rank_raw = float(rank_raw),
        rank_scaled = float(rank_mix * rank_raw),
        value_total = float(value_bce + rank_mix * rank_raw),
        policy_loss = float(policy_loss),
        value_grad_norm = norm(value_grad),
        rank_grad_norm = norm(rank_grad),
        scaled_rank_grad_norm = norm(scaled_rank_grad),
        combined_value_grad_norm = norm(combined_value_grad),
        value_rank_cosine = cosine(value_grad,scaled_rank_grad),
        value_rank_sign_conflict = sign_conflict_rate(value_grad,scaled_rank_grad),
        value_grad_mean = safe_mean(value_grad),
        value_grad_sum = float(np.sum(value_grad)),
        all_raise_fraction = float(np.mean(value_grad < 0.0)) if value_grad.size else 0.0,
        policy_grad_norm = norm(policy_grad),
        policy_to_value_grad_ratio = norm(policy_grad) / max(norm(combined_value_grad),1.0e-12),
        target_contrast_mean = safe_mean(target_contrasts),
        target_contrast_max = safe_max(target_contrasts),
        pred_contrast_mean = safe_mean(pred_contrasts),
        pred_contrast_max = safe_max(pred_contrasts),
        final_minus_start_target_mean = final_minus_start_mean(value_targets,value_indices),
        final_minus_start_pred_mean = final_minus_start_mean(predictions,value_indices),
        bo_v_trainable = not ai._is_non_trainable_param('BO_V'),
    )


def sequence_contrasts(values, indices):
    row = np.asarray(values).reshape(-1)
    contrasts = []
    for start,end in zip(indices[:-1],indices[1:]):
        if end > start:
            seq = row[start:end]
            contrasts.append(float(np.max(seq) - np.min(seq)))
    return np.asarray(contrasts,dtype = 'f')


def final_minus_start_mean(values, indices):
    row = np.asarray(values).reshape(-1)
    diffs = []
    for start,end in zip(indices[:-1],indices[1:]):
        if end > start:
            diffs.append(float(row[end - 1] - row[start]))
    return safe_mean(np.asarray(diffs,dtype = 'f'))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x,-60.0,60.0)))


def norm(x):
    return float(np.linalg.norm(np.asarray(x).reshape(-1)))


def cosine(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1.0e-12:
        return 0.0
    return float(np.dot(a,b) / denom)


def sign_conflict_rate(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    active = (np.abs(a) > 1.0e-9) & (np.abs(b) > 1.0e-9)
    if not np.any(active):
        return 0.0
    return float(np.mean(np.sign(a[active]) != np.sign(b[active])))


def safe_mean(x):
    x = np.asarray(x)
    return float(np.mean(x)) if x.size else 0.0


def safe_std(x):
    x = np.asarray(x)
    return float(np.std(x)) if x.size else 0.0


def safe_min(x):
    x = np.asarray(x)
    return float(np.min(x)) if x.size else 0.0


def safe_max(x):
    x = np.asarray(x)
    return float(np.max(x)) if x.size else 0.0


def print_report(diagnostics):
    print('Search3 loss diagnostic')
    print()
    for d in diagnostics:
        denom = max(1,d.step_count)
        print(f'[{d.name}] samples={d.sample_count} steps={d.step_count} BO_V_trainable={d.bo_v_trainable}')
        print(
            f'  target: min/mean/max/std={d.target_min:.4f}/{d.target_mean:.4f}/{d.target_max:.4f}/{d.target_std:.4f} '
            f'contrast_mean/max={d.target_contrast_mean:.4f}/{d.target_contrast_max:.4f} '
            f'final-start_mean={d.final_minus_start_target_mean:.4f}'
        )
        print(
            f'  pred: logit min/mean/max={d.logit_min:.4f}/{d.logit_mean:.4f}/{d.logit_max:.4f} '
            f'prob min/mean/max/std={d.pred_min:.4f}/{d.pred_mean:.4f}/{d.pred_max:.4f}/{d.pred_std:.4f} '
            f'contrast_mean/max={d.pred_contrast_mean:.4f}/{d.pred_contrast_max:.4f} '
            f'final-start_mean={d.final_minus_start_pred_mean:.4f}'
        )
        print(
            f'  loss/step: policy={d.policy_loss / denom:.6f} '
            f'value_bce={d.value_bce / denom:.6f} rank_raw={d.rank_raw / denom:.6f} '
            f'rank_scaled={d.rank_scaled / denom:.6f} value_total={d.value_total / denom:.6f}'
        )
        print(
            f'  output_grad: policy_norm={d.policy_grad_norm:.6f} '
            f'value_bce_norm={d.value_grad_norm:.6f} rank_norm={d.rank_grad_norm:.6f} '
            f'scaled_rank_norm={d.scaled_rank_grad_norm:.6f} combined_value_norm={d.combined_value_grad_norm:.6f} '
            f'policy/value_ratio={d.policy_to_value_grad_ratio:.3f}'
        )
        print(
            f'  value_grad: mean={d.value_grad_mean:.6f} sum={d.value_grad_sum:.6f} '
            f'raise_fraction={100*d.all_raise_fraction:.1f}% '
            f'cosine(bce,rank)={d.value_rank_cosine:.4f} '
            f'sign_conflict={100*d.value_rank_sign_conflict:.1f}%'
        )
        print_diagnosis(d)
        print()


def print_diagnosis(d):
    notes = []
    if d.target_std < 0.05 and d.target_contrast_mean < 0.1:
        notes.append('value targets have little contrast; BCE mainly learns a uniform offset.')
    if d.all_raise_fraction > 0.9:
        notes.append('BCE gradient mostly raises every value logit, so it gives weak ranking signal.')
    if not d.bo_v_trainable and abs(d.value_grad_sum) > d.value_grad_norm:
        notes.append('value bias is frozen while BCE wants a global shift; this can slow value-scale learning.')
    if d.scaled_rank_grad_norm > 0.0:
        if d.value_rank_cosine < -0.2 or d.value_rank_sign_conflict > 0.4:
            notes.append('rank auxiliary conflicts with BCE on many states.')
        elif d.value_rank_cosine < 0.2:
            notes.append('rank auxiliary is mostly not aligned with BCE.')
    if d.policy_to_value_grad_ratio > 5.0:
        notes.append('policy output gradient is much larger than value output gradient; shared trunk may follow policy.')
    if d.pred_contrast_mean < d.target_contrast_mean * 0.25:
        notes.append('predicted values have much less within-sequence contrast than targets.')
    if not notes:
        notes.append('no obvious aggregate bottleneck in this generated test.')
    print('  diagnosis: ' + ' '.join(notes))


def main():
    args = parse_args()
    configs = [(args.rank_mix,args.target_mode,'requested')]
    if not args.single:
        configs += [
            (0.0,args.target_mode,'bce_only'),
            (args.rank_mix,args.target_mode,'bce_plus_rank'),
            (0.0,'linear','linear_targets_bce'),
            (0.0,'binary-final','binary_final_bce'),
        ]
    diagnostics = [
        run_diagnostic(args,rank_mix,target_mode,name)
        for rank_mix,target_mode,name in configs
    ]
    print_report(diagnostics)


if __name__ == '__main__':
    main()
