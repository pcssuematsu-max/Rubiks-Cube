"""Diagnose Search2 value-loss geometry on generated training data."""

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

from ai.losses import MyLoss2, MyLoss2Pairwise, Myloss
from ai.rubiks_ai import Rubiks_3_AI
from model.search_result import data


@dataclass
class LossDiagnostic:
    name: str
    loss_type: str
    rank_apply_type: str
    rank_mix: float
    margin: float
    policy_loss: float
    value_total: float
    base_name: str
    base_loss: float
    rank_raw: float
    rank_scaled: float
    seq_count: int
    state_count: int
    value_mean: float
    value_std: float
    value_range: float
    diff_mean: float
    diff_std: float
    diff_min: float
    diff_max: float
    pair_violation_rate: float
    final_minus_start_mean: float
    final_minus_start_min: float
    final_minus_start_max: float
    base_grad_norm: float
    rank_grad_norm: float
    scaled_rank_grad_norm: float
    combined_grad_norm: float
    grad_cosine: float
    grad_sign_conflict_rate: float
    distance_uniform_gap_mean: float


def parse_args():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('--cube-size', type = int, default = 3)
    parser.add_argument('--samples', type = int, default = 64)
    parser.add_argument('--min-len', type = int, default = 2)
    parser.add_argument('--max-len', type = int, default = 8)
    parser.add_argument('--mid', type = int, default = 64)
    parser.add_argument('--layers', type = int, default = 2)
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--margin', type = float, default = 0.2)
    parser.add_argument('--rank-mix', type = float, default = 1.0)
    parser.add_argument('--rank-apply', default = 'distance')
    parser.add_argument('--loss-type', default = 'myloss2_pairwise')
    parser.add_argument('--single', action = 'store_true', help = 'Only run the requested loss configuration.')
    return parser.parse_args()


def generated_search2_data(cube, sample_count, min_len, max_len, rng):
    move_pool = face_move_pool(cube)
    samples = []
    for _ in range(sample_count):
        length = rng.randint(min_len,max_len)
        scramble = tuple(rng.choice(move_pool) for _ in range(length))
        moves = tuple(cube.invert_moves(scramble))
        samples.append(data(
            scramble,
            moves,
            None,
            source_search_mode = 'diagnostic',
            source_search2_value_loss_type = 'diagnostic',
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


def build_ai(args, loss_type, rank_apply_type):
    mid = [int(args.mid)] * int(args.layers)
    return Rubiks_3_AI(
        mid,
        cube_size = args.cube_size,
        search2_value_loss_type = loss_type,
        search2_value_loss_margin = args.margin,
        search2_rank_loss_mix = args.rank_mix,
        search2_rank_loss_apply_type = rank_apply_type,
    )


def run_diagnostic(args, samples, loss_type, rank_apply_type, name):
    np.random.seed(args.seed)
    random.seed(args.seed)
    ai = build_ai(args, loss_type, rank_apply_type)
    loss_inputs = ai._build_loss_inputs(samples)
    out = ai._predict_loss_outputs(loss_inputs['x'])
    policy_loss, value_loss = ai._compute_search2_losses(
        out,
        loss_inputs['args'],
        loss_inputs['indices'],
        loss_inputs['value_columns'],
        loss_inputs['value_indices'],
        loss_inputs['value_steps_to_goal'],
    )
    columns = loss_inputs['value_columns']
    values = out[-1:,columns]
    value_indices = loss_inputs['value_indices']
    steps = loss_inputs['value_steps_to_goal']
    base_name, base_loss, base_grad = base_loss_and_grad(ai, values, value_indices, steps)
    rank_loss, rank_grad = rank_loss_and_grad(values, value_indices)
    rank_applied = ai._uses_search2_rank_mix()
    scaled_rank_grad = args.rank_mix * rank_grad if rank_applied else np.zeros_like(rank_grad)
    combined_grad = base_grad + scaled_rank_grad
    diffs = adjacent_diffs(values, value_indices)
    final_minus_start = sequence_final_minus_start(values, value_indices)
    return LossDiagnostic(
        name = name,
        loss_type = ai.search2_value_loss_type,
        rank_apply_type = ai.search2_rank_loss_apply_type,
        rank_mix = ai.search2_rank_loss_mix,
        margin = ai.search2_value_loss_margin,
        policy_loss = float(policy_loss),
        value_total = float(value_loss),
        base_name = base_name,
        base_loss = float(base_loss),
        rank_raw = float(rank_loss),
        rank_scaled = float(args.rank_mix * rank_loss if rank_applied else 0.0),
        seq_count = len(value_indices) - 1,
        state_count = int(values.size),
        value_mean = safe_mean(values),
        value_std = safe_std(values),
        value_range = safe_range(values),
        diff_mean = safe_mean(diffs),
        diff_std = safe_std(diffs),
        diff_min = safe_min(diffs),
        diff_max = safe_max(diffs),
        pair_violation_rate = violation_rate(diffs, ai.search2_value_loss_margin),
        final_minus_start_mean = safe_mean(final_minus_start),
        final_minus_start_min = safe_min(final_minus_start),
        final_minus_start_max = safe_max(final_minus_start),
        base_grad_norm = norm(base_grad),
        rank_grad_norm = norm(rank_grad),
        scaled_rank_grad_norm = norm(scaled_rank_grad),
        combined_grad_norm = norm(combined_grad),
        grad_cosine = cosine(base_grad, scaled_rank_grad),
        grad_sign_conflict_rate = sign_conflict_rate(base_grad, scaled_rank_grad),
        distance_uniform_gap_mean = distance_uniform_gap_mean(ai, value_indices, steps),
    )


def base_loss_and_grad(ai, values, indices, steps):
    if ai.search2_value_loss_type == 'myloss':
        layer = Myloss()
        return 'myloss', layer.forward(values,indices), layer.backward()
    if ai.search2_value_loss_type == 'myloss2_pairwise':
        layer = MyLoss2Pairwise(ai.search2_value_loss_margin)
        return 'myloss2_pairwise', layer.forward(values,indices), layer.backward()
    layer = MyLoss2()
    return 'myloss2_distance', layer.forward(values,indices,steps,ai.value_target_gamma), layer.backward()


def rank_loss_and_grad(values, indices):
    layer = Myloss()
    loss = layer.forward(values,indices)
    return loss, layer.backward()


def adjacent_diffs(values, indices):
    pieces = []
    row = values.reshape(-1)
    for start,end in zip(indices[:-1],indices[1:]):
        if end - start > 1:
            pieces.append(row[start + 1:end] - row[start:end - 1])
    if not pieces:
        return np.zeros(0,dtype = 'f')
    return np.concatenate(pieces)


def sequence_final_minus_start(values, indices):
    row = values.reshape(-1)
    diffs = []
    for start,end in zip(indices[:-1],indices[1:]):
        if end > start:
            diffs.append(row[end - 1] - row[start])
    return np.asarray(diffs,dtype = 'f')


def distance_uniform_gap_mean(ai, indices, steps):
    if steps is None or len(steps) == 0:
        return 0.0
    gaps = []
    gamma = ai.value_target_gamma
    for start,end in zip(indices[:-1],indices[1:]):
        n = end - start
        if n <= 0:
            continue
        target = np.asarray(gamma,dtype = 'f') ** steps[start:end]
        target_sum = float(np.sum(target))
        if target_sum <= 0:
            continue
        target = target / target_sum
        uniform_loss = math.log(n)
        target_entropy = -float(np.sum(target * np.log(target + 1.0e-12)))
        gaps.append(uniform_loss - target_entropy)
    return safe_mean(np.asarray(gaps,dtype = 'f'))


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


def violation_rate(diffs, margin):
    if diffs.size == 0:
        return 0.0
    return float(np.mean(diffs < margin))


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


def safe_range(x):
    x = np.asarray(x)
    return float(np.max(x) - np.min(x)) if x.size else 0.0


def print_report(diagnostics):
    print('Search2 value-loss diagnostic')
    print()
    for d in diagnostics:
        print(f'[{d.name}] type={d.loss_type} rank_apply={d.rank_apply_type} rank_mix={d.rank_mix:.3g} margin={d.margin:.3g}')
        print(f'  data: sequences={d.seq_count} states={d.state_count}')
        print(f'  loss: policy={d.policy_loss / max(1,d.seq_count):.6f}/seq value_total={d.value_total / max(1,d.seq_count):.6f}/seq')
        print(
            f'  components: {d.base_name}={d.base_loss / max(1,d.seq_count):.6f}/seq '
            f'rank_raw={d.rank_raw / max(1,d.seq_count):.6f}/seq '
            f'rank_scaled={d.rank_scaled / max(1,d.seq_count):.6f}/seq'
        )
        print(
            f'  values: mean={d.value_mean:.6f} std={d.value_std:.6f} range={d.value_range:.6f} '
            f'final-start mean/min/max={d.final_minus_start_mean:.6f}/{d.final_minus_start_min:.6f}/{d.final_minus_start_max:.6f}'
        )
        print(
            f'  pairwise: diff mean/std/min/max={d.diff_mean:.6f}/{d.diff_std:.6f}/{d.diff_min:.6f}/{d.diff_max:.6f} '
            f'violation(<margin)={100*d.pair_violation_rate:.1f}%'
        )
        print(
            f'  grad(value logits): base_norm={d.base_grad_norm:.6f} '
            f'rank_norm={d.rank_grad_norm:.6f} scaled_rank_norm={d.scaled_rank_grad_norm:.6f} '
            f'combined_norm={d.combined_grad_norm:.6f} cosine(base,rank)={d.grad_cosine:.4f} '
            f'sign_conflict={100*d.grad_sign_conflict_rate:.1f}%'
        )
        if d.distance_uniform_gap_mean > 0:
            print(f'  distance target: uniform_loss_gap={d.distance_uniform_gap_mean:.6f}/seq')
        print_diagnosis(d)
        print()


def print_diagnosis(d):
    notes = []
    if d.loss_type == 'myloss2' and d.distance_uniform_gap_mean < 0.01:
        notes.append('distance target is nearly uniform; visible loss decrease can be tiny.')
    if d.loss_type == 'myloss2_pairwise' and d.margin >= 0.5:
        notes.append('pairwise margin is strong; value scale must grow quickly to satisfy adjacent gaps.')
    if d.scaled_rank_grad_norm > 0:
        ratio = d.scaled_rank_grad_norm / max(d.base_grad_norm,1.0e-12)
        if d.grad_cosine < -0.25 or d.grad_sign_conflict_rate > 0.4:
            notes.append('base and rank gradients conflict strongly.')
        elif d.grad_cosine < 0.1:
            notes.append('base and rank gradients are nearly orthogonal; mixed training may look noisy.')
        if ratio > 2.0:
            notes.append('rank gradient dominates the base value loss.')
        elif ratio < 0.2:
            notes.append('rank gradient is much weaker than the base value loss.')
    if d.value_std < 1.0e-3:
        notes.append('value outputs are almost flat at initialization/current params.')
    if d.loss_type == 'myloss2_pairwise' and d.pair_violation_rate > 0.8:
        notes.append('most adjacent pairs violate the margin; pairwise loss is the immediate bottleneck.')
    if not notes:
        notes.append('no obvious loss-geometry bottleneck from these aggregate metrics.')
    print('  diagnosis: ' + ' '.join(notes))


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    base_ai = build_ai(args, args.loss_type, args.rank_apply)
    samples = generated_search2_data(base_ai.cube,args.samples,args.min_len,args.max_len,rng)
    if args.single:
        configs = [(args.loss_type,args.rank_apply,'requested')]
    else:
        configs = [
            (args.loss_type,args.rank_apply,'requested'),
            ('myloss2_pairwise','none','pairwise_only'),
            ('myloss2_pairwise','all','pairwise_plus_rank'),
            ('myloss2','distance','distance_plus_rank'),
            ('myloss','all','myloss_rank_only'),
        ]
    diagnostics = [
        run_diagnostic(args,samples,loss_type,rank_apply_type,name)
        for loss_type,rank_apply_type,name in configs
    ]
    print_report(diagnostics)


if __name__ == '__main__':
    main()
