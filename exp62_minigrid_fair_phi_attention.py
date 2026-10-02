# -*- coding: utf-8 -*-
"""EXP62: strict MiniGrid comparison with Phi inside self-attention.

This is a repaired protocol following the failures diagnosed in EXP60/61.

Repairs:
* MultiRoom is excluded because the previous privileged demonstrator did not
  solve it reliably.
* Demonstrations are retained only when the privileged expert succeeds.
* Every task family contributes the same number of behavior-cloning windows.
* MLP, GRU, Transformer and Phi-Transformer use near-equal trainable budgets.
* Phi is not an action-logit bias.  It is an external dynamic relative-key
  bias added directly to every self-attention score.
* Phi is updated from the complete episode attention trace, not from the last
  action only.  The update remains causal: episode feedback affects only later
  episodes.

This remains a controlled behavior-cloning experiment, not an end-to-end RL
benchmark.  It asks whether an explicitly attention-level, feedback-updated
state can improve task switching after a common representation has been
trained on balanced successful demonstrations.
"""
from __future__ import annotations

import json
import os
from collections import deque

import gymnasium as gym
import matplotlib.pyplot as plt
import minigrid  # noqa: F401
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import exp59_minigrid_architecture_comparison as base
from exp56_minigrid import _front_pos, _obj_at, _passable


SEEDS = [0, 1, 2, 3, 4]
ENV_IDS = {
    "nav": "MiniGrid-Empty-5x5-v0",
    "door": "MiniGrid-DoorKey-5x5-v0",
    "lava": "MiniGrid-LavaGapS5-v0",
    "corridor": "MiniGrid-KeyCorridorS3R1-v0",
    "memory": "MiniGrid-MemoryS7-v0",
    "unlock": "MiniGrid-Unlock-v0",
}
EXPERT_STRATEGY = {
    "nav": "nav",
    "door": "keydoor",
    "lava": "safe",
    "corridor": "corridor",
    "unlock": "keydoor",
}
METHODS = ("mlp", "gru", "transformer", "phi_transformer")
DISPLAY = {
    "mlp": "MLP-BC",
    "gru": "GRU-BC",
    "transformer": "Transformer-BC",
    "phi_transformer": "Phi-Transformer-BC",
}
HISTORY = 8
DEMO_EPISODES_PER_ENV = 12
WINDOWS_PER_FAMILY = 256
EPOCHS = 12
BATCH_SIZE = 256
EVAL_REPEATS = 3
PHI_LR = 0.35
PHI_CLIP = 1.5
RESULT_JSON = os.path.join("results", "EXP62_minigrid_fair_phi_attention.json")
RESULT_PNG = os.path.join("results", "EXP62_minigrid_fair_phi_attention.png")


def _bfs_avoiding(env, target, forbidden=()):
    start = tuple(map(int, env.unwrapped.agent_pos))
    target = tuple(map(int, target))
    forbidden = {tuple(map(int, p)) for p in forbidden}
    if start == target:
        return [start]
    queue = deque([start])
    parent = {start: None}
    while queue:
        cur = queue.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (cur[0] + dx, cur[1] + dy)
            if not (0 <= nxt[0] < env.unwrapped.width and 0 <= nxt[1] < env.unwrapped.height):
                continue
            if nxt in parent or nxt in forbidden:
                continue
            if not _passable(_obj_at(env, nxt), avoid_lava=True):
                continue
            parent[nxt] = cur
            if nxt == target:
                path = [nxt]
                while path[-1] != start:
                    path.append(parent[path[-1]])
                return list(reversed(path))
            queue.append(nxt)
    return [start]


def _memory_expert_action(env):
    action = env.unwrapped.actions
    target = tuple(map(int, env.unwrapped.success_pos))
    failure = tuple(map(int, env.unwrapped.failure_pos))
    if _front_pos(env) == failure:
        return int(action.right)
    path = _bfs_avoiding(env, target, forbidden=(failure,))
    if len(path) <= 1:
        return int(action.forward)
    here = tuple(map(int, env.unwrapped.agent_pos))
    nxt = path[1]
    desired = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}[(nxt[0] - here[0], nxt[1] - here[1])]
    delta = (desired - int(env.unwrapped.agent_dir)) % 4
    if delta == 0:
        return int(action.forward)
    return int(action.right if delta == 1 else action.left)


def _expert_action(env, family):
    if family == "memory":
        return _memory_expert_action(env)
    return base._expert_action(env, EXPERT_STRATEGY[family])


def _obs_vector(obs):
    image = np.asarray(obs["image"], dtype=np.float32) / 10.0
    direction = np.zeros(4, dtype=np.float32)
    direction[int(obs["direction"])] = 1.0
    return np.concatenate([image.reshape(-1), direction]).astype(np.float32)


def _collect_family_windows(family, env_id, seed):
    """Collect balanced windows and reject any episode the expert cannot solve."""
    rng = np.random.default_rng(seed)
    windows, labels = [], []
    successful_episodes = 0
    attempts = 0
    while len(windows) < WINDOWS_PER_FAMILY and attempts < DEMO_EPISODES_PER_ENV * 10:
        attempts += 1
        env = gym.make(env_id)
        obs, _ = env.reset(seed=int(rng.integers(2**31 - 1)))
        history = deque(maxlen=HISTORY)
        vec = _obs_vector(obs)
        for _ in range(HISTORY):
            history.append(vec.copy())
        episode_windows, episode_labels = [], []
        total_reward = 0.0
        for _ in range(env.unwrapped.max_steps):
            episode_windows.append(np.stack(history))
            action = _expert_action(env, family)
            episode_labels.append(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            history.append(_obs_vector(obs))
            if terminated or truncated:
                break
        env.close()
        if total_reward <= 0.0:
            continue
        successful_episodes += 1
        windows.extend(episode_windows)
        labels.extend(episode_labels)
    if len(windows) < WINDOWS_PER_FAMILY:
        raise RuntimeError(
            f"expert validation failed for {family}: only {len(windows)} windows "
            f"after {attempts} attempts"
        )
    order = rng.permutation(len(windows))[:WINDOWS_PER_FAMILY]
    return np.asarray([windows[i] for i in order], dtype=np.float32), np.asarray([labels[i] for i in order], dtype=np.int64), successful_episodes


class MLPPolicy(nn.Module):
    def __init__(self, input_dim, hidden=62):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(input_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.actor = nn.Linear(hidden, 7)

    def forward(self, sequence, phi_bias=None, return_attention=False):
        del phi_bias, return_attention
        return self.actor(self.body(sequence[:, -1]))


class GRUPolicy(nn.Module):
    def __init__(self, input_dim, hidden=36):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.actor = nn.Linear(hidden, 7)

    def forward(self, sequence, phi_bias=None, return_attention=False):
        del phi_bias, return_attention
        z = torch.tanh(self.input(sequence))
        _, h = self.gru(z)
        return self.actor(h[-1])


class AttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, ff_dim)
        self.ff2 = nn.Linear(ff_dim, d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, phi_bias=None):
        batch, length, _ = x.shape
        residual = x
        q = self.q(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        causal = torch.triu(torch.ones(length, length, device=x.device, dtype=torch.bool), diagonal=1)
        logits = logits.masked_fill(causal[None, None], -1e9)
        if phi_bias is not None:
            # Phi is a per-head relative-key prior: [heads, key_position].
            if phi_bias.ndim == 3:
                phi_bias = phi_bias.to(logits.device)
            logits = logits + phi_bias[None, :, None, :length]
        weights = torch.softmax(logits, dim=-1)
        attended = torch.matmul(weights, v).transpose(1, 2).contiguous().view(batch, length, self.d_model)
        x = self.norm1(residual + self.out(attended))
        x = self.norm2(x + self.ff2(F.gelu(self.ff1(x))))
        return x, weights


class TransformerPolicy(nn.Module):
    def __init__(self, input_dim, d_model=24, n_heads=4, ff_dim=48, layers=2):
        super().__init__()
        self.input = nn.Linear(input_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, HISTORY, d_model))
        self.blocks = nn.ModuleList([AttentionBlock(d_model, n_heads, ff_dim) for _ in range(layers)])
        self.actor = nn.Linear(d_model, 7)

    def forward(self, sequence, phi_bias=None, return_attention=False):
        x = torch.tanh(self.input(sequence)) + self.pos
        traces = []
        for layer_index, block in enumerate(self.blocks):
            block_bias = phi_bias
            if phi_bias is not None and phi_bias.ndim == 3:
                block_bias = phi_bias[layer_index]
            x, weights = block(x, phi_bias=block_bias)
            traces.append(weights)
        logits = self.actor(x[:, -1])
        if return_attention:
            return logits, torch.stack(traces, dim=1)
        return logits


def _make(kind, input_dim):
    if kind == "mlp":
        return MLPPolicy(input_dim)
    if kind == "gru":
        return GRUPolicy(input_dim)
    return TransformerPolicy(input_dim)


class PhiAttentionState:
    """Dynamic relative-key bias injected into attention logits."""

    def __init__(self, heads=4, history=HISTORY, lr=PHI_LR):
        self.bias = np.zeros((heads, history), dtype=np.float32)
        self.lr = lr

    def tensor(self, device):
        return torch.as_tensor(self.bias, dtype=torch.float32, device=device)

    def update(self, attention_trace, success):
        if not attention_trace:
            return
        # Aggregate the last-query attention used across all steps and blocks.
        trace = np.stack(attention_trace, axis=0)
        # trace: [steps, transformer_layers, heads, relative_key]
        mean_attention = trace.mean(axis=(0, 1, 2))
        centered = mean_attention - (1.0 / mean_attention.shape[-1])
        reward = 1.0 if success else -1.0
        self.bias = np.clip(self.bias + self.lr * reward * centered[None], -PHI_CLIP, PHI_CLIP)


def _train(kind, X, y, seed):
    torch.manual_seed(seed)
    model = _make(kind, X.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    order = np.arange(len(y))
    tensor_x = torch.as_tensor(X, dtype=torch.float32)
    tensor_y = torch.as_tensor(y, dtype=torch.long)
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(EPOCHS):
        rng.shuffle(order)
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start:start + BATCH_SIZE]
            logits = model(tensor_x[idx])
            loss = nn.functional.cross_entropy(logits, tensor_y[idx])
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.eval()
    return model


def _schedule(seed):
    rng = np.random.default_rng(seed)
    families = list(ENV_IDS)
    stream = []
    for phase in ("calibration", "switch", "revisit"):
        block = families.copy()
        if phase != "calibration":
            rng.shuffle(block)
        for _ in range(EVAL_REPEATS):
            stream.extend((phase, family) for family in block)
    return stream


def _run_episode(model, phi, family, seed):
    env = gym.make(ENV_IDS[family])
    obs, _ = env.reset(seed=seed)
    history = deque(maxlen=HISTORY)
    vec = _obs_vector(obs)
    for _ in range(HISTORY):
        history.append(vec.copy())
    attention_trace = []
    total = 0.0
    steps = 0
    for steps in range(env.unwrapped.max_steps):
        sequence = torch.as_tensor(np.stack(history)[None], dtype=torch.float32)
        with torch.no_grad():
            if phi is None:
                logits, weights = model(sequence, return_attention=True) if isinstance(model, TransformerPolicy) else (model(sequence), None)
            else:
                logits, weights = model(sequence, phi_bias=phi.tensor(sequence.device), return_attention=True)
            action = int(torch.argmax(logits, dim=-1).item())
        if weights is not None:
            # Last query's attention, averaged over Transformer blocks.
            attention_trace.append(weights[0, :, :, -1, :].detach().cpu().numpy())
        obs, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        history.append(_obs_vector(obs))
        if terminated or truncated:
            break
    env.close()
    return bool(total > 0.0), steps + 1, attention_trace


def _evaluate(model, kind, seed):
    phi = PhiAttentionState() if kind == "phi_transformer" else None
    records = []
    for episode, (phase, family) in enumerate(_schedule(seed)):
        success, steps, trace = _run_episode(model, phi, family, seed * 10000 + episode)
        if phi is not None:
            phi.update(trace, success)
        records.append({"phase": phase, "family": family, "success": int(success), "steps": steps})
    return records, phi


def _summarize(records, model, phi):
    phases = ("calibration", "switch", "revisit")
    return {
        "overall_success_rate": float(np.mean([r["success"] for r in records])),
        "phase_success_rate": {p: float(np.mean([r["success"] for r in records if r["phase"] == p])) for p in phases},
        "family_success_rate": {f: float(np.mean([r["success"] for r in records if r["family"] == f])) for f in ENV_IDS},
        "mean_steps": float(np.mean([r["steps"] for r in records])),
        "trainable_parameters": int(sum(p.numel() for p in model.parameters())),
        "dynamic_state_floats": int(phi.bias.size) if phi is not None else 0,
        "phi_bias_l1": float(np.abs(phi.bias).sum()) if phi is not None else 0.0,
    }


def _plot(summary):
    names = list(METHODS)
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, key, title in ((axes[0], "overall_success_rate", "Overall success"), (axes[1], "switch", "Switch success"), (axes[2], "revisit", "Revisit success")):
        values = [summary[n]["mean"][key] if key == "overall_success_rate" else summary[n]["mean"]["phase_success_rate"][key] for n in names]
        errors = [summary[n]["std"][key] if key == "overall_success_rate" else summary[n]["std"]["phase_success_rate"][key] for n in names]
        ax.bar(x, values, yerr=errors, capsize=4, color=["#64748b", "#0ea5e9", "#8b5cf6", "#f97316"])
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.set_xticks(x, [DISPLAY[n] for n in names], rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("EXP62 MiniGrid: fair budget and Phi attention bias")
    fig.tight_layout()
    fig.savefig(RESULT_PNG, dpi=170)
    plt.close(fig)


def run_exp62(seeds=SEEDS):
    os.makedirs("results", exist_ok=True)
    family_parts = {}
    expert_stats = {}
    for family, env_id in ENV_IDS.items():
        parts_x, parts_y = [], []
        successes = 0
        for seed in seeds:
            x, y, count = _collect_family_windows(family, env_id, seed * 1000 + 17)
            parts_x.append(x)
            parts_y.append(y)
            successes += count
        family_parts[family] = (np.concatenate(parts_x), np.concatenate(parts_y))
        expert_stats[family] = {"successful_episodes": int(successes), "windows": int(len(family_parts[family][1]))}
    X = np.concatenate([family_parts[f][0] for f in ENV_IDS], axis=0)
    y = np.concatenate([family_parts[f][1] for f in ENV_IDS], axis=0)
    raw = {kind: [] for kind in METHODS}
    for kind in METHODS:
        for seed in seeds:
            model = _train(kind, X, y, seed)
            records, phi = _evaluate(model, kind, seed)
            raw[kind].append(_summarize(records, model, phi))
    summary = {}
    phases = ("calibration", "switch", "revisit")
    for kind in METHODS:
        rows = raw[kind]
        summary[kind] = {
            "mean": {
                "overall_success_rate": float(np.mean([r["overall_success_rate"] for r in rows])),
                "mean_steps": float(np.mean([r["mean_steps"] for r in rows])),
                "trainable_parameters": float(np.mean([r["trainable_parameters"] for r in rows])),
                "dynamic_state_floats": float(np.mean([r["dynamic_state_floats"] for r in rows])),
                "phi_bias_l1": float(np.mean([r["phi_bias_l1"] for r in rows])),
                "phase_success_rate": {p: float(np.mean([r["phase_success_rate"][p] for r in rows])) for p in phases},
                "family_success_rate": {f: float(np.mean([r["family_success_rate"][f] for r in rows])) for f in ENV_IDS},
            },
            "std": {
                "overall_success_rate": float(np.std([r["overall_success_rate"] for r in rows], ddof=1)),
                "phase_success_rate": {p: float(np.std([r["phase_success_rate"][p] for r in rows], ddof=1)) for p in phases},
            },
            "per_seed": rows,
        }
    _plot(summary)
    result = {
        "experiment": "EXP62",
        "title": "Fair MiniGrid comparison with Phi directly inside self-attention",
        "protocol": {
            "envs": ENV_IDS,
            "multiroom": "excluded because the previous expert was not validated",
            "seeds": list(seeds),
            "mission_visible": False,
            "task_label_visible": False,
            "successful_expert_only": True,
            "balanced_windows_per_family": WINDOWS_PER_FAMILY * len(seeds),
            "history": HISTORY,
            "evaluation_repeats": EVAL_REPEATS,
            "phi_update": "episode attention trace plus terminal success/failure; update affects later episodes only",
            "phi_attention_formula": "logits = QK^T/sqrt(d) + causal_mask + Phi(relative_key_bias)",
        },
        "expert_validation": expert_stats,
        "dataset": {"samples": int(len(y)), "input_dim": int(X.shape[-1]), "class_counts": np.bincount(y, minlength=7).tolist()},
        "parameter_budget": {kind: int(raw[kind][0]["trainable_parameters"]) for kind in METHODS},
        "summary": summary,
    }
    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


if __name__ == "__main__":
    result = run_exp62()
    for kind in METHODS:
        print(DISPLAY[kind], result["summary"][kind]["mean"])
