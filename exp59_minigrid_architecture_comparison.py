# -*- coding: utf-8 -*-
"""EXP59: controlled MiniGrid architecture comparison.

All models receive the same expert demonstrations from official MiniGrid and
are first behavior-cloned.  The evaluation stream hides mission text and task
labels, changes task families without boundaries, and gives episode feedback
only after the episode.  This controls the otherwise severe sparse-reward PPO
confound and compares the models' use of partial-observation history:

MLP (current view), GRU (recurrent history), Transformer (attention history),
and Phi+MLP (the same MLP plus an online action-organization state).
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

from exp56_minigrid import _expert_action


SEEDS = [0, 1, 2, 3, 4]
ENV_IDS = {
    "nav": "MiniGrid-Empty-5x5-v0",
    "door": "MiniGrid-DoorKey-5x5-v0",
    "lava": "MiniGrid-LavaGapS5-v0",
    "corridor": "MiniGrid-KeyCorridorS3R1-v0",
}
METHODS = ("mlp", "gru", "transformer", "phi_mlp", "phi_transformer")
DISPLAY = {
    "mlp": "MLP-BC",
    "gru": "GRU-BC",
    "transformer": "Transformer-BC",
    "phi_mlp": "Phi+MLP-BC",
    "phi_transformer": "Phi+Transformer-BC",
}
HISTORY = 8
DEMO_EPISODES_PER_ENV = 8
EPOCHS = 8
BATCH_SIZE = 256
RESULT_JSON = os.path.join("results", "EXP59_minigrid_architecture_comparison.json")
RESULT_PNG = os.path.join("results", "EXP59_minigrid_architecture_comparison.png")


def _obs_vector(obs):
    image = np.asarray(obs["image"], dtype=np.float32) / 10.0
    direction = np.zeros(4, dtype=np.float32)
    direction[int(obs["direction"])] = 1.0
    return np.concatenate([image.reshape(-1), direction]).astype(np.float32)


def _obs_key(obs):
    image = np.asarray(obs["image"], dtype=np.int16)
    hist = np.bincount(image[:, :, 0].ravel(), minlength=11)[:11]
    colors = np.bincount(image[:, :, 1].ravel(), minlength=6)[:6]
    states = np.bincount(image[:, :, 2].ravel(), minlength=3)[:3]
    return tuple(np.concatenate([hist, colors, states, [int(obs["direction"])]], dtype=np.int16))


def _collect_demos(seed):
    rng = np.random.default_rng(seed)
    sequences, actions = [], []
    for family, env_id in ENV_IDS.items():
        for episode in range(DEMO_EPISODES_PER_ENV):
            env = gym.make(env_id)
            obs, _ = env.reset(seed=int(rng.integers(2**31 - 1)))
            history = deque(maxlen=HISTORY)
            vec = _obs_vector(obs)
            for _ in range(HISTORY):
                history.append(vec.copy())
            for _ in range(env.unwrapped.max_steps):
                sequences.append(np.stack(history, axis=0))
                action = _expert_action(env, {"nav": "nav", "door": "keydoor", "lava": "safe", "corridor": "corridor"}[family])
                actions.append(action)
                obs, reward, terminated, truncated, _ = env.step(action)
                vec = _obs_vector(obs)
                history.append(vec.copy())
                if terminated or truncated:
                    break
            env.close()
    return np.asarray(sequences, dtype=np.float32), np.asarray(actions, dtype=np.int64)


class BasePolicy(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim

    def encode(self, sequence):
        raise NotImplementedError

    def forward(self, sequence):
        return self.actor(self.encode(sequence))


class MLPPolicy(BasePolicy):
    def __init__(self, input_dim):
        super().__init__(input_dim)
        self.body = nn.Sequential(nn.Linear(input_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.actor = nn.Linear(64, 7)

    def encode(self, sequence):
        return self.body(sequence[:, -1])


class GRUPolicy(BasePolicy):
    def __init__(self, input_dim):
        super().__init__(input_dim)
        self.input = nn.Linear(input_dim, 64)
        self.gru = nn.GRU(64, 64, batch_first=True)
        self.actor = nn.Linear(64, 7)

    def encode(self, sequence):
        z = torch.tanh(self.input(sequence))
        _, h = self.gru(z)
        return h[-1]


class TransformerPolicy(BasePolicy):
    def __init__(self, input_dim):
        super().__init__(input_dim)
        self.input = nn.Linear(input_dim, 64)
        self.pos = nn.Parameter(torch.zeros(1, HISTORY, 64))
        layer = nn.TransformerEncoderLayer(64, 4, 128, dropout=0.0, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.actor = nn.Linear(64, 7)

    def encode(self, sequence):
        z = torch.tanh(self.input(sequence)) + self.pos
        mask = torch.triu(torch.ones(HISTORY, HISTORY), diagonal=1).bool().to(sequence.device)
        return self.encoder(z, mask=mask)[:, -1]


class PhiTable:
    def __init__(self, action_dim=7, lr=0.2):
        self.rows = {}
        self.action_dim = action_dim
        self.lr = lr

    def bias(self, key):
        return self.rows.get(key, np.zeros(self.action_dim, dtype=np.float32)).copy()

    def update(self, key, action, success):
        row = self.rows.setdefault(key, np.zeros(self.action_dim, dtype=np.float32))
        target = 1.0 if success else -1.0
        row[action] = (1.0 - self.lr) * row[action] + self.lr * target


def _make(kind, input_dim):
    return MLPPolicy(input_dim) if kind in ("mlp", "phi_mlp") else GRUPolicy(input_dim) if kind == "gru" else TransformerPolicy(input_dim)


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
        for _ in range(3):
            stream.extend((phase, x) for x in block)
    return stream


def _run_episode(model, phi, family, seed):
    env = gym.make(ENV_IDS[family])
    obs, _ = env.reset(seed=seed)
    history = deque(maxlen=HISTORY)
    vec = _obs_vector(obs)
    for _ in range(HISTORY):
        history.append(vec.copy())
    context = _obs_key(obs)
    total = 0.0
    for step in range(env.unwrapped.max_steps):
        sequence = torch.as_tensor(np.stack(history)[None], dtype=torch.float32)
        with torch.no_grad():
            logits = model(sequence)
            if phi is not None:
                logits = logits + 0.35 * torch.as_tensor(phi.bias(context))[None]
            action = int(torch.argmax(logits, dim=-1).item())
        obs, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        history.append(_obs_vector(obs))
        if terminated or truncated:
            break
    env.close()
    return bool(total > 0.0), step + 1, context, action


def _evaluate(model, kind, seed):
    phi = PhiTable() if kind in ("phi_mlp", "phi_transformer") else None
    records = []
    for episode, (phase, family) in enumerate(_schedule(seed)):
        success, steps, context, action = _run_episode(model, phi, family, seed * 10000 + episode)
        if phi is not None:
            phi.update(context, action, success)
        records.append({"phase": phase, "family": family, "success": int(success), "steps": steps})
    return records, phi


def _summarize(records, model, phi):
    phase = {p: float(np.mean([x["success"] for x in records if x["phase"] == p])) for p in ("calibration", "switch", "revisit")}
    family = {f: float(np.mean([x["success"] for x in records if x["family"] == f])) for f in ENV_IDS}
    return {
        "overall_success_rate": float(np.mean([x["success"] for x in records])),
        "phase_success_rate": phase,
        "family_success_rate": family,
        "mean_steps": float(np.mean([x["steps"] for x in records])),
        "trainable_parameters": int(sum(p.numel() for p in model.parameters())),
        "dynamic_state_floats": int(len(phi.rows) * phi.action_dim) if phi is not None else 0,
        "phi_contexts": int(len(phi.rows)) if phi is not None else 0,
    }


def _plot(summary):
    names = list(METHODS)
    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, key, title in ((axes[0], "overall_success_rate", "Overall success"), (axes[1], "switch", "Switch success"), (axes[2], "revisit", "Revisit success")):
        values = [summary[n]["mean"][key] if key == "overall_success_rate" else summary[n]["mean"]["phase_success_rate"][key] for n in names]
        ax.bar(x, values, color=["#64748b", "#0ea5e9", "#8b5cf6", "#10b981"])
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.set_xticks(x, [DISPLAY[n] for n in names], rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("EXP59 MiniGrid: controlled architecture comparison")
    fig.tight_layout()
    fig.savefig(RESULT_PNG, dpi=170)
    plt.close(fig)


def run_exp59(seeds=SEEDS):
    os.makedirs("results", exist_ok=True)
    X_parts, y_parts = [], []
    for seed in seeds:
        X, y = _collect_demos(seed)
        X_parts.append(X)
        y_parts.append(y)
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    raw = {kind: [] for kind in METHODS}
    for kind in METHODS:
        for seed in seeds:
            model = _train(kind, X, y, seed)
            records, phi = _evaluate(model, kind, seed)
            raw[kind].append(_summarize(records, model, phi))
    summary = {}
    for kind in METHODS:
        rows = raw[kind]
        summary[kind] = {
            "mean": {
                "overall_success_rate": float(np.mean([x["overall_success_rate"] for x in rows])),
                "mean_steps": float(np.mean([x["mean_steps"] for x in rows])),
                "trainable_parameters": float(np.mean([x["trainable_parameters"] for x in rows])),
                "dynamic_state_floats": float(np.mean([x["dynamic_state_floats"] for x in rows])),
                "phase_success_rate": {p: float(np.mean([x["phase_success_rate"][p] for x in rows])) for p in ("calibration", "switch", "revisit")},
                "family_success_rate": {f: float(np.mean([x["family_success_rate"][f] for x in rows])) for f in ENV_IDS},
            },
            "std": {"overall_success_rate": float(np.std([x["overall_success_rate"] for x in rows], ddof=1))},
            "per_seed": rows,
        }
    _plot(summary)
    result = {
        "experiment": "EXP59",
        "title": "Controlled MiniGrid architecture comparison with hidden missions",
        "protocol": {
            "envs": ENV_IDS,
            "seeds": list(seeds),
            "mission_visible": False,
            "task_label_visible": False,
            "history": HISTORY,
            "demonstrations_per_env_per_seed": DEMO_EPISODES_PER_ENV,
            "behavior_cloning_epochs": EPOCHS,
            "evaluation_episodes_per_seed": len(_schedule(seeds[0])),
            "training_note": "expert trajectories control sparse-reward PPO confounding; expert is not available at evaluation",
        },
        "dataset": {"samples": int(len(y)), "input_dim": int(X.shape[-1]), "class_counts": np.bincount(y, minlength=7).tolist()},
        "summary": summary,
    }
    with open(RESULT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


if __name__ == "__main__":
    result = run_exp59()
    for kind in METHODS:
        print(DISPLAY[kind], result["summary"][kind]["mean"])
