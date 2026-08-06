import gymnasium as gym
import network_sim
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import MlpLstmPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

import torch
import numpy as np
import json
import time
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# ─────────────────────────────────────────────
# HARDWARE CONFIGURATION
# ─────────────────────────────────────────────
# Must be set before any model construction or env spawning.
#
# torch.set_num_threads: controls intra-op parallelism for CPU tensor ops
# (policy update fallback, data loading). Set to match physical core count.
# On a 2-core cloud instance, values above 2 cause context-switching overhead
# with no throughput benefit.
#
# DEVICE="cuda": forces policy forward/backward passes onto the T4 GPU.
# SB3's device="auto" is unreliable in some cloud environments — explicit
# is safer. Env stepping always runs on CPU regardless of this setting.

torch.set_num_threads(1)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"{'='*60}")
print(f"Hardware configuration")
print(f"  CPU cores:       {os.cpu_count()}")
print(f"  PyTorch threads: {torch.get_num_threads()}")
print(f"  CUDA available:  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU:             {torch.cuda.get_device_name(0)}")
print(f"  Training device: {DEVICE}")
print(f"{'='*60}\n")


# ─────────────────────────────────────────────
# EXPERIMENT CONFIGURATION
# ─────────────────────────────────────────────

TOTAL_TIMESTEPS  = 5000_000
NUM_EVAL_EPISODES = 10

# N_ENVS=2 matches the 2 physical CPU cores on this cloud instance.
# SubprocVecEnv spawns one subprocess per env; more envs than cores causes
# context-switching overhead that hurts both env throughput and GPU utilisation.
#
# N_STEPS=400 = MAX_STEPS (the env episode length).
# Aligning rollout length with episode length guarantees two things:
#   1. PPO value bootstrapping is applied only at true terminal states,
#      not mid-episode truncations, eliminating value estimation bias.
#   2. RecurrentPPO unrolls the LSTM over the full N_STEPS sequence per
#      update. With N_STEPS < MAX_STEPS the hidden state resets mid-episode
#      at rollout boundaries, preventing LSTM from learning dependencies
#      that span the full episode. N_STEPS=400 gives LSTM uninterrupted
#      access to the complete episode trajectory in every update.
#
# Effective buffer = N_STEPS x N_ENVS = 400 x 2 = 800 transitions/update.
# BATCH_SIZE=400 = N_STEPS: one complete episode sequence per worker per
# minibatch. RecurrentPPO requires sequences are never split across
# minibatches; BATCH_SIZE=400 satisfies this exactly. One minibatch
# consumes the full buffer per update.
# Gradient updates = 1_000_000 / 800 = 1250 over the full training run.
N_ENVS     = 4
N_STEPS    = 400    # = MAX_STEPS: episode boundaries = rollout boundaries
BATCH_SIZE = 400    # = N_STEPS: full sequences intact, no hidden state resets

# 5 seeds — N = 50 observations per group, matching methodology section
SEEDS = [2, 7, 13, 18, 24]

MODELS = {
    "stacking3HL":  {"policy": "MlpPolicy",     "history_len": 3},
    "stacking5HL":  {"policy": "MlpPolicy",     "history_len": 5},
    "stacking10HL": {"policy": "MlpPolicy",     "history_len": 10},
    #"lstm":         {"policy": "MlpLstmPolicy", "history_len": 1},
}

# OOD evaluation scenarios (training distribution excluded per methodology)
SCENARIOS = {
    "crossrtt": {"bandwidth": 200, "latency": 0.08},
    "flat":     {"bandwidth": 200, "latency": 0.03},
    "step":     {"bandwidth": 200, "latency": 0.03, "step_change": 100},
}

# Checkpoint frequency: save every 50k timesteps (5% of budget).
# At most 50k timesteps of work is lost on an unexpected shutdown.
CKPT_FREQ = 50_000


# ─────────────────────────────────────────────
# DIRECTORY LAYOUT
# ─────────────────────────────────────────────
#
#   logs/
#     checkpoints/<model_type>_seed<seed>/
#       rl_model_<N>_steps.zip          ← periodic SB3 checkpoints
#       train_log_snapshot_<N>.json     ← training log state at same checkpoint
#     models/
#       <model_type>_seed<seed>.zip     ← final trained model
#     train/
#       <model_type>_seed_<seed_id>.jsonl  ← compact episode summaries
#     eval/
#       <model_type>_<scenario>_seed_<seed_id>.jsonl
#     cost_files/
#       cost_<model_type>_<scenario>_seed<seed>.json
#       training_times.json
#       param_counts.json
#     progress.json                     ← atomic ledger (single source of truth)

LOGS_DIR      = str(REPO_ROOT / "logs")
CKPT_DIR      = os.path.join(LOGS_DIR, "checkpoints")
MODELS_DIR    = os.path.join(LOGS_DIR, "models")
COST_DIR      = os.path.join(LOGS_DIR, "cost_files")
PROGRESS_FILE = os.path.join(LOGS_DIR, "progress.json")

for d in [LOGS_DIR, CKPT_DIR, MODELS_DIR, COST_DIR,
          os.path.join(LOGS_DIR, "train"),
          os.path.join(LOGS_DIR, "eval")]:
    os.makedirs(d, exist_ok=True)


# ─────────────────────────────────────────────
# PROGRESS LEDGER
# ─────────────────────────────────────────────
# Schema — one entry per (model_type, seed) pair:
#
#   progress[model_type][str(seed)] = {
#       "training_done":      bool,
#       "training_time_min":  float | null,   # accumulated across resume sessions
#       "param_count":        int   | null,
#       "eval_done":          [scenario, ...]
#   }
#
# All writes are atomic: written to a .tmp file then os.replace()d.
# A crash during the write never leaves a corrupt ledger.

def _load_progress():
    if not os.path.exists(PROGRESS_FILE):
        return {}
    with open(PROGRESS_FILE) as f:
        return json.load(f)

def _save_progress(progress):
    tmp = PROGRESS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(progress, f, indent=2)
    os.replace(tmp, PROGRESS_FILE)

def _get_entry(progress, model_type, seed):
    return progress.setdefault(model_type, {}).setdefault(str(seed), {
        "training_done":     False,
        "training_time_min": None,
        "param_count":       None,
        "eval_done":         [],
    })


# ─────────────────────────────────────────────
# PATH HELPERS
# ─────────────────────────────────────────────

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def _final_model_path(model_type, seed):
    return os.path.join(MODELS_DIR, f"{model_type}_seed{seed}.zip")

def _ckpt_subdir(model_type, seed):
    return os.path.join(CKPT_DIR, f"{model_type}_seed{seed}")

def _train_log_path(model_type, seed_id):
    return os.path.join(LOGS_DIR, "train", f"{model_type}_seed_{seed_id}.jsonl")

def _train_log_snapshot_path(model_type, seed, ckpt_steps):
    """Path for the training log snapshot saved alongside a checkpoint."""
    return os.path.join(
        _ckpt_subdir(model_type, seed),
        f"train_log_snapshot_{ckpt_steps}.json"
    )

def _latest_checkpoint(model_type, seed):
    """Return (path, num_timesteps) of the latest SB3 checkpoint, or (None, 0)."""
    subdir = _ckpt_subdir(model_type, seed)
    if not os.path.isdir(subdir):
        return None, 0
    zips = [f for f in os.listdir(subdir) if f.endswith(".zip")]
    if not zips:
        return None, 0
    def _steps(name):
        try:
            return int(name.split("_steps.zip")[0].split("_")[-1])
        except (ValueError, IndexError):
            return -1
    best = max(zips, key=_steps)
    steps = _steps(best)
    return os.path.join(subdir, best), steps


# ─────────────────────────────────────────────
# TRAINING LOG SNAPSHOT  (checkpoint consistency)
# ─────────────────────────────────────────────
# Problem: SubprocVecEnv workers run in subprocesses. Each worker writes its
# training log independently. When a checkpoint is saved at step N, the
# training logs may be mid-episode or have a different number of completed
# episodes than the checkpoint represents. On resume, network_sim.__init__
# clears the log files — but it clears them to empty, losing the pre-checkpoint
# episodes that were legitimately completed.
#
# Solution: at every checkpoint save, snapshot the current training log content
# for all workers into a JSON file stored alongside the checkpoint zip. On
# resume, restore the log files from the snapshot before spawning new workers.
# This guarantees the training log state is exactly consistent with the
# checkpoint — no partial episodes, no missing episodes.

def _save_train_log_snapshot(model_type, seed, ckpt_steps, env):
    """Read training log files and reward_ewma values; save alongside checkpoint.

    env is the live SubprocVecEnv. reward_ewma is read from each worker
    subprocess via get_attr so EWMA continuity is preserved on resume.
    Falls back to 0.0 per worker if get_attr fails.
    """
    try:
        ewma_values = env.get_attr("reward_ewma")   # list of length N_ENVS
    except Exception:
        ewma_values = [0.0] * N_ENVS

    snapshot = {}
    for rank in range(N_ENVS):
        seed_id = seed + rank
        path = _train_log_path(model_type, seed_id)
        content = ""
        if os.path.exists(path):
            with open(path) as f:
                content = f.read()
        snapshot[str(seed_id)] = {
            "log_content": content,
            "reward_ewma": ewma_values[rank],
        }

    snap_path = _train_log_snapshot_path(model_type, seed, ckpt_steps)
    tmp = snap_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(snapshot, f)
    os.replace(tmp, snap_path)


def _restore_train_log_snapshot(model_type, seed, ckpt_steps, env):
    """Restore training log files and reward_ewma from the checkpoint snapshot.

    env is the already-constructed SubprocVecEnv — workers have run __init__
    and cleared their log files. We overwrite the cleared files with snapshot
    content, then push the saved reward_ewma into each worker subprocess via
    env_method so the EWMA filter continues from where it left off rather than
    restarting from zero (which would cause a sawtooth in the convergence plot).

    Ordering contract:
      1. _make_train_env()                  — workers start, clear log files
      2. _restore_train_log_snapshot(env)   — restore files + EWMA into workers
      3. model.learn()                      — training resumes
    """
    snap_path = _train_log_snapshot_path(model_type, seed, ckpt_steps)
    if not os.path.exists(snap_path):
        print(f"  [warn] no snapshot for {model_type} seed {seed} "
              f"at step {ckpt_steps} — log and EWMA continuity not guaranteed")
        return

    with open(snap_path) as f:
        snapshot = json.load(f)

    for rank in range(N_ENVS):
        seed_id = seed + rank
        entry   = snapshot.get(str(seed_id), {})

        # Backward-compatible: old snapshots stored a plain string, new ones
        # store {"log_content": ..., "reward_ewma": ...}
        if isinstance(entry, dict):
            log_content = entry.get("log_content", "")
            ewma        = entry.get("reward_ewma", 0.0)
        else:
            log_content = entry   # old format: raw JSONL string
            ewma        = 0.0     # no EWMA in old snapshots — restart from 0

        # Restore log file — overwrite the empty file workers created in __init__
        path = _train_log_path(model_type, seed_id)
        with open(path, "w") as f:
            f.write(log_content)

        # Restore reward_ewma into the live worker subprocess.
        # restore_ewma() is defined in network_sim.SimulatedNetworkEnv.
        # Wrapped in try/except: a subprocess communication failure degrades
        # gracefully — EWMA restarts from 0 for that worker rather than
        # crashing the run.
        try:
            env.env_method("restore_ewma", ewma, indices=[rank])
        except Exception as e:
            print(f"  [warn] could not restore ewma for worker {rank}: {e}")


# ─────────────────────────────────────────────
# CHECKPOINT CALLBACK WITH LOG SNAPSHOT
# ─────────────────────────────────────────────

class SnapshotCheckpointCallback(CheckpointCallback):
    """Extends CheckpointCallback to atomically snapshot training logs.

    Called by SB3 after each checkpoint zip is successfully written.
    The snapshot is written atomically (tmp + replace) so a crash between
    the zip write and the snapshot write leaves the previous snapshot intact
    — the worst case is one checkpoint's worth of log data is lost on resume,
    which is acceptable.
    """
    def __init__(self, model_type, seed, **kwargs):
        super().__init__(**kwargs)
        self.model_type = model_type
        self.seed       = seed

    def _on_step(self) -> bool:
        result = super()._on_step()
        # super()._on_step() saves the zip when the frequency is hit.
        # self.training_env is the live SubprocVecEnv — passed to
        # _save_train_log_snapshot so it can read reward_ewma from each
        # worker subprocess via get_attr.
        if self.n_calls % self.save_freq == 0:
            ckpt_steps = self.model.num_timesteps
            _save_train_log_snapshot(
                self.model_type, self.seed, ckpt_steps,
                self.training_env,
            )
        return result


# ─────────────────────────────────────────────
# ENV FACTORY
# ─────────────────────────────────────────────

def _make_train_env_fn(model_type, seed, rank):
    """Factory for SubprocVecEnv — must be picklable (no lambda)."""
    config = MODELS[model_type]
    seed_id = seed + rank
    history_len = config["history_len"]

    def _init():
        return gym.make(
            "PccNs-v0",
            history_len=history_len,
            bandwidth=200,
            latency=0.03,
            queue=5,
            loss=0.0,
            model_type=model_type,
            seed_id=seed_id,
            phase="train",
        )
    return _init


# ─────────────────────────────────────────────
# PPO KWARGS  (all hyperparameters, methodology-fixed)
# ─────────────────────────────────────────────

def _ppo_kwargs(seed):
    # All values match the methodology section.
    # N_STEPS and BATCH_SIZE are unified across model types —
    # state representation is the sole independent variable.
    return dict(
        learning_rate=3e-4,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device=DEVICE,
        seed=seed,
        verbose=0,
    )

def _build_fresh_model(model_type, seed, env):
    kwargs = _ppo_kwargs(seed)
    if model_type == "lstm":
        return RecurrentPPO(
            MODELS[model_type]["policy"], env, **kwargs,
            policy_kwargs=dict(
                lstm_hidden_size=32,
                net_arch=dict(pi=[32], vf=[32])
            )
        )
    return PPO(MODELS[model_type]["policy"], env, **kwargs)

def _load_model(model_type, env, zip_path):
    kwargs = dict(env=env, device=DEVICE)
    if model_type == "lstm":
        return RecurrentPPO.load(zip_path, **kwargs)
    return PPO.load(zip_path, **kwargs)


# ─────────────────────────────────────────────
# TRAINING  (with checkpoint / resume)
# ─────────────────────────────────────────────
#
#   A) training_done=True + final .zip exists
#      → reload model, return immediately.
#
#   B) training_done=False + checkpoint exists
#      → restore training log snapshot, then spawn workers (workers clear
#        logs in __init__), then overwrite logs with snapshot content so
#        pre-checkpoint episodes are preserved. Resume training.
#
#   C) training_done=False + no checkpoint
#      → fresh start.

def _make_train_env(model_type, seed):
    """Construct SubprocVecEnv for training."""
    return SubprocVecEnv(
        [_make_train_env_fn(model_type, seed, i) for i in range(N_ENVS)],
        start_method="fork",   # "fork" is faster on Linux; use "spawn" on Windows/macOS
    )

def train_model(model_type, seed, progress):
    entry      = _get_entry(progress, model_type, seed)
    final_path = _final_model_path(model_type, seed)

    # ── A: already done ─────────────────────────────────────────────────────
    if entry["training_done"] and os.path.exists(final_path):
        print(f"[{model_type} | seed {seed}] already complete — loading model")
        # Use DummyVecEnv for the reload-only case: we only need the env
        # to re-attach the policy; no training will occur.
        env = DummyVecEnv([_make_train_env_fn(model_type, seed, 0)])
        model = _load_model(model_type, env, final_path)
        return model, entry["training_time_min"], entry["param_count"]

    ckpt_path, ckpt_steps = _latest_checkpoint(model_type, seed)

    # ── B: resume from checkpoint ────────────────────────────────────────────
    if ckpt_path is not None:
        remaining = TOTAL_TIMESTEPS - ckpt_steps
        print(f"[{model_type} | seed {seed}] resuming from {ckpt_steps:,} steps "
              f"({remaining:,} remaining)")

        # Spawn workers first — their __init__ clears the log files.
        env = _make_train_env(model_type, seed)

        # Now restore snapshot content over the cleared files.
        # This must happen after env construction and before model.learn().
        _restore_train_log_snapshot(model_type, seed, ckpt_steps, env)

        model = _load_model(model_type, env, ckpt_path)
        model.num_timesteps = ckpt_steps
        reset_counter = False

    # ── C: fresh start ───────────────────────────────────────────────────────
    else:
        remaining = TOTAL_TIMESTEPS
        print(f"[{model_type} | seed {seed}] fresh start")
        env = _make_train_env(model_type, seed)
        model = _build_fresh_model(model_type, seed, env)
        reset_counter = True

    param_count = count_parameters(model.policy)
    print(f"[{model_type} | seed {seed}] "
          f"params: {param_count:,}  device: {model.device}")

    ckpt_subdir = _ckpt_subdir(model_type, seed)
    os.makedirs(ckpt_subdir, exist_ok=True)

    checkpoint_cb = SnapshotCheckpointCallback(
        model_type=model_type,
        seed=seed,
        save_freq=max(CKPT_FREQ // N_ENVS, 1),
        save_path=ckpt_subdir,
        name_prefix="rl_model",
        verbose=0,
    )

    start = time.time()
    model.learn(
        total_timesteps=remaining,
        callback=checkpoint_cb,
        reset_num_timesteps=reset_counter,
    )
    elapsed_min = round((time.time() - start) / 60.0, 4)

    # Close subprocess workers cleanly before saving
    env.close()

    model.save(final_path)
    print(f"[{model_type} | seed {seed}] "
          f"done in {elapsed_min:.2f} min — saved to {final_path}")

    prior_min = entry["training_time_min"] or 0.0
    total_min = round(prior_min + elapsed_min, 4)

    entry["training_done"]     = True
    entry["training_time_min"] = total_min
    entry["param_count"]       = param_count
    _save_progress(progress)

    return model, total_min, param_count


# ─────────────────────────────────────────────
# EVALUATION  (with resume)
# ─────────────────────────────────────────────
# DummyVecEnv for evaluation — single env, no subprocess overhead needed.
# All-or-nothing per scenario: cost JSON written only after all episodes
# complete, so a crash leaves no partial data in the stats pipeline.

def evaluate_model(model, model_type, seed, scenario_name, params, progress):
    entry = _get_entry(progress, model_type, seed)

    if scenario_name in entry["eval_done"]:
        print(f"[{model_type} | seed {seed} | {scenario_name}] already done — skipping")
        return

    def make_eval_env(p=params, s=seed):
        return network_sim.SimulatedNetworkEnv(
            history_len=MODELS[model_type]["history_len"],
            bandwidth=p["bandwidth"],
            latency=p["latency"],
            step_bandwidth=p.get("step_change", None),
            scenario_name=scenario_name,
            model_type=model_type,
            seed_id=s,
            phase="eval",
        )

    env       = DummyVecEnv([make_eval_env])
    inner_env = env.envs[0]

    inference_log = []
    obs = env.reset()

    for ep in range(NUM_EVAL_EPISODES):
        done = False
        ep_step_times = []

        while not done:
            t0 = time.time()
            action, _ = model.predict(obs, deterministic=True)
            ep_step_times.append(time.time() - t0)
            obs, reward, done, info = env.step(action)

        inference_log.append({
            "episode":     ep + 1,
            "mean_ms":     round(float(np.mean(ep_step_times)) * 1000, 4),
            "total_steps": len(ep_step_times),
        })
        print(f"[{model_type} | seed {seed} | {scenario_name}] episode {ep + 1} done")

    inner_env.seal_final_episode()
    env.close()

    # Write cost file atomically — all episodes or nothing
    timefile = os.path.join(COST_DIR, f"cost_{model_type}_{scenario_name}_seed{seed}.json")
    tmp = timefile + ".tmp"
    with open(tmp, "w") as f:
        json.dump(inference_log, f, indent=2)
    os.replace(tmp, timefile)

    entry["eval_done"].append(scenario_name)
    _save_progress(progress)


# ─────────────────────────────────────────────
# AGGREGATED COST FILE HELPERS
# ─────────────────────────────────────────────

def _update_cost_logs(progress):
    """Rebuild training_times.json and param_counts.json from the ledger."""
    training_times = {}
    param_counts   = {}
    for model_type, seeds in progress.items():
        for seed_str, entry in seeds.items():
            if entry["training_done"]:
                training_times.setdefault(model_type, {})[seed_str] = \
                    entry["training_time_min"]
                if entry["param_count"] is not None:
                    param_counts[model_type] = entry["param_count"]

    for path, data in [
        (os.path.join(COST_DIR, "training_times.json"), training_times),
        (os.path.join(COST_DIR, "param_counts.json"),   param_counts),
    ]:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Guard required for SubprocVecEnv on some platforms — prevents workers
    # from re-executing the main script on import.

    progress = _load_progress()
    print(f"Loaded progress ledger: {PROGRESS_FILE}\n")

    for seed in SEEDS:
        for model_type in MODELS:

            print(f"\n{'='*60}")
            print(f"[{model_type} | seed {seed}]")
            print(f"{'='*60}")

            model, training_time, param_count = train_model(model_type, seed, progress)
            _update_cost_logs(progress)

            _update_cost_logs(progress)

    print("\nAll training runs complete. Run eval_models.py to evaluate saved models.")
