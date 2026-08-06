"""run_eval_only.py

Re-runs evaluation for all (model_type, seed, scenario) combinations using
the final trained model zips saved by Train5.py. Produces:

  logs/eval/<model_type>_<scenario>_seed_<seed>.jsonl
      Full per-step JSONL logs — consumed by the CDF inference-time plotter.

  logs/cost_files/cost_<model_type>_<scenario>_seed<seed>.json
      Per-episode inference timing — consumed by the existing cost analysis pipeline.

Design constraints
------------------
- No training, no checkpointing, no ledger writes.
- progress.json is never read or modified — this script is stateless.
- All-or-nothing per (model_type, seed, scenario): cost JSON and eval JSONL
  are written only after all NUM_EVAL_EPISODES complete. A crash mid-evaluation
  leaves no partial files, so rerunning the script is always safe.
- Existing eval files are overwritten unconditionally — this is a clean re-run.
- Missing model zips are reported and skipped; the script continues with the
  remaining combinations rather than aborting.

Usage
-----
  python run_eval_only.py

  # Re-run only specific models or seeds:
  # Edit MODELS or SEEDS at the top of this file before running.
"""

import gymnasium as gym
import network_sim
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv

import torch
import numpy as np
import json
import time
import os
from pathlib import Path

# ─────────────────────────────────────────────
# HARDWARE  (inference only — GPU not critical
# but used if available for consistency)
# ─────────────────────────────────────────────

torch.set_num_threads(1)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"{'='*60}")
print(f"Eval-only run")
print(f"  Device:          {DEVICE}")
print(f"  CPU cores:       {os.cpu_count()}")
print(f"{'='*60}\n")


# ─────────────────────────────────────────────
# CONFIGURATION  (must match Train5.py exactly)
# ─────────────────────────────────────────────

NUM_EVAL_EPISODES = 10

SEEDS = [2, 7, 13, 18, 24]

MODELS = {
    "stacking3HL":  {"policy": "MlpPolicy",     "history_len": 3},
    "stacking5HL":  {"policy": "MlpPolicy",     "history_len": 5},
    "stacking10HL": {"policy": "MlpPolicy",     "history_len": 10},
    "lstm":         {"policy": "MlpLstmPolicy", "history_len": 1},
}

SCENARIOS = {
    "crossrtt": {"bandwidth": 200, "latency": 0.08},
    "flat":     {"bandwidth": 200, "latency": 0.03},
    "step":     {"bandwidth": 200, "latency": 0.03, "step_change": 100},
}


# ─────────────────────────────────────────────
# PATHS  (must match Train5.py directory layout)
# ─────────────────────────────────────────────
# All paths are anchored to the directory containing this script,
# not the working directory from which Python is invoked.
# This means the script works correctly regardless of where you run it from:
#   python src/gym/eval_models.py                 OK
#   cd src/gym && python eval_models.py           OK

REPO_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = REPO_ROOT / "logs"
MODELS_DIR = LOGS_DIR / "models"
# Archived model zips used the pre-standardisation location. The fallback
# keeps them usable while all newly trained models use logs/models/.
LEGACY_MODELS_DIR = Path(__file__).resolve().parent / "logs" / "models"
COST_DIR = LOGS_DIR / "cost_files"
EVAL_DIR = LOGS_DIR / "eval"

for d in [COST_DIR, EVAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────

def _final_model_path(model_type, seed):
    canonical = MODELS_DIR / f"{model_type}_seed{seed}.zip"
    if canonical.exists():
        return str(canonical)
    return str(LEGACY_MODELS_DIR / f"{model_type}_seed{seed}.zip")

def _load_model(model_type, seed):
    """Load a final trained model zip and attach a minimal dummy env.

    A single-env DummyVecEnv is used purely to satisfy SB3's requirement
    that a model has an env attached — it is never stepped during loading.
    """
    zip_path = _final_model_path(model_type, seed)
    if not os.path.exists(zip_path):
        return None

    # Minimal env just for model attachment — history_len must match training
    def _dummy():
        return gym.make(
            "PccNs-v0",
            history_len=MODELS[model_type]["history_len"],
            bandwidth=200,
            latency=0.03,
            queue=5,
            loss=0.0,
            model_type=model_type,
            seed_id=0,
            phase="train",    # suppress logging — this env is never stepped
        )

    env = DummyVecEnv([_dummy])

    if model_type == "lstm":
        model = RecurrentPPO.load(zip_path, env=env, device=DEVICE)
    else:
        model = PPO.load(zip_path, env=env, device=DEVICE)

    env.close()
    return model


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def run_evaluation(model, model_type, seed, scenario_name, params):
    """Evaluate one (model_type, seed, scenario) triple.

    Writes two output files atomically after all episodes complete:
      - logs/eval/<model_type>_<scenario>_seed_<seed>.jsonl  (per-step, for CDF)
      - logs/cost_files/cost_<model_type>_<scenario>_seed<seed>.json (timing)

    Both are overwritten unconditionally — this is a clean re-run.
    A crash before both writes leaves no partial output files (tmp + replace).
    """

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

    # inference_log stores every individual step timing across all episodes.
    # Structure: list of {"episode": int, "step": int, "inference_ms": float}
    # This gives the CDF plotter the full distribution of per-step times,
    # not just per-episode means which would lose the tail behaviour.
    inference_log = []
    obs = env.reset()

    for ep in range(NUM_EVAL_EPISODES):
        done   = False
        step_n = 0

        while not done:
            t0 = time.time()
            action, _ = model.predict(obs, deterministic=True)
            elapsed_ms = round((time.time() - t0) * 1000, 4)

            inference_log.append({
                "episode":      ep + 1,
                "step":         step_n,
                "inference_ms": elapsed_ms,
            })
            step_n += 1

            obs, reward, done, info = env.step(action)

        ep_times = [r["inference_ms"] for r in inference_log
                    if r["episode"] == ep + 1]
        print(f"  episode {ep + 1}/{NUM_EVAL_EPISODES} done  "
              f"steps={len(ep_times)}  "
              f"mean={float(np.mean(ep_times)):.4f} ms  "
              f"p99={float(np.percentile(ep_times, 99)):.4f} ms")

    inner_env.seal_final_episode()
    env.close()

    # Write cost file atomically — all-or-nothing
    # Contains every per-step inference time across all episodes.
    cost_path = os.path.join(
        COST_DIR, f"cost_{model_type}_{scenario_name}_seed{seed}.json"
    )
    tmp = cost_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(inference_log, f, indent=2)
    os.replace(tmp, cost_path)

    total_steps = len(inference_log)
    print(f"  cost file written: {cost_path}  ({total_steps} step records)")


# ─────────────────────────────────────────────
# DIRECTORY → ZIP CONVERSION
# ─────────────────────────────────────────────
# When Kaggle dataset transfer unpacks zip files into directories, SB3's
# load() cannot read them directly (raises IsADirectoryError). This function
# detects any directory-format models and repacks them into zip files in-place
# before the pre-flight check runs. Safe to call multiple times — skips any
# model that is already a zip file.

def _repack_directory_models():
    """Convert any directory-format models to zip files SB3 can load."""
    import zipfile

    repacked = []
    for model_type in MODELS:
        for seed in SEEDS:
            zip_path = _final_model_path(model_type, seed)   # e.g. .../lstm_seed2.zip
            dir_path = zip_path[:-4]                          # strip .zip → directory name

            if os.path.isdir(dir_path) and not os.path.exists(zip_path):
                # Kaggle unpacked the zip into a directory — repack it
                print(f"  Repacking {os.path.basename(dir_path)}/ → .zip ...")
                with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
                    for fname in os.listdir(dir_path):
                        zf.write(os.path.join(dir_path, fname), arcname=fname)
                repacked.append(zip_path)

            elif os.path.exists(zip_path):
                # Already a zip — nothing to do
                pass

    if repacked:
        print(f"  Repacked {len(repacked)} model(s) to zip format.\n")
    else:
        print(f"  All models already in zip format.\n")


# ─────────────────────────────────────────────
# PRE-FLIGHT CHECK
# ─────────────────────────────────────────────

def check_models():
    """Verify all expected model zips are present before starting.

    Reports missing files and returns the set of (model_type, seed) pairs
    that are available. The main loop skips missing pairs rather than aborting.
    """
    print("Checking model zip availability...")
    available = []
    missing   = []

    for model_type in MODELS:
        for seed in SEEDS:
            path = _final_model_path(model_type, seed)
            if os.path.exists(path):
                # Model may be a directory (SB3 directory save) or a zip file.
                if os.path.isdir(path):
                    n_files = len(os.listdir(path))
                    desc = f"dir, {n_files} file(s)"
                else:
                    size_mb = os.path.getsize(path) / (1024 * 1024)
                    desc = f"{size_mb:.2f} MB"
                available.append((model_type, seed))
                print(f"  OK   {model_type:15s} seed={seed}  ({desc})")
            else:
                missing.append((model_type, seed))
                print(f"  MISS {model_type:15s} seed={seed}  — {path}")

    print(f"\n{len(available)}/{ len(MODELS) * len(SEEDS)} model zips found.")
    if missing:
        print(f"Skipping {len(missing)} missing pairs: "
              + ", ".join(f"{m}|seed{s}" for m, s in missing))
    print()
    return set(available)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("Checking for directory-format models (Kaggle dataset unpack)...")
    _repack_directory_models()
    available_pairs = check_models()

    total    = len(available_pairs) * len(SCENARIOS)
    done_count = 0
    failed   = []

    for seed in SEEDS:
        for model_type in MODELS:

            if (model_type, seed) not in available_pairs:
                continue

            print(f"\n{'='*60}")
            print(f"Loading: {model_type} | seed {seed}")
            print(f"{'='*60}")

            model = _load_model(model_type, seed)
            if model is None:
                print(f"  [error] failed to load model — skipping all scenarios")
                failed.append((model_type, seed))
                continue

            print(f"  Loaded successfully  device={model.device}")

            for scenario_name, params in SCENARIOS.items():
                print(f"\n  Scenario: {scenario_name}")
                try:
                    run_evaluation(model, model_type, seed, scenario_name, params)
                    done_count += 1
                except Exception as e:
                    print(f"  [error] {model_type} | seed {seed} | "
                          f"{scenario_name}: {e}")
                    failed.append((model_type, seed, scenario_name))

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Evaluation complete.")
    print(f"  Successful: {done_count}/{total}")
    if failed:
        print(f"  Failed ({len(failed)}):")
        for item in failed:
            print(f"    {item}")
    print(f"\nOutputs:")
    print(f"  Per-step JSONL (CDF input):  {EVAL_DIR}/")
    print(f"  Per-episode timing (costs):  {COST_DIR}/")
    print(f"{'='*60}")
