# Temporal State Representation in PPO-Based Congestion Control

## Overview

This repository contains the source code, trained models, experimental logs, and analysis scripts accompanying the study:

> **Temporal State Representation in PPO-Based Congestion Control**

The project investigates how different temporal state representation mechanisms influence the performance of reinforcement learning (RL) congestion-control policies. Specifically, it compares finite-horizon observation stacking with recurrent Long Short-Term Memory (LSTM) representations within a Proximal Policy Optimization (PPO) framework.

The study evaluates the effect of temporal representation on:

* Throughput
* Latency
* Packet loss
* Policy stability
* Training convergence
* Inference cost

while maintaining identical PPO algorithms, reward functions, training budgets, and network environments.

---

## Research Objective

Most prior RL-based congestion-control studies focus on reward design and algorithm selection. In contrast, this work investigates whether the mechanism used to encode temporal information influences congestion-control behavior.

Two temporal representation strategies are compared:

* **Finite-Horizon Observation Stacking**

  * Stacking-3 (K = 3)
  * Stacking-5 (K = 5)
  * Stacking-10 (K = 10)

* **Recurrent Memory**

  * LSTM (Hidden Size = 32)

The study isolates temporal state representation as the sole architectural variable while keeping all other training and evaluation settings fixed.

---

## Repository Structure

```text
Temporal_State_in_PPO_CC/

├── src/gym/
|   |--logs/models/
|   |    |-- model1.zip
|   |    |....
│   ├── Train.py
│   ├── eval_models.py
│   ├── network_sim.py
│   └── ...
│
├── analysis/
│   ├── inference_analysis.py
│   ├── stat_tests.py
│   ├── training_curve.py
│   ├── run_analysis.py
│   └── ...
│
├── logs/
│   ├── train/
│   ├── eval/
│   |── cost_files/
|   |-- models/
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Experimental Setup

### Models Evaluated

| Model       | Temporal Representation             |
| ----------- | ----------------------------------- |
| Stacking-3  | Observation Stacking (K = 3)        |
| Stacking-5  | Observation Stacking (K = 5)        |
| Stacking-10 | Observation Stacking (K = 10)       |
| LSTM        | Recurrent Memory (Hidden Size = 32) |

### Training Configuration

* PPO algorithm
* Training budget: 5,000,000 timesteps
* Five random seeds: (2,7,13,18,24)
* Four parallel simulation workers
* PyTorch backend
* Gymnasium environment interface
* Stable-Baselines3 implementation

### Training Environment

| Parameter         | Value      |
| ----------------- | ---------- |
| Bandwidth         | 200 Mbps   |
| Propagation Delay | 30 ms      |
| Queue Size        | 5 × BDP    |
| Packet Size       | 1500 Bytes |
| Background Loss   | 0          |

---

## Evaluation Scenarios

The trained policies are evaluated under three network conditions:

### Flat

Training distribution:

* Bandwidth = 200 Mbps
* RTT = 30 ms

### Cross-RTT

Tests sensitivity to increased latency:

* RTT = 80 ms

### Step

Tests adaptation to bandwidth variation:

* Bandwidth changes from 200 Mbps to 100 Mbps

---

## Performance Metrics

The following metrics are used throughout the study:

* Throughput (Mbps)
* Latency (s)
* Packet Loss Rate
* Episode Reward
* Send Rate Variability
* Inference Latency

Additional analyses include:

* Training convergence
* Throughput–Latency tradeoff
* Throughput–Loss tradeoff
* Inference latency distributions
* Step-level inference drift

---

## Installation

Clone the repository:

```bash
git clone https://github.com/mbrk-tofa/Temporal_State_in_PPO_CC.git
cd Temporal_State_in_PPO_CC
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

### Linux / macOS

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training

Train a model using:

```bash
python gym/Train.py
```

Model-specific configurations can be adjusted within the corresponding configuration files.

---

## Evaluation

Evaluate trained policies using:

```bash
python gym/eval_models.py
```

Evaluation outputs are stored in the evaluation log directory.

---

## Analysis and Figure Generation

The analysis scripts reproduce all statistical results, figures, and tables reported in the manuscript.

### Training Convergence

```bash
python analysis/train_conv.py
```

### Inference Cost Analysis

```bash
python analysis/inference_analysis.py
```

### Generate Other Publication Figures and Statistical Tests

```bash
python analysis/run_analysis.py
```
All generated files will be stored in Temporal_State_in_PPO_CC/results
---

## Reproducibility

To facilitate reproducibility:

* Fixed random seeds are used.
* PPO hyperparameters are identical across all models.
* Training environments are controlled.
* Evaluation protocols are standardized.
* Analysis scripts are included.
* Experimental logs are provided.

All reported figures and statistical analyses can be regenerated from the repository contents.

---

## Data Availability

The source code, experimental configurations, training scripts, evaluation scripts, trained models, training and evaluation logs, and analysis scripts supporting the findings of the associated publication are publicly available in this repository.

---

## Generative AI Disclosure

Generative AI tools were used for code development, code-review, assistance during manuscript preparation for draft review and language refinement, and research discussions. All experimental design decisions, code implementation, model training, data collection, statistical analyses, and scientific conclusions were performed and verified by the author.

---

## Citation

If you use this repository in your research, please cite:

```bibtex
@article{.......,
  title={Temporal State Representation in PPO-Based Congestion Control},
  author={Bello, Mubarak Abubakar Bello},
  journal={.....},
  year={2026}
}
```

(to be Updated after publication.)

---

## License

This project is released under the MIT License.

See the [LICENSE](LICENSE) file for details.
