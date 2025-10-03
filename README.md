# Lux AI Season 3 — Reinforcement Learning Neural Agent

Kaggle Competition
[https://www.kaggle.com/competitions/lux-ai-season-3](https://www.kaggle.com/competitions/lux-ai-season-3)

---

## Overview

This project is an **experimental reinforcement learning (RL) agent** built for the **Lux AI Season 3** competition (NeurIPS 2024) on Kaggle. Lux AI is a strategy game played on a 24×24 grid where two players control units to gather resources, fight, and capture relics. The goal of this project was not only to compete, but to **learn end-to-end deep RL in a real setting**. Using **PPO (Proximal Policy Optimization)** from Stable-Baselines3 (SB3), I customized the agent to handle the game’s complex observation/action spaces. Although the final model exceeded Kaggle’s submission size limit (~100 MB) and thus wasn’t submitted officially, the core objective—**hands-on design and implementation of deep RL**—was achieved.

---

## Tech Stack

* **Language:** Python
* **DL Framework:** PyTorch
* **Reinforcement Learning:** Stable-Baselines3 — [https://stable-baselines3.readthedocs.io/en/master/](https://stable-baselines3.readthedocs.io/en/master/)
* **Game Environment:** Simulated in the Lux AI Season 3 environment (`luxai_s3` Python package). The environment itself is JAX-based but wrapped for Python, exposing state and reward mechanics.
* **Tools & Dev Environment:** Jupyter Notebook, VS Code; training on an Ubuntu Linux system with GPU acceleration
* **Visualization:** TensorBoard
* **OS:** Linux (Ubuntu Desktop 24.04 LTS)

---

## Key Features

* **Custom Neural Architecture:** Designed a **custom policy/value network**. Spatial 24×24 grid inputs (resources/terrain, etc.) are processed by a **CNN feature extractor**, while numeric features are flattened and **concatenated**—a **multi-input** architecture that effectively fuses spatial and non-spatial signals.

```
MultiInputActorCriticPolicy(
  (features_extractor, pi_features_extractor, vf_features_extractor): CustomFeatureExtractor(
    (cnn_extractor): OptimizedModule(
      (_orig_mod): Sequential(
        (0): Conv2d(4, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): SiLU()
        (3): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (5): SiLU()
        (6): Flatten(start_dim=1, end_dim=-1)
        (7): Dropout(p=0.1, inplace=False)
      )
    )
    (extractors): ModuleDict(
      (enemy_energies): Flatten(start_dim=1, end_dim=-1)
      (enemy_positions): Flatten(start_dim=1, end_dim=-1)
      (enemy_spawn_location): Flatten(start_dim=1, end_dim=-1)
      (enemy_visible_mask): Flatten(start_dim=1, end_dim=-1)
      (map_explored_status): Flatten(start_dim=1, end_dim=-1)
      (map_features_energy): Flatten(start_dim=1, end_dim=-1)
      (map_features_tile_type): Flatten(start_dim=1, end_dim=-1)
      (match_steps): Flatten(start_dim=1, end_dim=-1)
      (my_spawn_location): Flatten(start_dim=1, end_dim=-1)
      (relic_nodes): Flatten(start_dim=1, end_dim=-1)
      (relic_nodes_mask): Flatten(start_dim=1, end_dim=-1)
      (sensor_mask): Flatten(start_dim=1, end_dim=-1)
      (steps): Flatten(start_dim=1, end_dim=-1)
      (team_id): Flatten(start_dim=1, end_dim=-1)
      (team_points): Flatten(start_dim=1, end_dim=-1)
      (team_wins): Flatten(start_dim=1, end_dim=-1)
      (unit_active_mask): Flatten(start_dim=1, end_dim=-1)
      (unit_energies): Flatten(start_dim=1, end_dim=-1)
      (unit_move_cost): Flatten(start_dim=1, end_dim=-1)
      (unit_positions): Flatten(start_dim=1, end_dim=-1)
      (unit_sap_cost): Flatten(start_dim=1, end_dim=-1)
      (unit_sap_range): Flatten(start_dim=1, end_dim=-1)
      (unit_sensor_range): Flatten(start_dim=1, end_dim=-1)
    )
  )
  (mlp_extractor): OptimizedModule(
    (_orig_mod): MlpExtractor(
      (policy_net): Sequential(
        (0): Linear(in_features=20897, out_features=4096, bias=True)
        (1): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        (2): SiLU()
        (3): Dropout(p=0.1, inplace=False)
        (4): Linear(in_features=4096, out_features=2048, bias=True)
        (5): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
        (6): SiLU()
        (7): Dropout(p=0.1, inplace=False)
        (8): Linear(in_features=2048, out_features=1024, bias=True)
        (9): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        (10): SiLU()
        (11): Dropout(p=0.1, inplace=False)
      )
      (value_net): Sequential(
        (0): Linear(in_features=20897, out_features=4096, bias=True)
        (1): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        (2): SiLU()
        (3): Dropout(p=0.1, inplace=False)
        (4): Linear(in_features=4096, out_features=2048, bias=True)
        (5): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
        (6): SiLU()
        (7): Dropout(p=0.1, inplace=False)
        (8): Linear(in_features=2048, out_features=1024, bias=True)
        (9): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        (10): SiLU()
        (11): Dropout(p=0.1, inplace=False)
        (12): Linear(in_features=1024, out_features=512, bias=True)
        (13): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
        (14): SiLU()
        (15): Dropout(p=0.1, inplace=False)
        (16): Linear(in_features=512, out_features=256, bias=True)
        (17): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
        (18): SiLU()
        (19): Dropout(p=0.1, inplace=False)
        (20): Linear(in_features=256, out_features=128, bias=True)
        (21): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (22): SiLU()
        (23): Dropout(p=0.1, inplace=False)
      )
    )
  )
  (action_net): Linear(in_features=1024, out_features=576, bias=True)
  (value_net): Linear(in_features=128, out_features=1, bias=True)
)
```

<sub>**▲ Model Architecture**</sub>

* **Custom Gym Wrapper:** Implemented an OpenAI Gym-compatible wrapper to output a **dictionary observation space** and handle multi-agent settings in the Lux environment. This exposes 24×24 terrain/energy maps, visibility masks, and unit-state vectors in a format that plugs cleanly into RL libraries.
* **Stable-Baselines3 Customization:** Extended SB3 by subclassing/modifying internals to support the custom network and **MultiInputPolicy**. Integrated with the PPO training loop to reuse SB3’s stable advantages/optimization routines while keeping the inner architecture fully customized.
* **Multi-Discrete Action Handling:** Lux AI requires **simultaneous actions for up to 16 units** (composite actions with types and coordinates). Implemented a **custom action distribution** and sampling logic in PyTorch so the policy’s forward pass emits all per-unit actions, including conditional sub-actions (e.g., direction on move/attack).
* **Self-Play Training:** Since the environment is two-player, the agent improves via **self-play**. Trained alternatingly against a cloned self or older checkpoints, laying the groundwork for multi-agent learning.

---

## Training Process

**RL Setup:** The agent was trained with **PPO** (SB3) using multiple parallel environments (VecEnv) for faster experience collection. In each iteration, player-0 fought a copy of itself or an older checkpoint. The **reward function** followed the game’s scoring (relic capture, victory, etc.), which the agent seeks to maximize.

**Network & Policy:** The policy ingests **multimodal observations** and outputs actions for all units. The `CustomFeatureExtractor` processes spatial maps (e.g., four 24×24 channels) via CNN and flattens other features, then fuses them into a single vector. A deep fully-connected MLP (SiLU activations, LayerNorm) produces latent features. PPO splits this latent into **policy** (action probabilities) and **value** (state value). The policy head is customized for the **multi-discrete** action space of up to 16 units, producing structured distributions (with conditional sub-actions) per unit.

**PPO Loop:** After defining environment and policy, training proceeds in rollouts. Trajectories (obs, actions, rewards, etc.) are stored in a buffer, and PPO updates run for several epochs (gradient-based optimization toward higher returns). **TensorBoard** logs average reward, policy loss, value loss, etc., aiding tuning and debugging. Training is compute-intensive; GPU acceleration and PyTorch 2.x optimizations (incl. partial compilation) were used.

**Self-Play & Curriculum:** Started against the provided starter agent to learn fundamentals, then switched to self-play. Periodically swapped the opponent with the latest policy or a prior version to avoid overfitting to a single style. A dual-policy setup (`policy`/`policy_2`) alternated control across episodes, exposing the agent to diverse strategies.

---

## Results & Lessons Learned

**Training Outcomes:** Although not submitted officially, the agent showed **clear learning progress**—average rewards and win rates rose across iterations. Replays reveal sensible, non-trivial strategies such as grouping units for combat and aggressively contesting relics and resources.

**Model Size Constraint:** A major blocker was **Kaggle’s model size limit (~100 MB)**. The CNN+MLP architecture exceeded the limit, preventing submission. Shrinking the model would have hurt performance; thus, the project’s value is framed as **research/engineering practice**. Key takeaway: in real-world settings or competitions, constraints like **size and speed** must be considered from day one.

**Technical Highlights:**

* Tailoring SB3 (policy definitions, internal tweaks) to the problem
* Multi-agent training with self-play and its stability/diversity challenges
* Neural designs for mixed spatial/non-spatial inputs and composite actions
* Practical debugging for complex training (reward scaling, grad-norm control, curriculum tuning)

**Misc:** Implemented a **GreedyLR** scheduler (experimentally) based on “[Zeroth-Order GreedyLR: An Adaptive Learning Rate Scheduler for Deep Neural Network Training](https://www.amazon.science/publications/zeroth-order-greedylr-an-adaptive-learning-rate-scheduler-for-deep-neural-network-training)”, but it wasn’t successful in this setting.

Despite no medal, I consider this a success in the true objective: **real-world RL engineering experience and growth**. Building an agent beyond starter code—through design, trial, and iteration—substantially leveled up my ML engineering skills.

---

## Training Curves (Example)

Below is an example plot of training metrics (average reward, loss curves, etc.).

![Training Metrics Example](<images/Screenshot from 2025-03-09 23-57-56.png>) <sub>**▲ Training Metrics Example**</sub>

---

## Project Structure

```
kaggle-lux-stable-baseline3/
├── GreedyLRScheduler/             # GreedyLR implementation
├── Notebooks/                     # Jupyter notebooks
│   ├── Agent_Development/         # Agent development & experiments
│   └── EDA/                       # Exploratory analysis
├── images/
└── modified_packages/             # Patched packages
    ├── luxai_s3/                  # Competition game environment
    └── stable_baseline3/          # RL framework (modified)
```

---

## How to Reproduce

1. **Clone the repository**

```bash
git clone https://github.com/madmax0404/kaggle-lux-stable-baseline3.git
cd kaggle-lux-stable-baseline3
```

2. **Download the dataset**

* Join the competition: **[NeurIPS 2024 — Lux AI Season 3](https://www.kaggle.com/competitions/lux-ai-season-3)**
* Download the data and place it in the appropriate directory.

3. **Create a virtual environment & install dependencies**

```bash
conda create -n kaggle_lux_stable_baseline3 python=3.12  # or venv
conda activate kaggle_lux_stable_baseline3
pip install -r requirements.txt
```

4. **Run Jupyter Notebook**

```bash
jupyter notebook Notebooks
```

Follow the notebooks to run preprocessing, training, and evaluation.

---

## Acknowledgements

Thanks to the **Lux AI Challenge** and **Kaggle** for the dataset and competition platform.

This project was supported by the following open-source tools: Python, PyTorch, Stable-Baselines3, TensorBoard, pandas, numpy, matplotlib, seaborn, Jupyter, SciPy, Ubuntu.

All data usage complies with the competition rules and licenses.

---

## License

Code © 2025 **Jongyun Han (Max)**. Released under the **MIT License**.
See the LICENSE file for details.

**Note:** Datasets are **NOT** redistributed in this repository.
Please download them from the official Kaggle competition page and comply with the competition rules/EULA.