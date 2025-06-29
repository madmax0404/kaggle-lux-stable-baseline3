# Lux AI Season 3 – Reinforcement Learning Neural Net Agent (Kaggle Competition)
https://www.kaggle.com/competitions/lux-ai-season-3

## Overview
This project implements an **experimental reinforcement learning agent** for the **Lux AI Season 3** competition on Kaggle (NeurIPS 2024). Lux AI is a two-player strategy game where each side controls units on a 24x24 grid to collect resources, fight opponents, and capture relics. The goal here was not just to compete, but to **learn about deep reinforcement learning** by building an agent from scratch. The agent was trained using **Proximal Policy Optimization (PPO)** from the Stable Baselines3 (SB3) library, heavily customized to handle the complex observation and action space of the game. This was a learning-focused project: the journey of designing novel neural network architectures and training pipelines was the priority, even though the final model was too large to submit under Kaggle’s hidden file size limits (hence no official leaderboard result).

## Highlights
* **Custom Gym Environment Wrapper**: Developed a custom OpenAI Gym-compatible wrapper for the Lux AI environment to output a rich **dictionary observation space** and handle the multi-agent setup. The observation includes multiple feature planes (e.g. a 24x24 map of terrain and energy, vision masks) and unit state vectors for both teams. This wrapper makes it possible to interface Lux AI with standard RL libraries.
* **Neural Network Architecture Innovations**: Designed and integrated a **custom neural network** for the agent’s policy/value function. A **CNN-based feature extractor** processes spatial 24x24 grid data (e.g. maps of resources/terrain) while other numerical features are flattened and concatenated. This **Multi-input** architecture combines convolutional layers for spatial features with fully-connected layers for non-spatial data, enabling the agent to handle the diverse observation inputs.

```
MultiInputActorCriticPolicy(
  (features_extractor): CustomFeatureExtractor(
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
  (pi_features_extractor): CustomFeatureExtractor(
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
  (vf_features_extractor): CustomFeatureExtractor(
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
<sub>**▲Model Architecture**</sub>

* **Stable Baselines3 Customization**: Extended the SB3 framework by subclassing and modifying its components to use the custom network. For example, a bespoke MultiInputPolicy was implemented to incorporate the CNN extractor and an enhanced MLP backbone. We integrated these components into SB3’s PPO training loop, effectively **injecting our custom model into the SB3 pipeline** while reusing stable training algorithms (e.g. advantage estimation, optimization routines). This demonstrates deep understanding of the RL library’s internals and how to extend them.
* **Multi-Discrete Action Handling**: The Lux AI game requires selecting actions for up to 16 units simultaneously, with each action composed of multiple parts (e.g. action type and target coordinates). We implemented a **custom action distribution** to handle this multi-discrete action space. The policy’s forward pass outputs a structured set of logits which are then sampled into per-unit actions (including conditional sub-actions for targeting). This involved building a tailored output layer and sampling procedure (using PyTorch) to ensure the agent can issue commands to all units each time-step.
* **Self-Play Training Setup**: To train in a two-player environment, the agent was configured for **self-play**. The training pipeline can initialize two instances of the policy (one for each team) so that the agent competes against a clone of itself or a previous version. This approach is crucial for multi-agent learning, and it required managing two networks and alternating their roles during experience collection. Self-play ensures the agent improves even in the absence of a human-defined opponent, by continuously adapting to its own strategies.

## Training Process

**Reinforcement Learning Setup**: We trained the agent using **PPO (Proximal Policy Optimization)**, an on-policy RL algorithm, with Stable Baselines3 as the foundation. The environment was set up with multiple parallel instances (using SB3’s VecEnv) to collect experience faster. At each training iteration, the agent (as player 0) played games against an opponent (which in self-play was a copy of the same agent or a past checkpoint). The **reward function** is the game’s scoring mechanism (points for collecting relics and winning the match), which the agent tries to maximize over the course of an episode.

**Neural Network & Policy**: The agent’s policy network receives a **multi-modal observation** and outputs actions for all units. Under the hood, our CustomFeatureExtractor first processes the observation dict: four 24×24 feature maps (e.g. explored terrain, energy on the map, etc.) go through convolutional layers, while all other features (unit counts, coordinates, team scores, etc.) are flattened. These are concatenated into one large feature vector. Next, a deep fully-connected **MLP** (with SiLU activations and layer normalization) maps this vector into latent features. PPO then splits this latent representation into two heads: one for the **policy** (action probabilities) and one for the **value function** (state-value estimation). The policy head in our case is custom-built to produce the multi-discrete action output for 16 units. Rather than treating each of the many discrete action components independently, the network outputs structured logits that are reshaped into the appropriate per-unit action distributions. We sample an action for each unit from these distributions to form the complete MultiDiscrete action. This design allowed implementing conditional actions (for example, only units that choose the “move” or “attack” action need a direction – our network outputs direction logits that are only used if the base action is of that type).

**PPO Training Loop**: With the environment and policy defined, training proceeded in iterations. Each iteration, the agent plays for a number of time-steps (collecting observations, actions, and rewards). This experience is stored in SB3’s rollout buffer. After each rollout, we perform several epochs of PPO update: the policy’s parameters are updated via gradient descent to improve the probability of rewarding actions (while avoiding too large a change per PPO’s clipped objective). We made use of **TensorBoard logging** to track key metrics like average reward, policy loss, value loss, etc., which helped in debugging and hyperparameter tuning. We experimented with hyperparameters such as learning rate (e.g. 1e-5 to 6e-4), entropy bonus (to encourage exploration), and network sizes to stabilize training. Training was computationally intensive – the final model has on the order of tens of millions of parameters – so we utilized GPU acceleration and PyTorch 2.0 optimizations (even compiling parts of the model for speed).

**Self-Play and Curriculum**: Early in development, we trained the agent against a fixed script (the provided Kaggle starter agent) to give it basic skills. As it improved, we moved to self-play: the agent would play against a copy of itself. We introduced a mechanism to occasionally update the opponent to the latest policy (or keep a pool of past versions) to ensure the learning agent always has a challenging adversary. This self-play approach is critical in competitive games to avoid the agent overfitting to a static strategy. In our implementation, we leveraged the custom dual-policy setup – the training code would alternate between using policy and policy_2 for the two players each episode, and updates were applied to the main policy network. This helped the agent gradually learn strategies that work well against different instances of itself.

## Results and Lessons Learned

**Training Outcome**: Although an official competition submission wasn’t achieved, the agent showed clear learning progress. Over many training iterations, the average rewards and win-rates (against baseline bots) improved significantly, indicating the agent was picking up useful strategies. For instance, the training curves showed the agent learning to efficiently gather energy and contest relics as training went on (see the **learning curve plot below**). Qualitatively, when we watched game replays of the trained agent, it demonstrated sensible behavior like grouping units for battles and securing resources on the map. These are encouraging signs that the complex neural network and training setup were successful in teaching the agent non-trivial tactics.

**File Size Challenge**: One unexpected hurdle was the **model size constraint** on Kaggle. Our final policy network – due to the large CNN + MLP architecture needed for the task – resulted in a model file well over 100MB. Unfortunately, the competition’s submission system had a limit (approximately 100MB per submission) that wasn’t clearly advertised in the rules. This meant our agent couldn’t be submitted for official evaluation without significant downsizing or simplification. Compressing or pruning the model further would likely hurt its performance, so we made the difficult decision to accept that this was primarily a research endeavor. While we didn’t get a public leaderboard ranking, this outcome highlighted the trade-off between model complexity and deployability. It’s a valuable lesson in considering practical constraints (like file size or runtime) in addition to raw performance. In a real-world setting or future competitions, we would incorporate such constraints into the design from the start (e.g. using smaller networks or model compression techniques).

**Technical Takeaways**: This project was a deep dive into reinforcement learning engineering. We gained experience in:
* Modifying and extending a popular RL framework (SB3) to fit a new problem – from custom policy definition to low-level algorithm tweaks.
* Handling **multi-agent RL** via self-play, and the challenges of stability and diversity it brings.
* Designing neural network architectures that merge different input modalities (spatial and tabular data) and output structured actions.
* Debugging training of a complex model: we often had to investigate why learning stalled, which involved analyzing reward scaling, gradient norms, and occasionally instrumenting the code with printouts of distributions or implementing curriculum adjustments. This mirrors real-world RL development, which requires equal parts intuition and empirical tuning.

Despite the lack of a competition medal, the **primary goal of learning was absolutely met**. The project showcases the ability to **take initiative** and build a complex AI agent from the ground up, navigate through research challenges, and adapt open-source tools beyond their original scope. This kind of experimental project is invaluable preparation for tackling practical machine learning problems in a professional environment.

## Training Curves (Placeholder)
Below is a placeholder for training performance graphs (e.g. average reward per episode over time, loss curves, etc.). These charts illustrate the agent’s learning progress. In a complete README, we would include screenshots or plots of the training curves here for visualization.

![Training Metrics Example](<images/Screenshot from 2025-03-09 23-57-56.png>)
<sub>**▲Training Metrics Example**</sub>

## Project Structure
kaggle-lux-stable-baseline3/
├── GreedyLRScheduler/             # GreedyLR implementation
├── Notebooks/                     # 주피터 노트북 파일들
│   ├── Agent_Development/         # 에이전트 개발 및 실험
│   └── EDA/                       # 탐색적 데이터 분석
├── images/
└── modified_packages/             # 수정된 패키지들
    ├── luxai_s3/                  # 대회용 게임 environment 패키지
    └── stable_baseline3/          # 강화학습용 패키지
