# Lux AI Season 3 – 강화학습 신경망 에이전트

Kaggle 대회

https://www.kaggle.com/competitions/lux-ai-season-3

---

## 기술 스택
* **언어**: Python
* **딥러닝 프레임워크**: PyTorch
* **강화학습**: Stable Baselines 3 - https://stable-baselines3.readthedocs.io/en/master/
* **게임 환경**: Lux AI Season 3 게임 환경(luxai_s3 Python 패키지)에서 시뮬레이션 진행. 환경 자체는 JAX 기반이지만 Python에서 사용 가능하도록 래핑되어 있어 게임의 상태와 보상 메커니즘을 제공함.
* **툴 및 개발 환경**: Jupyter Notebook과 VS Code를 활용하여 개발 및 실험을 진행. 훈련은 GPU 가속을 지원하는 Ubuntu Linux 시스템에서 수행.
* **시각화**: TensorBoard
* **운영체제**: Linux (Ubuntu Desktop 24.04 LTS)

---

## 개요
이 프로젝트는 Kaggle(NeurIPS 2024)에서 진행된 **Lux AI Season 3** 대회를 위한 **실험적 강화학습 에이전트** 구현 사례입니다. Lux AI는 24x24 격자 맵 위에서 두 플레이어가 유닛을 조종해 자원을 수집하고, 상대와 전투하며, 유물을 점령하는 전략 게임입니다. 본 프로젝트의 목표는 단순히 대회 참가에 그치지 않고, **딥러닝 기반 강화학습의 전반을 실전에서 경험하고 배우는 것**이었습니다. Stable Baselines3(SB3)의 **PPO(Proximal Policy Optimization)** 알고리즘을 바탕으로 게임의 복잡한 관측/행동 공간을 다룰 수 있도록 에이전트를 직접 커스터마이즈했습니다. 이 과정 자체가 학습의 핵심이었으며, 최종적으로 생성된 모델이 Kaggle 제출 파일 용량 제한(약 100MB)을 초과해 공식 제출은 못 했지만, 실제 RL 설계와 구현을 깊이 경험할 수 있었습니다.

---

## 주요 특징
* **커스텀 신경망 아키텍처**: 에이전트의 정책/가치 함수에 맞게 **커스텀 신경망**을 설계·적용했습니다. 24x24 격자 데이터(자원/지형 등)는 **CNN 기반 feature extractor**로 처리하고, 그 외 수치 피처들은 펼쳐서(flatten) 함께 연결(concat)하는 **멀티-입력(Multi-input) 아키텍처**를 도입했습니다. 이를 통해 공간적 데이터와 비공간적 데이터를 모두 효과적으로 처리할 수 있었습니다.

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
<sub>**▲Model Architecture**</sub>

* **커스텀 Gym 환경 래퍼**: OpenAI Gym 호환 래퍼를 직접 구현해, Lux AI 환경의 다양한 관측값을 **딕셔너리 기반 관측 공간**으로 출력하고, 멀티 에이전트 설정을 처리했습니다. 24x24 지형/에너지 맵, 시야 마스크, 각 유닛의 상태 벡터 등 다양한 정보를 RL 라이브러리와 쉽게 연동 가능하게 만들었습니다.
* **Stable Baselines 3 커스터마이징**: SB3의 구조를 상속/수정하여 커스텀 네트워크와 멀티 입력 정책(MultiInputPolicy)이 동작하도록 프레임워크를 확장했습니다. PPO 학습 루프에 맞게 모델을 통합해 SB3의 안정적인 학습 알고리즘(Advantage estimation, 최적화 루틴 등)을 재사용하되, 내부 구조는 완전히 커스터마이즈했습니다.
* **멀티-디스크리트 액션 처리**: Lux AI는 최대 16개 유닛에 대해 동시 행동(행동 종류 및 좌표 등 복합형 액션)을 요구하므로, 이를 위해 **커스텀 액션 분포**와 출력/샘플링 로직을 별도로 구현했습니다. PyTorch로 구현된 정책의 forward pass가 유닛별 액션을 모두 산출할 수 있도록 설계했으며, 조건부 하위 액션(예: 이동·공격시 방향 지정)도 동적으로 처리했습니다.
* **Self-Play(자가 대전) 훈련**: 두 명의 플레이어가 대전하는 환경 특성상, 에이전트가 **자가 대전(self-play)**을 통해 스스로와 대결하며 발전하도록 파이프라인을 구성했습니다. 두 개의 정책을 번갈아 학습시키거나, 이전 버전과 대결하게 하여 멀티에이전트 학습의 기반을 마련했습니다.

---

## 학습 과정

**강화학습 세팅**: 에이전트는 SB3 기반 **PPO**로 훈련되었고, 빠른 경험 수집을 위해 여러 환경 인스턴스(VecEnv)를 병렬로 사용했습니다. 각 학습 반복에서 에이전트(player 0)는 복제된 자신 또는 과거 체크포인트와 대전했습니다. **보상 함수**는 게임 내 점수(유물 수집 및 승리 등)를 기반으로 하여, 최종적으로 최대화하는 것이 목표입니다.

**신경망 및 정책**: 정책 네트워크는 **멀티모달 관측값**을 받아 모든 유닛의 행동을 출력합니다. CustomFeatureExtractor가 관측값(24×24 맵 4장 등)을 CNN으로, 그 외 피처는 펼쳐서 하나의 벡터로 만든 후, 깊은 fully-connected MLP(활성화 함수 SiLU, LayerNorm 적용)에서 잠재 피처로 변환합니다. PPO는 이 잠재벡터를 정책(행동 확률)과 가치함수(상태가치)로 분리해 처리합니다. 특히 정책 부분은 16유닛 멀티-디스크리트 액션에 맞춰 커스텀 구조로 설계해, 각 유닛별로 행동·타겟 등을 구조화된 확률분포로 산출합니다. 조건부 행동(예: 이동/공격시 방향 지정 등)도 네트워크에서 유연하게 지원합니다.

**PPO 학습 루프**: 환경과 정책이 정의된 후, 학습은 반복적으로 진행됩니다. 일정 시간(step)마다 경험(관측, 행동, 보상 등)을 rollout buffer에 저장하고, 여러 에포크에 걸쳐 PPO 업데이트를 실시합니다(정책 파라미터를 gradient descent로 보상 극대화 방향으로 조정). **TensorBoard**로 평균 보상, 정책 손실, 가치 손실 등 주요 지표를 로깅해 하이퍼파라미터 튜닝 및 디버깅에 활용했습니다. 학습은 매우 고연산을 요구해 GPU 및 PyTorch 2.0 최적화(일부 모델 부분 컴파일 등)를 적극적으로 사용했습니다.

**Self-Play와 커리큘럼**: 초기에는 제공된 starter agent(스크립트)를 상대해 기본기를 익혔고, 이후 self-play 체제로 전환했습니다. 일정 주기로 상대를 최신 정책 또는 이전 버전으로 교체해 주며, 단일 전략에 과적합되는 것을 방지했습니다. dual-policy 구조로 두 플레이어를 각각 policy/policy_2로 할당하고, 에피소드마다 교대로 업데이트함으로써 다양한 전략을 학습할 수 있었습니다.

---

## 결과 및 배운 점

**훈련 결과**: 공식 대회 제출은 못 했지만, 에이전트가 학습을 통해 실질적으로 발전하는 모습을 확인할 수 있었습니다. 여러 반복에서 평균 보상 및 승률이 눈에 띄게 향상됐으며, 학습 커브에서도 점차적으로 에너지를 효율적으로 수집하거나 유물을 적극적으로 쟁취하는 등 전략적 행동을 익히는 과정이 보였습니다. 트레이닝된 에이전트의 리플레이를 보면, 유닛들을 모아 전투를 벌이거나 자원을 차지하는 등 합리적이고 비트리비얼한 전략을 구사하는 것이 관찰됐습니다.

**모델 용량 이슈**: 예상치 못한 문제 중 하나는 **Kaggle 제출 파일 크기 제한**이었습니다. CNN + MLP 기반 대규모 네트워크 덕분에 모델 파일이 100MB를 훌쩍 넘어서게 되었고, 이로 인해 공식 평가를 받지 못했습니다. 모델 크기를 더 줄이면 성능 저하가 불가피해, 이번 프로젝트는 연구 및 실험에 의미를 두는 것으로 방향을 정했습니다. 이 경험을 통해 실제 현업이나 미래 대회에서는 성능뿐만 아니라 실질적 제약(용량, 속도 등)까지 반드시 설계 초기부터 고려해야 함을 깨달았습니다.

**기술적 요점**: 이 프로젝트를 통해 강화학습 엔지니어링을 심도 있게 경험했습니다.
* RL 프레임워크(SB3)를 상황에 맞게 커스터마이즈(정책 정의, 내부 알고리즘 수정 등)
* self-play를 활용한 멀티에이전트 학습과 그 난이도 및 다양성 확보 문제
* 다양한 입력(공간/비공간)과 복합형 액션을 다루는 신경망 구조 설계
* 복잡한 모델 학습에서 reward scaling, gradient norm, 커리큘럼 조정 등 디버깅 노하우 축적

**기타**: GreedyLR 스케쥴러를 구현해 사용해 보았지만 성공적이진 않았습니다. (https://www.amazon.science/publications/zeroth-order-greedylr-an-adaptive-learning-rate-scheduler-for-deep-neural-network-training) 

비록 메달은 얻지 못했지만, **진짜 목표였던 실전 강화학습 경험 및 성장**이라는 측면에서는 최고의 성과를 거뒀다고 자부합니다. 오픈소스 도구의 한계를 넘어, 직접 AI 에이전트를 설계하고 시행착오를 겪으며 학습한 이 과정 자체가 머신러닝 실무 역량을 크게 끌어올려 준 소중한 경험이었습니다.

---

## 학습 곡선 예시
아래는 학습 성과를 시각화한 그래프 예시(평균 보상, 손실 곡선 등)입니다.

![Training Metrics Example](<images/Screenshot from 2025-03-09 23-57-56.png>)
<sub>**▲Training Metrics Example**</sub>

---

## 프로젝트 구조
    kaggle-lux-stable-baseline3/
    ├── GreedyLRScheduler/             # GreedyLR 구현
    ├── Notebooks/                     # 주피터 노트북 파일들
    │   ├── Agent_Development/         # 에이전트 개발 및 실험
    │   └── EDA/                       # 탐색적 데이터 분석
    ├── images/
    └── modified_packages/             # 수정된 패키지들
        ├── luxai_s3/                  # 대회용 게임 환경 패키지
        └── stable_baseline3/          # 강화학습용 패키지

---

## 프로젝트 실행 방법

분석 및 모델 훈련 재현 방법:

1.  **Repository 복제:**
    ```bash
    git clone [https://github.com/madmax0404/kaggle-lux-stable-baseline3.git](https://github.com/madmax0404/kaggle-lux-stable-baseline3.git)
    cd kaggle-lux-stable-baseline3
    ```
2.  **데이터셋 다운로드:**
    * 캐글에서 대회에 참가하세요. [NeurIPS 2024 - Lux AI Season 3](https://www.kaggle.com/competitions/lux-ai-season-3)
    * 데이터를 다운받은 후 알맞은 디렉토리에 저장하세요.
3.  **가상 환경을 생성하고 필요한 라이브러리들을 설치해주세요:**
    ```bash
    conda create -n kaggle_lux_stable_baseline3 python=3.12 # or venv
    conda activate kaggle_lux_stable_baseline3
    pip install -r requirements.txt
    ```
4.  **Jupyter Notebook을 실행해주세요:**
    ```bash
    jupyter notebook Notebooks
    ```
    데이터 처리, 모델 훈련 및 평가를 실행하려면 노트북의 단계를 따르세요.

---

## Acknowledgements

데이터셋과 대회 플랫폼을 제공한 Lux AI Challenge와 Kaggle에 감사드립니다.

본 프로젝트는 다음 오픈소스의 도움을 받았습니다: Python, PyTorch, Stable Baselines 3, TensorBoard, pandas, numpy, matplotlib, seaborn, Jupyter, SciPy, Ubuntu.

모든 데이터 이용은 대회 규정과 라이선스를 준수합니다.

---

## License

Code © 2025 Jongyun Han (Max). Released under the MIT License.
See the LICENSE file for details.

Note: Datasets are NOT redistributed in this repository.
Please download them from the official Kaggle competition page
and comply with the competition rules/EULA.
