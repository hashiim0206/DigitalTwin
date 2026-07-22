# Digital-Twin-in-the-Loop AI Traffic Signal Control

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://digitaltwinai.streamlit.app)

An end-to-end system that bridges the **Simulation-to-Reality (Sim-to-Real) Gap** in AI-based traffic signal control by constructing a physics-calibrated Digital Twin from real-world drone footage and training a constrained Reinforcement Learning agent within it.

> **Live Dashboard:** [https://digitaltwinai.streamlit.app](https://digitaltwinai.streamlit.app)

---

## About

Traditional traffic signals operate on fixed timing plans, causing unnecessary delays and congestion. While Reinforcement Learning (RL) can dynamically optimize signal phases, a critical vulnerability exists: agents trained in standard microsimulators exploit unrealistic physics (e.g., instantaneous braking), producing policies that fail catastrophically on real roads.

This project solves the problem by building a **Digital Twin** — an exact digital replica of a real intersection — calibrated with Bayesian Machine Learning to match actual human driving behavior extracted from drone video. The RL agent is then trained inside this hyper-realistic environment with an explicit fidelity penalty, ensuring its learned policy transfers safely to real-world deployment.

---

## System Architecture & Flow

The system operates through four sequential modules:

```
┌─────────────────────┐     ┌─────────────────────────┐     ┌──────────────────────────┐     ┌──────────────────────────┐
│  Module 1            │     │  Module 2                │     │  Module 3                 │     │  Module 4                 │
│  Video Trajectory    │────▶│  Microsimulation         │────▶│  AI Controller            │────▶│  Performance              │
│  Extraction &        │     │  Physics Calibration     │     │  Optimization (RL)        │     │  Analytics & Safety       │
│  Validation          │     │  (Bayesian Optimization) │     │                           │     │                           │
└─────────────────────┘     └─────────────────────────┘     └──────────────────────────┘     └──────────────────────────┘
```

1. **Video Trajectory Extraction:** Computer vision (YOLO + DeepSORT) parses overhead drone footage to extract X/Y coordinates, speed, and lane assignments of every vehicle.
2. **Digital Twin Validation:** Extracted data is injected into the SUMO microsimulator. Turn type proportions and temporal volume distributions are verified against Ground Truth to confirm the baseline simulation mirrors reality.
3. **Microsimulation Physics Calibration:** Optuna (Bayesian Optimization) runs 50+ trials to tune the Krauss Car-Following Model parameters (`tau`, `accel`, `decel`, `minGap`, `sigma`), minimizing Spatial RMSE against real-world trajectories.
4. **AI Controller Optimization:** A PPO Reinforcement Learning agent trains inside the calibrated Digital Twin. It is rewarded for reducing system delay and penalized for deviating from calibrated fidelity, causing excessive CO2 emissions, or triggering safety conflicts.

---

## Project Structure

```
DigitalTwin/
├── configs/                         # SUMO network and route configuration files
│   ├── version.net.xml              # SUMO road network definition
│   └── empty.rou.xml                # Base empty route file template
│
├── data/                            # Input data
│   ├── input.mp4                    # Source drone video of the intersection
│   └── gg.csv                       # Ground truth vehicle counts (manual annotation)
│
├── experiments/                     # Experiment configuration
│   ├── calibration_bayesopt.yaml    # Bayesian optimization hyperparameters
│   └── rl_config.yaml               # Reinforcement learning training config
│
├── src/                             # Source code
│   ├── app.py                       # Streamlit dashboard (main entry point)
│   ├── traffic_master.py            # Master orchestrator for the full pipeline
│   ├── comprehensive_comparison.py  # Validation: AI vs Ground Truth vs SUMO metrics
│   ├── dashboard_generator.py       # Static plot generation (Matplotlib)
│   ├── extract_sumo_state.py        # Extracts SUMO vehicle state via TraCI to JSON
│   ├── traci_runner.py              # TraCI interface for running SUMO simulations
│   ├── trajectory_analyzer.py       # Spatial trajectory RMSE analysis
│   │
│   ├── calibration/                 # Module 2: Bayesian physics calibration
│   │   ├── objective.py             # Optuna objective function
│   │   ├── optimizer.py             # Bayesian optimization runner
│   │   ├── match_real_sim.py        # Real-to-simulated trajectory matching
│   │   ├── metrics.py               # Spatial & temporal RMSE computation
│   │   ├── evaluate_baseline.py     # Evaluates default SUMO physics
│   │   ├── evaluate_optimized.py    # Evaluates calibrated SUMO physics
│   │   ├── plots.py                 # Calibration visualization plots
│   │   └── report.py                # LaTeX & CSV report generation
│   │
│   ├── control/                     # Module 3: Reinforcement Learning
│   │   ├── env.py                   # Custom Gymnasium environment (SUMO + TraCI)
│   │   ├── train.py                 # PPO training with fidelity constraint
│   │   ├── train_baseline.py        # PPO training without fidelity (ablation)
│   │   ├── evaluate.py              # Strategy benchmarking & emissions/safety
│   │   └── ablation.py              # Sensitivity analysis across fidelity weights
│   │
│   └── experiments/                 # Scenario testing
│       ├── generate_scenario_sumo_files.py  # Generates variant SUMO configs
│       ├── scenario_runner.py       # Runs scenario simulations
│       └── run_split_study.py       # Green-split timing study
│
├── outputs/                         # Generated outputs (after running the pipeline)
│   ├── traffic_data.json            # AI-detected vehicle trajectories from video
│   ├── sumo_state.json              # SUMO-simulated vehicle states
│   ├── best_params.json             # Best calibrated Krauss parameters
│   ├── calibrated.rou.xml           # Calibrated SUMO route file
│   ├── rl_logs/                     # TensorBoard training logs & saved models
│   ├── rl_logs_baseline/            # Baseline (unconstrained) training logs
│   └── scenarios/                   # Scenario simulation configs & results
│
├── results/                         # Final evaluation metrics
│   ├── evaluation_metrics.csv       # Strategy comparison (Delay, RMSE, CO2, Safety)
│   ├── ablation_metrics.csv         # Fidelity weight sensitivity results
│   ├── baseline_metrics_summary.csv # Default physics performance
│   └── optimized_metrics_summary.csv# Calibrated physics performance
│
├── models/                          # Pre-trained model weights
│   └── yolov8n.pt                   # YOLOv8 nano weights for vehicle detection
│
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── .gitignore                       # Git ignore rules
```

---

## Requirements & Installation

### Prerequisites
| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.8+ | Runtime |
| SUMO | 1.18+ | Traffic microsimulation engine |
| pip | Latest | Package management |

> **Important:** SUMO must be installed and the `SUMO_HOME` environment variable must be set correctly on your system PATH. You can verify by running `sumo --version` in your terminal.

### Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/hashiim0206/DigitalTwin.git
   cd DigitalTwin
   ```

2. **Install Python Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### How to Run the Dashboard
Launch the interactive Streamlit Command Center:
```bash
streamlit run src/app.py
```
A browser window will automatically open at `http://localhost:8501`.

The dashboard contains five tabs:
| Tab | Description |
|-----|-------------|
| **System Validation** | Compares AI detection vs Ground Truth vs SUMO (volume, turn accuracy, temporal distribution) |
| **Microsimulation Tuning** | Displays Bayesian optimization convergence and calibrated physics parameters |
| **AI Controller Optimization** | Live RL training curves (reward convergence, episode length) |
| **Performance Analytics** | Strategy benchmarking — Delay, Fidelity RMSE, CO2 Emissions, Safety Conflicts |
| **Sensitivity Analysis** | Ablation study showing the Delay vs Fidelity tradeoff across constraint weights |

---

## Full Pipeline Execution (Re-running from Scratch)

If you wish to re-train the AI agent, re-run Bayesian optimization, or generate brand-new simulation results from scratch, execute the following steps in sequence:

1. **Module 1: Video Trajectory Extraction**
   ```bash
   python src/traffic_master.py
   ```
   *Processes raw drone video (`data/input.mp4`) using YOLOv8 + Kalman Filtering to extract vehicle coordinates and turn types. Generates `outputs/traffic_data.json`.*

2. **Module 2: Microsimulation Physics Calibration**
   ```bash
   python src/calibration/optimizer.py
   ```
   *Runs 50 Optuna Bayesian Optimization trials in SUMO to calibrate driver physics parameters (`tau`, `accel`, `decel`, `minGap`, `sigma`). Generates `outputs/best_params.json` and `outputs/sumo_state.json`.*

3. **Module 3: AI Controller Training**
   ```bash
   python src/control/train.py
   ```
   *Trains the PPO Reinforcement Learning agent inside the calibrated Digital Twin using the Spatial RMSE fidelity penalty. Generates `outputs/rl_logs/final_ppo_model.zip` and `outputs/rl_logs/evaluations.npz`.*

4. **Module 4: Baseline Strategy Benchmarking**
   ```bash
   python src/control/evaluate.py
   ```
   *Evaluates the AI agent against Fixed-Time, Max-Pressure, and Baseline RL across Delay, Fidelity RMSE, HBEFA CO2 Emissions, and Safety Conflicts. Generates `results/evaluation_metrics.csv`.*

5. **Module 5: Sensitivity & Ablation Study**
   ```bash
   python src/control/ablation.py
   ```
   *Evaluates trade-offs across varying constraint weights (`w_fidelity`). Generates `results/ablation_metrics.csv`.*

After running the pipeline, launch the dashboard (`streamlit run src/app.py`) to visualize the newly generated metrics!

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'comprehensive_comparison'` | Run `streamlit run src/app.py` from the **root** `DigitalTwin/` directory, not from inside `src/`. |
| `ModuleNotFoundError: No module named 'traci'` | Ensure SUMO is installed and `SUMO_HOME` is set in your environment variables. |
| Dashboard shows empty charts | Ensure the `outputs/` directory contains `traffic_data.json`, `sumo_state.json`, and the `results/` directory contains `evaluation_metrics.csv`. |
| `FileNotFoundError` on `data/gg.csv` | The Ground Truth CSV must exist in the `data/` folder. This file is included in the repository. |

---

## Future Work
*   **Multi-Modal Operations:** Injecting pedestrian crosswalks and priority bus lanes to test how the constrained AI handles real-world multi-modal constraints.
*   **City-Wide Network Scaling:** Expanding from a single intersection to a multi-intersection grid, requiring cooperative multi-agent RL to manage macroscopic traffic waves.
*   **Extended Compute Training:** Migrating the training pipeline to a High-Performance Computing (HPC) cluster (1,000,000+ timesteps) to drive the policy toward theoretical optimal limits.
