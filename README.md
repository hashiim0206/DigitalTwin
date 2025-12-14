# Digital Twin Traffic Simulation

## Overview
This project builds a high-fidelity digital twin of a real intersection using SUMO, Python, and real sensor/video data.

It includes:
- trajectory extraction
- coordinate transformation
- SUMO network + routes generation
- simulation + validation
- plots, metrics, and analytics

## Repository Structure
\\\
src/                Python scripts
sumo_configs/       SUMO network, routes, configs
data_raw/           raw inputs (video, raw CSVs)
data_processed/     cleaned trajectory CSVs
output/             simulation results
docs/               diagrams and documentation
\\\

## How to Run
\\\
pip install -r requirements.txt
sumo-gui -c sumo_configs/configs/main.sumocfg
\\\

## Features
- Real-to-SUMO coordinate mapping
- Route generation from raw trajectories
- Automated SUMO simulation
- RMSE + timing drift evaluation

## License
MIT (or your choice)
