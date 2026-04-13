# Workplace Stress ABM + ML + RL

This project models workplace stress, burnout, productivity, and absenteeism using an agent-based simulation framework. It combines a theory-driven rule-based model with machine learning and reinforcement learning components to study how worker coping behavior and organizational interventions affect long-term outcomes.

## Project Overview

The goal of this project is to simulate how workplace demands, resources, and coping strategies interact over time. The project compares different behavioral architectures in the same simulated workplace environment:

- Rule-based coping policy
- Machine learning-based worker action policy
- Reinforcement learning-based intervention policy

This project is motivated by research questions in occupational stress, burnout, and decision-making under workplace strain.

## Methods

### Rule-Based Agent-Based Model
Each worker agent is assigned stable traits such as:
- autonomy
- social support
- job insecurity
- stress sensitivity
- recovery rate

Daily worker states include:
- workload
- fatigue
- stress
- productivity
- burnout risk
- absenteeism

The rule-based model simulates how these factors evolve over time under different workplace conditions.

### Machine Learning Extension
A GRU-based model is trained on worker history and current state variables to predict coping actions. The ML policy is compared against the original rule-based policy to evaluate whether learned behavior improves simulated outcomes.

### Reinforcement Learning Extension
A DQN-based intervention policy is used to test organizational strategies such as reducing workload or increasing support. The RL agent learns which interventions best reduce stress and burnout while preserving productivity.

## Scenarios

The simulation compares multiple workplace settings:
- Baseline
- High-strain
- Intervention

These scenarios allow evaluation of stress and burnout dynamics under different organizational conditions.

## Results

Key outcomes analyzed include:
- average stress
- productivity
- cumulative absenteeism
- burnout onset share
- burnout onset timing

Main findings from the current version:
- The ML policy reduced stress in high-strain and intervention settings
- The ML policy improved productivity relative to the rule-based baseline
- Burnout onset was delayed under the learned policy
- Cumulative absenteeism was substantially reduced in difficult workplace conditions

## Repository Structure

```text
workplace_stress_abm_ml_rl/
├── README.md
├── requirements.txt
├── src/
├── results/
└── figures/
