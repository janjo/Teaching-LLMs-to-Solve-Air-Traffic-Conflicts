# Teaching LLMs to Solve Air Traffic Conflicts: A Curiosity-Driven Approach Using Replay-Generated Data

This repository contains the code and supplementary materials for my MSc Artificial Intelligence project at the University of Leeds.

The project explores the use of curiosity-driven reinforcement learning and generative replay to train Large Language Models (LLMs) for solving air traffic conflicts. It combines modern AI techniques with traditional air traffic control (ATC) practices to build intelligent conflict resolution agents.

## 🚀 Features
- ATC simulator environment for conflict generation and resolution.
- Curiosity-based exploration model using Graph Neural Networks (GNNs).
- Variational Autoencoder (VAE) for trajectory representation.
- Integration of fine-tuned LLMs into the ATC simulation loop.
- Prioritized Generative Replay (PGR) mechanism for efficient training.

## 📂 Project Structure
- `curiosity/` — Curiosity model and training scripts.
- `vae/` — Variational Autoencoder components for encoding replay data.
- `llm/` — LLM evaluation, finetuning, and testing modules.
- `simulator/` — Air traffic control simulation environment.
- `utils/` — Utility functions.
- `checkpoints/` — Folder for saving trained models and intermediate outputs.
- `README.md` — This file.
- `LICENSE` — Project license (MIT).
