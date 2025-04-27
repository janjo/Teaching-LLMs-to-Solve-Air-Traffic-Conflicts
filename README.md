# Teaching LLMs to Solve Air Traffic Conflicts: A Curiosity-Driven Approach Using Replay-Generated Data

This repository contains the code and supplementary materials for my MSc Artificial Intelligence project at the University of Leeds.

The project explores the use of curiosity-driven reinforcement learning and generative replay to train Large Language Models (LLMs) for solving air traffic conflicts. It combines modern AI techniques with traditional air traffic control (ATC) practices to build intelligent conflict resolution agents.

## 🚀 Features
- Conflict-focused training using synthetically generated short-term conflict scenarios.
- Generative replay to produce high-risk situations.
- Curiosity-driven prioritization to focus training on valuable experiences.
- Finetuning of large language models (LLMs) of various sizes.
- Natural language reasoning for enhanced interpretability and human-AI collaboration.
- Improved conflict resolution accuracy compared to unmodified base models.
- Pathways for future extensions to handle broader conflict types.

## 📂 Project Structure
- `curiosity/` — Curiosity model and training scripts.
- `vae/` — Variational Autoencoder components for encoding replay data.
- `llm/` — LLM evaluation, finetuning, and testing modules.
- `simulator/` — Air traffic control simulation environment.
- `utils/` — Utility functions.
- `README.md` — This file.
- `LICENSE` — Project license (MIT).
