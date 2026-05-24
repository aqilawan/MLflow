# 📊 LLM Evaluation with MLflow + Groq + DagsHub

This project demonstrates how to evaluate Large Language Models (LLMs) using **MLflow**, **DagsHub**, and **Groq API (LLaMA-3.1)**.  
It tracks experiments, compares outputs with ground truth, and logs performance metrics in a reproducible ML pipeline.

---

## Project Overview

In this project, we:

- Connect MLflow tracking with DagsHub
- Use Groq API (LLaMA-3.1-70B) as the LLM backend
- Build an evaluation dataset (LoRA & QLoRA questions)
- Compare model outputs with ground truth answers
- Log metrics and artifacts using MLflow

---

## Key Concepts Used

- LLM Evaluation
- MLflow Experiment Tracking
- DagsHub Integration
- Groq API (OpenAI-compatible endpoint)
- Prompt Engineering
- Answer Similarity Metrics

---

## Project Structure

Gen AI Using ML Flow/
│
├── main.py            # Main evaluation script
├── requirements.txt   # Dependencies
├── .gitignore         # Ignored files
├── eval.csv           # Generated evaluation results
└── README.md          # Project documentation
