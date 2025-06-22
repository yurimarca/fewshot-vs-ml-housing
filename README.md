# fewshot-vs-ml-housing

**A comparative study of few-shot LLM inference vs. traditional machine learning for house price prediction using an Indian real estate dataset.**

## Overview

This project explores the effectiveness of few-shot large language model (LLM) inference compared to traditional machine learning models for predicting housing prices. We use a structured real estate dataset from India and benchmark model performance across both paradigms using metrics like MAE and R².

## Dataset

- **Source**: [House Price Dataset of India (Kaggle)](https://www.kaggle.com/datasets/mohamedafsal007/house-price-dataset-of-india)
- **Size**: ~14,000 listings with features such as area, location, bedrooms, and price

## Methods

- **Traditional ML**: Linear Regression, Random Forest, and other baselines
- **LLM Inference**: Few-shot prompts using LLM models avialable
- **Tracking**: Experiment tracking and comparison via MLflow

## Goals

- Compare performance between ML and LLM few-shot inference
- Analyze trade-offs in accuracy, interpretability, and usability
- Showcase reproducible experimentation using MLflow

