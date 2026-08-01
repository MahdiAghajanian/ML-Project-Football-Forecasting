<h1 align="center">
Forecasting Competitive Football
</h1>

<p align="center">
  <img src="assets/github-banner.png" width="100%">
</p>

<p align="center">
Probabilistic Football Forecasting with Natural Gradient Boosting
</p>

---

## Overview

This repository contains the implementation for the Machine Learning course project **Forecasting Competitive Football**.

The objective is to develop an end-to-end forecasting pipeline using **StatsBomb Open Data**, producing calibrated probabilistic predictions for football matches before kick-off and throughout live play.

The primary contribution of this work is a reimplementation of **Natural Gradient Boosting (NGBoost)** based on the original paper, adapted for football outcome prediction and goal-margin estimation.

---

## Features

- End-to-end data integration from StatsBomb Open Data
- Leakage-aware feature engineering pipeline
- Pre-match outcome prediction
- Goal-margin regression
- In-play probabilistic forecasting
- NGBoost reimplementation
- Model calibration and uncertainty estimation
- SHAP-based model interpretation

---

## Methodology

```
Raw Event Data
        │
        ▼
Data Integration
        │
        ▼
Feature Engineering
        │
        ▼
NGBoost
        │
        ▼
Probabilistic Prediction
        │
        ▼
Evaluation
```

---

## Repository Structure

```
.
├── assets/
├── data/
├── notebooks/
```

---

## References

This implementation is based on the following work:

> O'Malley, M., Sykulski, A. M., Lumpkin, R., & Schuler, A. (2023).  
> **Probabilistic Prediction of Oceanographic Velocities with Multivariate Gaussian Natural Gradient Boosting.**  
> *Environmental Data Science*, 2, e10.  
> https://doi.org/10.1017/eds.2023.4

Open-access paper:
https://www.cambridge.org/core/journals/environmental-data-science/article/probabilistic-prediction-of-oceanographic-velocities-with-multivariate-gaussian-natural-gradient-boosting/F26F2BD51213758208B0EBAE51D1A973

---