# U.S. Yields Prediction

[![ENSAE](https://img.shields.io/badge/ENSAE%20Paris-2025--2026-blue)](https://www.ensae.fr/)
[![Course](https://img.shields.io/badge/Course-Machine%20Learning%20for%20Portfolio%20Management-green)]()

## Context

This project was conducted as part of the Machine Learning for Portfolio Management course taught by Sylvain Champonnois at ENSAE Paris during the first semester of the 2025-2026 academic year.

The objective was to compare linear models against complex non-linear models to determine if significant non-linearities in financial variables allow complex architectures to outperform simple linear baselines in forecasting the U.S. yield curve.

## Project Overview

This repository hosts a comprehensive empirical study on forecasting the direction of U.S. Treasury yields.

The project is designed as a single, self-contained Jupyter Notebook. It encompasses the entire research pipeline—from data import via the FED API to model training and statistical testing.

**Research Goal:** To predict whether weekly U.S. government bond yields will increase ($Y=1$) or decrease ($Y=0$) over a 20-year period and comparing the efficacy of classical econometric approaches versus modern machine learning techniques.

## Key Findings

Our analysis of 171 models trained on 15-year rolling windows yielded the following critical insights:

* **Methodological Pivot:** Initial attempts at daily regression using penalized linear models (Ridge, Lasso, Elastic Net) failed to identify predictive relationships (OOS $R^2 \approx 0$). PCA did not improve results. Consequently, the project pivoted to a weekly binary classification framework.
* **Short-End Predictability:** Our models have predictive power mostly on the short end of the yield curve (< 1 year). Models predicting these maturities statistically outperformed the "majority class" benchmark (verified via McNemar's tests).
* **Best Model Performance:**
    * **Logistic Regression** was the most consistent performer, achieving up to **71% out-of-sample accuracy** for the 3-month yield.
    * **Random Forest** offered marginal improvements in some cases on either very short term yields (1 month maturity) or on the middle of the yield curve, but at the cost of interpretability.
    * **XGBoost, LSTM:** More complex models like XGBoost and LSTM suffered from massive overfitting, predicting mostly zeros and failing to generalize.
* **Features:**
    * For most maturities, the best results came from datasets containing macroeconomic and financial variables.
    * However, the 1-month yield was best predicted (65% accuracy) using only functions of past yields. This could highlight a stronger trend-following and mean-reverting behavior at the very short end of the curve.
    * Long-End Efficiency: For long-term yields (> 2 years), results were disappointing. Models rarely outperformed the naive benchmark, suggesting these markets are less dependent on the macroeconomic variables used.

## Methodology and models

1.  Data Source: Automated fetching of macroeconomic and financial data via the FRED API.
2.  Feature Engineering:
    * Stationarity transformations (differencing/log-differencing).
    * Endogenous feature creation: Lags, rolling means, quantiles.
    * Feature Selection: We created multiple datasets based on Mutual Information (MI) thresholds ($>0.03, >0.035, >0.04, >0.05$) to test if models could handle high-dimensional noise versus pre-filtered data.
3.  Models Evaluated: Penalized Logistic Regression, Random Forest, XGBoost, Deep Learning: Long Short-Term Memory (LSTM) Network.
4.  Evaluation:
    * 15-year rolling-window training period and predicting the subsequent 4 weeks.
    * Statistical significance verification using McNemar's Test against a "Majority Class Classifier" benchmark.

## Future Extensions

Potential directions for further work on this project includes:
* Ensemble Learning: Aggregating the signals from our trained models to improve stability and accuracy.
* Time-Varying Feature Importance: Analyzing how feature importance evolves over the rolling periods to better understand regime changes.
* Trading Strategy Implementation: Utilizing the strong signal on the short end of the curve (maturity < 1 year) to backtest a concrete trading strategy using bond price data.

### Dependencies Used
The analysis relies on the following core libraries:
* `numpy`, `pandas` (Data Manipulation)
* `matplotlib`, `seaborn` (Visualization)
* `scikit-learn` (Machine Learning)
* `xgboost` (Gradient Boosting)
* `torch` (Deep Learning)
* `statsmodels` (Econometrics)
* `fredapi` (Data Source)
