# Business Requirements — app2 Tunable Retrain Streamlit App

Version: 1.0
Date: 2025-09-19

## Purpose
This document summarises the business and product-level requirements for the interactive Streamlit application `app2.py`. The tool is designed to let stakeholders explore, tune, and safely retrain a regularized linear model (Ridge) for predicting hourly wages. It emphasizes interpretability (coefficient variability, sensitivity), safe retraining UX, and evidence-based stability reporting.

## Stakeholders
- Product Owner / Business Analyst
- Data Scientist / ML Engineer
- End users / Analysts who want to test model changes interactively

## Scope
In-scope:
- Interactive model tuning (`Ridge alpha`, target transform) and controlled retraining
- Train/test split, quick fits and optional CV-based coefficient variability checks
- Visualizations: prediction error plot, coefficient variability, metrics table
- Evidence-based stability messaging and actionable remediation guidance

Out-of-scope:
- Automated production deployment, model storage in model registry
- Advanced model types beyond linear regression (not currently required)

## High-level requirements
1. Data ingestion: fetch dataset from OpenML (id=534) using a project-local scikit-learn cache directory.
2. Preprocessing: one-hot encode categorical columns, passthrough numeric features.
3. Modeling: support regularized linear regression (Ridge) with optional target transforms (`identity`, `log10`, `log1p`).
4. Training: fast single-fit training (cached) and gated CV (RepeatedKFold) for coefficient variability analysis.
5. UX: step-driven UI with explicit confirmation for heavy operations; sidebar controls with clear guidance.
6. Stability detection: compute train/test MAE and R² and flag possible overfitting using configurable tolerances.
7. Explainability: show colored gap values and provide an "Explain this result" control with actionable suggestions.

## Functional modules (brief)
- Environment & Setup: local OpenML cache and page config.
- Safe rerun helper: version-agnostic rerun fallback to support different Streamlit installations.
- Data split (Step 5): create and store `X_train`, `X_test`, `y_train`, `y_test` in session state.
- Preprocessor & pipeline builder: OneHotEncoder + Ridge / TransformedTargetRegressor inside a pipeline.
- `ensure_model_trained`: validate inputs, train quick fit, cache model and artifacts in `st.session_state`.
- Cross-validation (Step 11 & 13): repeated K-fold CV with `return_estimator=True` and robust coefficient extraction.
- Metrics & narrative (Step 14–17): compute MAE/RMSE/MedAE/R², store in `latest_metrics`, and render stability message.
- Sidebar: hyperparameters (`alpha`, `transform_choice`) and stability thresholds (`mae_tol`, `r2_tol`), plus help text.

## Acceptance criteria
- The app starts and presents the step-based UI without unhandled exceptions.
- Training populates `st.session_state.model` and `last_trained_at` after confirm/recompute.
- CV steps run when explicitly invoked and produce coefficient variability plots and data.
- The stability message uses the computed metrics and the sidebar tolerances to classify stability and color gap values.
- The "Explain this result" control opens an expander with appropriate, actionable guidance.

## Test cases (high level)
1. Start app with no network: verify error handling for OpenML fetch.
2. Run Step 5 → verify session state keys and shapes.
3. Train with `identity` and `log1p` transforms → verify no exceptions and model predictions work.
4. Run Step 11 (CV) → verify coefficient variability DataFrame shape and plots.
5. Force overfitting (very small `alpha`) → verify stability message flags overfitting and colored gaps.
6. Click "Explain this result" → verify the expander content matches the flag.

## Non-functional requirements
- Performance: CV operations must be gated by user action and present a spinner during execution.
- Robustness: Avoid duplicating Streamlit widget keys; close Matplotlib figures after rendering.
- Maintainability: key logic (training, stability, CV) should be easy to unit test.

## Run instructions (brief)
Refer to `requirements.md` for dependencies. Typical run (PowerShell):

```powershell
cd C:\Users\asbho\stremlit_project\streamlit_venv
# Activate venv if needed
.\Scripts\Activate.ps1
streamlit run .\app2.py
```

## Next recommended roadmap items
- Persist `latest_metrics` and `last_trained_at` to disk so messages survive restarts.
- Add a one-click remediation action in the Explain panel (e.g., set `alpha *= 10` and retrain).
- Add unit tests for `ensure_model_trained` and the stability helper.

---
(End of brief business requirements)
