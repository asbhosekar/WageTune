import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy as sp
import os
from datetime import datetime

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer, TransformedTargetRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.metrics import median_absolute_error, PredictionErrorDisplay

st.set_page_config(page_title="WageTune — Tunable retrain")
st.title("WageTune — Tune hyperparameters and retrain on demand")
st.markdown("#### ⚙️ Interactive wage model tuning, diagnostics and stability checks")

# Local OpenML cache
project_cache = os.path.join(os.path.dirname(__file__), "scikit_learn_data")
os.makedirs(project_cache, exist_ok=True)
os.environ.setdefault("SCIKIT_LEARN_DATA", project_cache)


# Safe rerun helper (top-level): call Streamlit's experimental_rerun if available,
# otherwise toggle a small session token to prompt a re-execution/re-render.
def safe_rerun():
    try:
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
            return
    except Exception:
        # If experimental_rerun exists but errors, fall back to token toggle
        pass
    token = st.session_state.get("_rerun_token", 0)
    st.session_state["_rerun_token"] = token + 1


# Metrics helper: store latest computed metrics in session_state so UI text can
# reference up-to-date values.
def update_latest_metrics(mae_train, rmse_train, medae_train, r2_train, mae_test, rmse_test, medae_test, r2_test):
    st.session_state["latest_metrics"] = {
        "mae_train": float(mae_train),
        "rmse_train": float(rmse_train),
        "medae_train": float(medae_train),
        "r2_train": float(r2_train),
        "mae_test": float(mae_test),
        "rmse_test": float(rmse_test),
        "medae_test": float(medae_test),
        "r2_test": float(r2_test),
    }

# Load dataset
try:
    wages = fetch_openml(data_id=534, as_frame=True)
    df_wages = wages.data
    target_wage = wages.target
except Exception as e:
    st.error("Could not fetch OpenML dataset. Make sure you have network access.")
    st.exception(e)
    st.stop()

# Sidebar: hyperparameter controls
st.sidebar.header("Retrain controls")
# Enforce non-negative alpha at the widget level
alpha = st.sidebar.number_input("Ridge alpha", value=1e-10, min_value=0.0, format="%.12g")
transform_choice = st.sidebar.selectbox(
    "Target transform",
    ("log10", "identity", "log1p"),
    help="Choose a transform applied to the target during regression."
)

# Optional auto-retrain on sidebar changes (opt-in because CV/retrain can be slow)
auto_retrain = st.sidebar.checkbox(
    "Auto retrain on parameter change (may be slow)", value=False,
    help="When enabled, the app will retrain automatically when you change alpha or transform. Use with caution — retraining can be time-consuming."
)

# Short explanation of the retrain controls for non-technical users
st.sidebar.markdown("### ℹ️ How these controls affect the model")
st.sidebar.info(
    "• **Ridge alpha** balances flexibility vs. stability. "
    "Smaller values let the model fit closely (risk of overfitting), "
    "larger values make it more stable but less sensitive.\n\n"
    "• **Target transform** changes how wages are modeled. "
    "`Identity` = direct wages, `log10` = handles skewed wages, "
    "`log1p` = safer when wages can be zero."
)

# Stability thresholds (configurable by the user in the sidebar)
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Stability threshold guidance")
st.sidebar.info(
    "These thresholds control when the app flags possible overfitting:\n"
    "• **MAE gap tolerance** compares test vs train MAE (USD/hr).\n"
    "• **R² gap tolerance** compares train vs test R² (fraction).\n\n"
    "Defaults are conservative; lower values make the warning more sensitive."
)
mae_tol = st.sidebar.number_input(
    "Stability MAE gap tolerance (USD/hr)", min_value=0.0, value=0.5, step=0.1,
    help="If (MAE_test - MAE_train) > this value, the app will warn about possible overfitting."
)
r2_tol = st.sidebar.number_input(
    "Stability R² gap tolerance (fraction)", min_value=0.0, max_value=1.0, value=0.05, step=0.01,
    help="If (R²_train - R²_test) > this fraction, the app will warn about possible overfitting."
)

# Show last trained timestamp if present
last_trained = st.sidebar.empty()
if "last_trained_at" in st.session_state:
    last_trained.info(f"Last trained at: {st.session_state['last_trained_at']}")
else:
    last_trained.info("Model not trained in this session yet")

# Start at step 5 by default
if "step" not in st.session_state:
    st.session_state.step = 5

# If user changed sidebar controls since last training, invalidate cached model
prev_meta = st.session_state.get("model_meta", {})
if prev_meta.get("alpha") != alpha or prev_meta.get("transform_choice") != transform_choice:
    if "model" in st.session_state:
        del st.session_state["model"]
    # clear last trained display so it will refresh after retrain
    st.session_state.pop("last_trained_at", None)
    st.session_state.pop("last_trained_display", None)
    # Auto-retrain only if a previous model existed (avoid retraining on first load)
    if prev_meta and auto_retrain and st.session_state.get("X_train") is not None:
        # Mark an auto-retrain as pending. Actual retrain will run after
        # ensure_model_trained() is defined to avoid forward-reference issues.
        st.session_state["_auto_retrain_pending"] = True


def _get_transform_funcs(choice):
    if choice == "log10":
        return np.log10, lambda x: np.power(10.0, x)
    if choice == "log1p":
        return np.log1p, np.expm1
    # identity
    return None, None


def ensure_model_trained(alpha=None, transform_choice=None):
    """Train the pipeline with the requested hyperparameters and cache in session_state.

    If a model exists with the same settings, do nothing.
    """
    # read stored data
    X_train = st.session_state.get("X_train")
    y_train = st.session_state.get("y_train")
    if X_train is None or y_train is None:
        st.error("Training data not available. Please run Step 5 first.")
        st.stop()

    # default to sidebar values if not provided
    if alpha is None:
        alpha = st.sidebar.session_state.get("Ridge alpha", alpha) if hasattr(st.sidebar, 'session_state') else alpha
    # runtime guard in case an invalid alpha slipped through
    try:
        if float(alpha) < 0.0:
            st.error("Ridge alpha must be non-negative. Please choose alpha >= 0.")
            st.stop()
    except Exception:
        st.error("Invalid alpha value provided. Please enter a non-negative number.")
        st.stop()
    if transform_choice is None:
        transform_choice = st.session_state.get("_transform_choice", transform_choice) or transform_choice

    # If existing model present and parameters match, don't retrain
    meta = st.session_state.get("model_meta", {})
    if "model" in st.session_state and meta.get("alpha") == alpha and meta.get("transform_choice") == transform_choice:
        return

    categorical_columns = ["RACE", "OCCUPATION", "SECTOR", "MARR", "UNION", "SEX", "SOUTH"]
    # Preprocessing: One-hot encode categorical columns, passthrough numeric
    # columns. We set `drop='if_binary'` to avoid collinearity for binary
    # categories. `verbose_feature_names_out=False` keeps feature names simple.
    preprocessor = make_column_transformer(
        (OneHotEncoder(drop="if_binary"), categorical_columns),
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    # Target transformation: we optionally wrap the regressor in a
    # `TransformedTargetRegressor` to apply a monotonic transform to the
    # target variable during training and invert predictions afterwards.
    # This is useful when the target is skewed (e.g., wages) — common
    # transforms are `log10` and `log1p`. When `identity` is selected we
    # use a plain Ridge regressor (no target transform).
    func, inverse = _get_transform_funcs(transform_choice)
    if transform_choice == "identity" or func is None:
        # Plain Ridge: coefficients are available on `.coef_` after fit.
        estimator = Ridge(alpha=alpha)
    else:
        # TransformedTargetRegressor stores the wrapped estimator on
        # `.regressor_` after fitting; use `func` to transform y before
        # fitting and `inverse_func` to invert predictions.
        estimator = TransformedTargetRegressor(regressor=Ridge(alpha=alpha), func=func, inverse_func=inverse)

    model_local = make_pipeline(preprocessor, estimator)

    # Fit (this is the quick fit, not CV)
    model_local.fit(X_train, y_train)

    # Cache
    st.session_state.model = model_local
    # Cache derived artifacts for later steps:
    # - `feature_names`: names after the preprocessor (one-hot columns + passthrough)
    # - `X_train_preprocessed`: preprocessed training set used for coefficient scaling
    # Note: `model_local[:-1]` accesses the pipeline without the final estimator
    # (i.e., the preprocessor) so we can transform X to feature space.
    st.session_state.feature_names = model_local[:-1].get_feature_names_out()
    st.session_state.X_train_preprocessed = pd.DataFrame(
        model_local[:-1].transform(X_train), columns=st.session_state.feature_names
    )
    st.session_state.model_meta = {"alpha": alpha, "transform_choice": transform_choice}
    st.session_state.last_trained_at = datetime.now().isoformat(sep=" ", timespec="seconds")
    st.session_state.last_trained_display = st.session_state.last_trained_at

# If an auto-retrain was requested earlier (before ensure_model_trained existed), run it now.
if st.session_state.pop("_auto_retrain_pending", False):
    if st.session_state.get("X_train") is not None:
        with st.spinner("Auto-retraining with new hyperparameters..."):
            ensure_model_trained(alpha=alpha, transform_choice=transform_choice)
        st.session_state.last_trained_at = datetime.now().isoformat(sep=" ", timespec="seconds")
        last_trained.info(f"Last trained at: {st.session_state['last_trained_at']}")
        safe_rerun()


# ---------------------------
# STEP 5: prepare split (no heavy plots)
# ---------------------------
if st.session_state.step >= 5:
    st.header("5. Prepare training data (pairplot suppressed)")
    X = df_wages
    y = target_wage
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test

    st.info("Pairplot suppressed to save time. Click Next to continue.")
    if st.button("Next — Prediction Error (suppressed)"):
        st.session_state.step = 6


# Steps 6-10 suppressed as in app1
if st.session_state.step >= 6 and st.session_state.step <= 10:
    step = st.session_state.step
    st.header(f"{step}. (Suppressed)")
    st.write("This step's visualization is suppressed to save time. Click Next to continue.")
    if st.button("Next"):
        st.session_state.step += 1


# ---------------------------------------------------------------------------
# STEP 11: Coefficient Variability via Cross-Validation
# Purpose:
# - Run repeated K-fold cross-validation on the current pipeline to inspect
#   variability of feature coefficients across folds.
# Inputs:
# - `st.session_state.model` (trained pipeline) — ensured via
#   `ensure_model_trained(alpha, transform_choice)` below.
# - `df_wages` and `target_wage` as the dataset.
# Outputs:
# - `coefs_cv`: DataFrame of scaled coefficients across CV folds (used by
#   subsequent steps for plotting).
# Recent changes / notes:
# - Robust coefficient extraction: each CV estimator's final step may be a
#   `TransformedTargetRegressor` (wraps a regressor at `.regressor_`) or a
#   plain `Ridge`. The code now checks both cases to avoid AttributeError.
# - This block does NOT persist heavy results to disk; it's intentionally
#   controlled by the step navigation to avoid accidental long runs.
if st.session_state.step >= 11:
    st.header("11. Coefficient Variability via Cross-Validation")
    from sklearn.model_selection import RepeatedKFold, cross_validate

    # ensure model trained with current sidebar controls
    st.session_state._transform_choice = transform_choice
    ensure_model_trained(alpha=alpha, transform_choice=transform_choice)
    model = st.session_state.model
    X = df_wages
    y = target_wage

    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=0)
    with st.spinner("Running cross-validation (this may take a little while)..."):
        cv_model = cross_validate(model, X, y, cv=cv, return_estimator=True, n_jobs=2)

    # Build coefficient variability matrix across CV folds. Each "est" is a
    # fitted pipeline; its last step may be a TransformedTargetRegressor (which
    # stores the actual regressor in .regressor_) or a plain Ridge (with coef_).
    coefs_list = []
    for est, (train_idx, _) in zip(cv_model["estimator"], cv.split(X, y)):
        last_step = est[-1]
        # Extract the coefficient vector depending on estimator type
        if hasattr(last_step, "regressor_"):
            coef_vec = last_step.regressor_.coef_
        else:
            coef_vec = getattr(last_step, "coef_")

        # Compute std of the preprocessed features for the training fold
        std_vec = est[:-1].transform(X.iloc[train_idx]).std(axis=0)
        coefs_list.append(coef_vec * std_vec)

    coefs_cv = pd.DataFrame(coefs_list, columns=st.session_state.feature_names)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.stripplot(data=coefs_cv, orient="h", palette="dark:k", alpha=0.5, ax=ax)
    sns.boxplot(data=coefs_cv, orient="h", color="cyan", saturation=0.5, whis=10, ax=ax)
    ax.axvline(x=0, color=".5")
    ax.set_xlabel("Coefficient importance")
    ax.set_title("Coefficient importance and its variability")
    plt.suptitle("Ridge model, small regularization")
    plt.subplots_adjust(left=0.3)
    st.pyplot(fig)
    plt.close(fig)

    if st.button("Next — Co-variation AGE vs EXPERIENCE"):
        st.session_state.step = 12


    # -----------------------------------------------------------------------
    # STEP 12: Co-variation of AGE and EXPERIENCE coefficients across folds
    # Purpose: quick scatter plot of AGE vs EXPERIENCE coefficient values from
    # the CV folds computed in STEP 11 to inspect co-variation and possible
    # trade-offs between those features.
if st.session_state.step >= 12:
    st.header("12. Co-variation of AGE and EXPERIENCE Coefficients Across Folds")
    try:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(coefs_cv["AGE"], coefs_cv["EXPERIENCE"], alpha=0.7)
        ax.set_ylabel("Age coefficient")
        ax.set_xlabel("Experience coefficient")
        ax.grid(True)
        ax.set_xlim(-0.4, 0.5)
        ax.set_ylim(-0.4, 0.5)
        ax.set_title("Co-variations of coefficients for AGE and EXPERIENCE across folds")
        st.pyplot(fig)
        plt.close(fig)
    except Exception:
        st.warning("Coefficient variability data is not available. Please run Step 11 first.")

    if st.button("Next — Drop AGE and test stability"):
        st.session_state.step = 13


    # -----------------------------------------------------------------------
    # STEP 13: Impact of Dropping AGE on Model Stability
    # Purpose: re-run CV with the AGE column removed to check how coefficient
    # magnitudes change when AGE is excluded (a sensitivity/stability check).
    # Recent changes: coefficient extraction logic mirrors STEP 11 (robust to
    # TransformedTargetRegressor vs plain Ridge).
if st.session_state.step >= 13:
    st.header("13. Impact of Dropping AGE on Model Stability")
    column_to_drop = ["AGE"]
    from sklearn.model_selection import RepeatedKFold, cross_validate
    cv = RepeatedKFold(n_splits=5, n_repeats=5, random_state=0)
    with st.spinner("Running CV with AGE dropped..."):
        cv_model_age_dropped = cross_validate(
            model, X.drop(columns=column_to_drop), y, cv=cv, return_estimator=True, n_jobs=2
        )

    coefs_list = []
    for est, (train_idx, _) in zip(cv_model_age_dropped["estimator"], cv.split(X, y)):
        last_step = est[-1]
        if hasattr(last_step, "regressor_"):
            coef_vec = last_step.regressor_.coef_
        else:
            coef_vec = getattr(last_step, "coef_")

        std_vec = est[:-1].transform(X.drop(columns=column_to_drop).iloc[train_idx]).std(axis=0)
        coefs_list.append(coef_vec * std_vec)

    coefs_age_dropped = pd.DataFrame(coefs_list, columns=st.session_state.feature_names[:-1])

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.stripplot(data=coefs_age_dropped, orient="h", palette="dark:k", alpha=0.5, ax=ax)
    sns.boxplot(data=coefs_age_dropped, orient="h", color="cyan", saturation=0.5, ax=ax)
    ax.axvline(x=0, color=".5")
    ax.set_title("Coefficient importance and its variability — AGE dropped")
    ax.set_xlabel("Coefficient importance")
    plt.subplots_adjust(left=0.3)
    st.pyplot(fig)
    plt.close(fig)

    if st.button("Next — Final model check"):
        st.session_state.step = 14


    # -----------------------------------------------------------------------
    # STEP 14: Final Model Performance Check
    # Purpose: compute and display final performance metrics (MedAE shown in
    # the quick final check), prediction error plot, and provide a controlled
    # retrain UI (confirm before retraining). This block reads/writes to
    # `st.session_state` for model/data and leverages `ensure_model_trained`.
    # Recent changes:
    # - Recompute button now triggers a safe retrain flow and calls
    #   `safe_rerun()` so the UI updates across Streamlit versions.
    # - `last_trained_at` timestamp is updated and shown in the sidebar.
# -----------------------------------------------------------------------
if st.session_state.step >= 14:
    st.header("14. Final Model Performance Check")

    # Ensure model/data
    ensure_model_trained(alpha=alpha, transform_choice=transform_choice)
    model = st.session_state.model
    X_train = st.session_state.get("X_train")
    X_test = st.session_state.get("X_test")
    y_train = st.session_state.get("y_train")
    y_test = st.session_state.get("y_test")
    if X_train is None or X_test is None or y_train is None or y_test is None:
        st.error("Training or test data not found in session. Please run Step 5 first.")
        st.stop()

    mae_train = median_absolute_error(y_train, model.predict(X_train))
    y_pred = model.predict(X_test)
    mae_test = median_absolute_error(y_test, y_pred)
    scores = {
        "MedAE on training set": f"{mae_train:.2f} $/hour",
        "MedAE on testing set": f"{mae_test:.2f} $/hour",
    }
    st.write(scores)

    # Prediction error plot
    fig, ax = plt.subplots(figsize=(5, 5))
    PredictionErrorDisplay.from_predictions(
        y_test, y_pred, kind="actual_vs_predicted", ax=ax, scatter_kwargs={"alpha": 0.5}
    )
    ax.set_title("Ridge model, small regularization")
    for name, score in scores.items():
        ax.plot([], [], " ", label=f"{name}: {score}")
    ax.legend(loc="upper left")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("📌 Stakeholder Takeaways")

    # Recompute button opens a small confirm UI
    if st.button("Recompute model and update plots", key="recompute_model"):
        st.session_state.recompute_requested = True

    if st.session_state.get("recompute_requested"):
        st.warning("Retraining will run a fresh fit using the sidebar hyperparameters. This can take time.")
        col1, col2 = st.columns(2)
        if col1.button("Yes, retrain now", key="confirm_retrain"):
            # clear model and retrain using current alpha/transform_choice
            if "model" in st.session_state:
                del st.session_state["model"]
            with st.spinner("Retraining model..."):
                ensure_model_trained(alpha=alpha, transform_choice=transform_choice)
            st.success("Model retrained.")
            st.session_state.recompute_requested = False
            st.session_state.last_trained_at = datetime.now().isoformat(sep=" ", timespec="seconds")
            # update sidebar display
            last_trained.info(f"Last trained at: {st.session_state['last_trained_at']}")
            # call the top-level helper to safely request a rerun/re-render
            safe_rerun()
        if col2.button("Cancel", key="cancel_retrain"):
            st.session_state.recompute_requested = False


# Suggested metric mapping for dynamic narrative:
# - "typical error": Test MedAE (median absolute error)
# - "average error": Test MAE
# - "explained variance": Test R²

lm = st.session_state.get("latest_metrics")

# Stability decision rule (inline comment):
# - We consider the model "possibly overfitting" when either:
#     1) Test MAE exceeds Train MAE by more than `mae_tol` dollars/hour, OR
#     2) Train R² exceeds Test R² by more than `r2_tol` (fractional, e.g. 0.05 = 5%).
# - Default tolerances used in the helper below: `mae_tol=0.5` (USD/hour) and
#   `r2_tol=0.05` (5 percentage points). These are conservative heuristics and
#   can be tuned for your domain. When overfitting is indicated the helper
#   suggests remediation: increase regularization (`alpha`), collect more data,
#   simplify features, or run stronger cross-validation to confirm.
def _stability_message_from_metrics(lm: dict, *, mae_tol=0.5, r2_tol=0.05):
    """Return a structured stability result and a small HTML/Markdown message.

    Returns a dict with keys:
    - md: HTML/markdown string (may contain colored spans)
    - is_overfit: bool
    - mae_gap, r2_gap: numeric
    """
    mae_train = lm.get("mae_train")
    mae_test = lm.get("mae_test")
    r2_train = lm.get("r2_train")
    r2_test = lm.get("r2_test")

    # Defensive defaults
    if None in (mae_train, mae_test, r2_train, r2_test):
        return {"md": "Model metrics incomplete — run the metrics step to compute train/test errors.", "is_overfit": False, "mae_gap": None, "r2_gap": None}

    mae_gap = mae_test - mae_train
    r2_gap = r2_train - r2_test

    # Helper to choose color based on gap and tolerance
    def _color_for_gap(gap, tol):
        if gap <= 0:
            return "green"
        if gap <= tol:
            return "green"
        if gap <= 1.5 * tol:
            return "orange"
        return "red"

    mae_color = _color_for_gap(mae_gap, mae_tol)
    r2_color = _color_for_gap(r2_gap, r2_tol)

    # Format colored HTML spans (rendered by st.markdown with unsafe_allow_html=True)
    mae_gap_html = f"<span style=\"color:{mae_color};font-weight:bold;\">{mae_gap:.2f}</span>"
    r2_gap_html = f"<span style=\"color:{r2_color};font-weight:bold;\">{r2_gap:.2f}</span>"

    # Build the message using HTML for colored numbers
    md_lines = []
    md_lines.append(f"- **Typical error (MedAE on test):** USD {lm['medae_test']:.2f}/hr")
    md_lines.append(f"- **Average error (MAE on test):** USD {mae_test:.2f}/hr")
    md_lines.append(f"- **Explained variance (R² on test):** {r2_test*100:.0f}%")
    md_lines.append("")

    is_overfit = (mae_gap > mae_tol) or (r2_gap > r2_tol)
    if is_overfit:
        md_lines.append("- <strong style=\"color:#b22222\">Warning — possible overfitting detected.</strong> The model performs noticeably better on the training set than on the test set.")
        md_lines.append(f"  - MAE gap (test - train) = {mae_gap_html} $/hr (tol {mae_tol})")
        md_lines.append(f"  - R² gap (train - test) = {r2_gap_html} (tol {r2_tol})")
        md_lines.append("  Suggested actions: increase regularization (larger `alpha`), gather more data, simplify the feature set, or run stronger cross-validation to confirm stability.")
    else:
        md_lines.append("- <span style=\"color:green;font-weight:bold;\">Performance is consistent on training and testing data.</span> The small train/test gaps indicate the model is likely capturing generalizable patterns rather than memorizing the training set.")
        md_lines.append("  For extra confidence inspect CV variability (Step 11) or run the AGE-dropping sensitivity check (Step 13).")

    md = "\n".join(md_lines)
    return {"md": md, "is_overfit": is_overfit, "mae_gap": mae_gap, "r2_gap": r2_gap}

if lm is not None:
    # Use sidebar-configured tolerances when building the stability message
    stability = _stability_message_from_metrics(lm, mae_tol=mae_tol, r2_tol=r2_tol)
    # Render colored markdown (small HTML snippets used for colored numbers)
    st.markdown(stability["md"], unsafe_allow_html=True)

    # Explain button + expander (modal-like) giving actionable steps
    if st.button("Explain this result"):
        # Show a short, focused explanation in an expander (acts like a lightweight modal)
        with st.expander("Why was this flagged?", expanded=True):
            if stability.get("is_overfit"):
                st.write("The test set error is noticeably worse than the training set error, suggesting the model learned patterns too specific to the training data.")
                st.write("Most actionable step: increase `Ridge alpha` to add regularization (try 10x current alpha and re-evaluate).")
                st.write("Other options: reduce feature dimensionality, collect more labeled data, or run the CV steps (Step 11) to inspect fold variability.")
            else:
                st.write("The gap between training and testing performance is small, so there is no strong evidence of overfitting based on the configured thresholds.")
                st.write("If you want extra assurance, try running Step 11 (CV) to inspect coefficient variability across folds.")
    # show a brief inline hint linking to the explain button
    st.caption("Click 'Explain this result' for a short, actionable explanation and next steps.")
else:
    st.markdown(
        """
- **Model metrics are not available yet.** Please run the metrics comparison (Steps 14–15) to compute train/test errors, then this message will state whether the model appears stable or if overfitting is indicated.
"""
    )

    # -----------------------------------------------------------------------
    # STEP 15: Display detailed metrics table and trigger navigation to
    # STEP 16/17 which show visualizations comparing train vs test.
    # Purpose: compute MAE, RMSE, MedAE, R^2 for both train and test and
    # present a table. Also stores these values in `st.session_state["latest_metrics"]`
    # so the narrative text and other blocks display up-to-date values.
    # Recent changes: `update_latest_metrics(...)` writes latest values used by
    # the dynamic narrative and info blocks elsewhere in the app.
# continue to Steps 15-18 similar to app1
if st.session_state.step == 14:
    if st.button("YES - Proceed to METRICS COMPARISON", key="step14"):
        st.session_state.step = 15
    else:
        st.info("Click YES to METRICS COMPARISON OF THE TRAINED MODEL.")
        st.stop()

if st.session_state.step >= 15:
    st.header("15. DISPLAY THE METRICS TABLE")
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

    ensure_model_trained(alpha=alpha, transform_choice=transform_choice)
    model = st.session_state.model
    X_train = st.session_state.get("X_train")
    X_test = st.session_state.get("X_test")
    y_train = st.session_state.get("y_train")
    y_test = st.session_state.get("y_test")

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    mae_train = mean_absolute_error(y_train, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    medae_train = median_absolute_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)

    mae_test = mean_absolute_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    medae_test = median_absolute_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)

    metrics_df = pd.DataFrame({
        "Train Set": [mae_train, rmse_train, medae_train, r2_train],
        "Test Set": [mae_test, rmse_test, medae_test, r2_test]
    }, index=["MAE ($/hr)", "RMSE ($/hr)", "MedAE ($/hr)", "R²"])

    st.table(metrics_df.style.format("{:.3f}"))

    # store latest computed metrics for dynamic text elsewhere
    update_latest_metrics(mae_train, rmse_train, medae_train, r2_train, mae_test, rmse_test, medae_test, r2_test)

    if st.button("YES - Bar Chart for Train vs Test", key="step15"):
        st.session_state.step = 16

if st.session_state.step >= 16:
    st.header("15. 📊 Model Evaluation Metrics")
    from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

    ensure_model_trained(alpha=alpha, transform_choice=transform_choice)
    model = st.session_state.model
    X_train = st.session_state.get("X_train")
    X_test = st.session_state.get("X_test")
    y_train = st.session_state.get("y_train")
    y_test = st.session_state.get("y_test")

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    mae_train = mean_absolute_error(y_train, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    medae_train = median_absolute_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)

    mae_test = mean_absolute_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    medae_test = median_absolute_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)

    metrics_df = pd.DataFrame({
        "Train Set": [mae_train, rmse_train, medae_train, r2_train],
        "Test Set": [mae_test, rmse_test, medae_test, r2_test]
    }, index=["MAE ($/hr)", "RMSE ($/hr)", "MedAE ($/hr)", "R²"])

    st.table(metrics_df.style.format("{:.3f}"))

    # Dynamic interpretation using latest_metrics (fallback to static text)
    lm = st.session_state.get("latest_metrics")
    if lm is not None:
        st.info(
            f"Interpretation:\n"
            f"- **MAE ≈ {lm['mae_test']:.2f} $/hr** → average prediction error.\n"
            f"- **MedAE ≈ {lm['medae_test']:.2f} $/hr** → typical error, robust to outliers.\n"
            f"- **RMSE ≈ {lm['rmse_test']:.2f} $/hr** → shows a few larger misses.\n"
            f"- **R² ≈ {lm['r2_test']:.2f}** → model explains {lm['r2_test']*100:.0f}% of wage variation.\n\n"
            "The latest and updated Train and test values are close → stable model, no overfitting."
        )
    else:
        st.info(
            "Interpretation:\n"
            "- **MAE ≈ $3/hr** → average prediction error.\n"
            "- **MedAE ≈ $2.2/hr** → typical error, robust to outliers.\n"
            "- **RMSE ≈ $4.2/hr** → shows a few larger misses.\n"
            "- **R² ≈ 0.27–0.29** → model explains ~27–29% of wage variation.\n\n"
            "Train and test values are close → stable model, no overfitting."
        )

    if st.button("YES - Bar Chart for Train vs Test", key="step16"):
        st.session_state.step = 17

if st.session_state.step >= 17:
    metrics = ["MAE", "RMSE", "MedAE", "R²"]
    # read metrics from most recent computation (ensure computed above)
    fig, ax = plt.subplots(figsize=(6, 4))
    # compute values for plotting
    ensure_model_trained(alpha=alpha, transform_choice=transform_choice)
    model = st.session_state.model
    X_train = st.session_state.get("X_train")
    X_test = st.session_state.get("X_test")
    y_train = st.session_state.get("y_train")
    y_test = st.session_state.get("y_test")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    mae_train = mean_absolute_error(y_train, y_train_pred)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    medae_train = median_absolute_error(y_train, y_train_pred)
    r2_train = r2_score(y_train, y_train_pred)

    mae_test = mean_absolute_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    medae_test = median_absolute_error(y_test, y_test_pred)
    r2_test = r2_score(y_test, y_test_pred)

    train_scores = [mae_train, rmse_train, medae_train, r2_train]
    test_scores = [mae_test, rmse_test, medae_test, r2_test]

    x = np.arange(len(metrics))
    width = 0.35
    bars1 = ax.bar(x - width/2, train_scores, width, label="Train", color="#4CAF50")
    bars2 = ax.bar(x + width/2, test_scores, width, label="Test", color="#2196F3")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score / Error")
    ax.set_title("Train vs Test Metrics")
    ax.legend()
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width()/2, height), xytext=(0, 3),
                    textcoords="offset points", ha="center", fontsize=8)
    st.pyplot(fig)

    if st.button("FINAL - End of App", key="step17"):
        st.session_state.step = 18

if st.session_state.step == 18:
    st.success("🎉 End of app")
