# WageTune — Interactive Ridge Regression Tuning App
> Interactive Streamlit app for wage prediction with Ridge regression, stability checks, and stakeholder‑friendly narratives.  
> **Frozen Release v1.0 — 2025‑09‑19**

---

## 🚀 Quick Start

```powershell
# Clone repo
git clone git@github.com:asbhosekar/WageTune.git
cd WageTune

# (Optional) activate your virtual environment
.\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.md

# Run the app
streamlit run app2.py
📌 Features
Ridge Regression with α tuning — experiment with regularization strength

Target transforms — identity, log10, log1p

Cross‑validation (CV) — gated runs to inspect coefficient variability

Metrics dashboard — MAE, RMSE, MedAE, R² with train/test comparison

Stability check — configurable tolerances (mae_tol, r2_tol) with colored gap highlighting

Explainability — “Explain this result” button with concise remediation steps

Robust session state — safe reruns, cached training, and OpenML dataset caching

🆕 v1.0 Highlights
Renamed app to WageTune (previously app2.py)
Added ensure_model_trained(...) for lazy, cached training
Local OpenML cache (scikit_learn_data) to avoid permission errors
Configurable stability thresholds in sidebar (mae_tol, r2_tol)
Colored stability messages (green/orange/red)
Sidebar help text for Ridge alpha, target transforms, and thresholds
Cleaned up plotting (close figures after rendering)
Added BUSINESS_REQS.md and requirements.md

📂 Repository Structure
Code
WageTune/
├── app2.py              # Main Streamlit app (WageTune)
├── BUSINESS_REQS.md     # Business requirements summary
├── requirements.md      # Dependencies
└── .gitignore           # Repo hygiene
🛠 Troubleshooting
OpenML fetch fails → ensure scikit_learn_data exists and is writable
CV runs are slow → expected, as they are compute‑intensive
Matplotlib warnings → latest app2.py closes figures after rendering

📈 Roadmap
Persist latest_metrics and last_trained_at across restarts
Add one‑click remediation in Explain panel (e.g., auto‑adjust α)
Add unit tests for stability helpers
Debug mode for faster iteration (reduced CV repeats)
Background/offline CV jobs for production scaling

✅ Acceptance Criteria (v1.0)
App starts without unhandled exceptions
Step 5 produces X_train/X_test in the session state
Training populates st.session_state.model and last_trained_at
The stability message shows colored gap values and an explanation
CV plots render without leaking figures

👤 Author
Ashish Bhosekar
Code
---

✨ This version is concise at the top (Quick Start + Features), detailed in the middle (Highlights, Troubleshooting, Roadmap), and professional at the bottom (Acceptance Criteria + Author). It reads like a proper open‑source project page.

Would you like me to also suggest a **short GitHub tagline** (the one‑liner
