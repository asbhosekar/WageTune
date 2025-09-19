# Requirements & Setup — app2 Streamlit project

Date: 2025-09-19

This file documents the runtime and developer dependencies to run `app2.py` locally, plus basic troubleshooting.

## Python & environment
- Python 3.10+ recommended (the virtualenv included appears to be for Python 3.13 in this workspace).
- It's recommended to use the provided virtual environment located at `./` (see `Scripts/` folder).

## Key Python packages
The app relies on these major packages (already present in the included `Lib/site-packages`):
- streamlit
- pandas
- numpy
- scipy
- scikit-learn
- matplotlib
- seaborn

If you need to recreate the environment, a minimal `pip` install set would be:

```powershell
pip install streamlit pandas numpy scipy scikit-learn matplotlib seaborn
```

## Running the app
1. Activate your virtualenv (Windows PowerShell):

```powershell
cd C:\Users\asbho\stremlit_project\streamlit_venv
.\Scripts\Activate.ps1
```

2. Run Streamlit:

```powershell
streamlit run .\app2.py
```

3. Navigate to the URL printed by Streamlit (usually `http://localhost:8501`).

## Troubleshooting
- OpenML fetch fails with PermissionError
  - We set `SCIKIT_LEARN_DATA` to a project-local `scikit_learn_data` directory to avoid this. Ensure the directory exists and the process has write access.
- Streamlit duplicate widget key errors
  - Avoid editing widget labels/keys in a way that duplicates keys between reruns. The app tries to use unique keys for confirm/cancel flows where needed.
- Matplotlib "Too many figures" warning
  - The app calls `plt.close(fig)` after rendering with `st.pyplot()` to prevent leaks. If you still see warnings, check for other plotting code that may create figures without closing them.

## Dev notes
- To run unit tests (if added), run `pytest` from the workspace root after installing `pytest`.

## Optional: export dependency versions
You can create a pinned `requirements.txt` from the current environment:

```powershell
pip freeze > requirements.txt
```

This will produce a full list of versions installed in the virtual environment.

---
(End of requirements.md)
