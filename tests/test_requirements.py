"""The app lock must pin the same versions as the main lock the models were trained with."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pins(name: str) -> dict[str, str]:
    found = {}
    for line in (ROOT / name).read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([A-Za-z0-9_.-]+)==([^\s;]+)", line)
        if match:
            found[match.group(1).lower()] = match.group(2)
    return found


def test_app_lock_is_a_subset_of_the_main_lock():
    main, app = pins("requirements.txt"), pins("requirements-app.txt")
    assert app, "requirements-app.txt has no pins"
    assert {"streamlit", "lightgbm", "scikit-learn", "pandas"} <= set(app)
    mismatched = {pkg: (ver, main.get(pkg)) for pkg, ver in app.items() if main.get(pkg) != ver}
    assert not mismatched, f"app lock differs from main lock: {mismatched}"


def test_app_lock_leaves_out_analysis_only_packages():
    app = pins("requirements-app.txt")
    assert not {"shap", "numba", "statsmodels", "nbconvert", "ipykernel", "pytest"} & set(app)


def test_streamlit_cloud_requirements_are_a_copy_of_the_app_lock():
    # Streamlit Community Cloud installs the requirements file next to the entrypoint.
    cloud = (ROOT / "app" / "requirements.txt").read_text(encoding="utf-8")
    assert cloud == (ROOT / "requirements-app.txt").read_text(encoding="utf-8")
