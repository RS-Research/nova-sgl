import os
import shutil
import site
from pathlib import Path


def find_recbole_general_recommender_dir():
    site_packages = site.getsitepackages()

    for path in site_packages:
        candidate = Path(path) / "recbole" / "model" / "general_recommender"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find recbole/model/general_recommender in site-packages."
    )


def main():
    repo_root = Path(__file__).resolve().parents[1]
    src_model = repo_root / "novasgl" / "models" / "novasgl.py"

    if not src_model.exists():
        raise FileNotFoundError(f"Model file not found: {src_model}")

    target_dir = find_recbole_general_recommender_dir()
    target_model = target_dir / "novasgl.py"
    init_file = target_dir / "__init__.py"

    shutil.copyfile(src_model, target_model)
    print(f"Copied NOVASGL model to: {target_model}")

    import_line = "from recbole.model.general_recommender.novasgl import NOVASGL\n"

    with open(init_file, "r", encoding="utf-8") as f:
        content = f.read()

    if import_line not in content:
        with open(init_file, "a", encoding="utf-8") as f:
            f.write("\n" + import_line)
        print("Registered NOVASGL in RecBole general_recommender __init__.py")
    else:
        print("NOVASGL is already registered.")


if __name__ == "__main__":
    main()