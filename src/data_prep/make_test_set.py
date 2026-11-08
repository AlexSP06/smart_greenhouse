import shutil
from pathlib import Path

FILTERED = Path("data/filtered")
TRAIN = Path("data/train_15")
TEST = Path("data/test")

for class_dir in sorted(p for p in FILTERED.iterdir() if p.is_dir()):
    used = {p.name for p in (TRAIN / class_dir.name).iterdir()}
    out = TEST / class_dir.name
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in class_dir.iterdir():
        if p.is_file() and p.name not in used and not p.name.startswith("."):
            shutil.copy(p, out / p.name)
            n += 1
    print(f"{class_dir.name}: {n} poze de test")
