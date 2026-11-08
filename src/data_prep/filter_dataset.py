import json, random, re, shutil, hashlib
from pathlib import Path
from PIL import Image
import cv2

SRC = Path("data/raw/Herbify-Dataset")   # folderul care conține direct plantele
FILTERED = Path("data/filtered")
TRAIN = Path("data/train_15")
FRACTION = 0.15
MIN_SIZE = 100
BLUR_THRESHOLD = 50
SEED = 42
EXT = {".jpg", ".jpeg", ".png"}

random.seed(SEED)
seen = set()

def is_good(path):
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            w, h = im.size
        if min(w, h) < MIN_SIZE:
            return False
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest in seen:
            return False
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False
        if cv2.Laplacian(img, cv2.CV_64F).var() < BLUR_THRESHOLD:
            return False
        seen.add(digest)
        return True
    except Exception:
        return False

def make_abbreviations(names):
    """Prescurtare unică pentru fiecare plantă, din inițialele cuvintelor."""
    result, used = {}, set()
    for name in names:
        clean = re.sub(r"\(.*?\)", "", name)              # scoate parantezele
        words = re.findall(r"[A-Za-z0-9]+", clean) or [name]
        abbr = None
        for k in (1, 2, 3):                                # 1 literă, apoi 2, apoi 3 din fiecare cuvânt
            cand = "".join(w[:k] for w in words).upper()
            if cand not in used:
                abbr = cand
                break
        if abbr is None:                                   # ultimul resort: adaugă un număr
            base, i = "".join(w[:3] for w in words).upper(), 2
            while f"{base}{i}" in used:
                i += 1
            abbr = f"{base}{i}"
        used.add(abbr)
        result[name] = abbr
    return result

classes = sorted(p for p in SRC.iterdir() if p.is_dir() and not p.name.startswith("."))
abbrevs = make_abbreviations([c.name for c in classes])
Path("data").mkdir(exist_ok=True)
Path("data/abbreviations.json").write_text(json.dumps(abbrevs, indent=2, ensure_ascii=False))

# ETAPA 1: Am filtrat pozele si am pastrat același folder pe plantă ca în raw
for class_dir in classes:
    abbr = abbrevs[class_dir.name]
    out = FILTERED / class_dir.name
    out.mkdir(parents=True, exist_ok=True)
    total = good = 0
    for p in sorted(class_dir.rglob("*")):
        if p.suffix.lower() in EXT and not p.name.startswith("."):
            total += 1
            if is_good(p):
                good += 1
                shutil.copy(p, out / f"{abbr}_{good:05d}_filtered{p.suffix.lower()}")
    print(f"{class_dir.name} [{abbr}]: {good}/{total} păstrate după filtrare")

# ETAPA 2: Am extras 15% din pozele filtrate, pe fiecare plantă pentru antrenarea modelului
for class_dir in sorted(p for p in FILTERED.iterdir() if p.is_dir()):
    imgs = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in EXT)
    random.shuffle(imgs)
    n = max(1, int(len(imgs) * FRACTION))
    out = TRAIN / class_dir.name
    out.mkdir(parents=True, exist_ok=True)
    for p in imgs[:n]:
        shutil.copy(p, out / p.name)
    print(f"{class_dir.name}: {n} poze în setul de antrenare")