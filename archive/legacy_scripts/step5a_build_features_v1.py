# ARCHIVED 2026-10-31: moved from src/ to archive/legacy_scripts/ and renamed. Only file paths were
# updated for the new folder layout; the logic is unchanged. See reports/audit/.
r"""
Build_Dataset.py
Builds the training table: one row per method, label renamed = 1 / 0,
plus features computed ONLY from the Before code (no leakage from the rename).

Positives : data/interim/train_positives.csv
Negatives : every other method in the same Before files that still exists,
            with the same name, in the After file (so it was NOT renamed)
Output    : data/processed/train_features.csv

Run: python Src\Build_Dataset.py
"""
from pathlib import Path
import ast, re, warnings
import pandas as pd

warnings.filterwarnings("ignore", category=SyntaxWarning)

PROJECT = Path(__file__).resolve().parents[2]
COMMITS = PROJECT / "data" / "raw" / "actref" / "commits"
POS_CSV = PROJECT / "data" / "interim" / "final_labels" / "rename_positive_labels.csv"
OUT     = PROJECT / "archive" / "old_datasets" / "method_features_v1_rename_files_only.csv"

VERBS = {
    "get", "set", "is", "has", "can", "should", "add", "remove", "delete", "create",
    "make", "build", "init", "load", "save", "read", "write", "parse", "compute",
    "calc", "calculate", "run", "start", "stop", "update", "check", "validate",
    "find", "search", "fetch", "send", "recv", "receive", "handle", "process",
    "convert", "to", "from", "apply", "train", "predict", "fit", "evaluate", "eval",
    "test", "reset", "clear", "close", "open", "register", "generate", "extract",
    "transform", "encode", "decode", "render", "show", "print", "log", "format",
    "prepare", "setup", "forward", "call", "copy", "merge", "split", "sample",
}
SHORT_OK = {"id", "io", "ip", "ok", "db", "os", "ui", "x", "y", "z", "fn", "nn", "lr"}


# ---------------------------------------------------------------- name helpers
def split_name(name):
    """snake_case and camelCase -> lower-case tokens."""
    parts = re.split(r"_+", name.strip("_"))
    tokens = []
    for p in parts:
        tokens += re.findall(r"[A-Z]+(?=[A-Z][a-z]|\d|\b)|[A-Z]?[a-z]+|[A-Z]+|\d+", p)
    return [t.lower() for t in tokens if t]


def style_of(name):
    core = name.strip("_")
    if not core:
        return "other"
    if "_" in core:
        return "snake"
    if core.islower():
        return "lower"          # single word, fits snake_case too
    if re.search(r"[a-z][A-Z]", core):
        return "camel"
    return "other"


def name_features(name):
    toks = split_name(name)
    vowels = set("aeiou")
    return {
        "name_len": len(name),
        "name_tokens": len(toks),
        "name_is_private": int(name.startswith("_")),
        "name_is_camel": int(style_of(name) == "camel"),
        "name_has_digit": int(any(c.isdigit() for c in name)),
        "name_short_tokens": sum(1 for t in toks if len(t) <= 2 and t not in SHORT_OK and not t.isdigit()),
        "name_no_vowel_tokens": sum(1 for t in toks if len(t) > 1 and not (set(t) & vowels) and not t.isdigit()),
        "name_starts_verb": int(bool(toks) and toks[0] in VERBS),
        "name_single_char": int(len(name.strip("_")) == 1),
    }


# ---------------------------------------------------------------- code helpers
BRANCHES = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler, ast.With,
            ast.AsyncWith, ast.IfExp, ast.comprehension, ast.Assert)


def max_depth(node, depth=0):
    best = depth
    for child in ast.iter_child_nodes(node):
        d = depth + 1 if isinstance(child, (ast.If, ast.For, ast.While, ast.With, ast.Try)) else depth
        best = max(best, max_depth(child, d))
    return best


def code_features(func, name):
    nodes = list(ast.walk(func))
    a = func.args
    params = [x.arg for x in a.args + a.kwonlyargs if x.arg not in ("self", "cls")]
    doc = ast.get_docstring(func) or ""
    body = func.body[1:] if doc else func.body
    only_stub = all(
        isinstance(s, ast.Pass)
        or (isinstance(s, ast.Raise) and "NotImplementedError" in ast.unparse(s))
        or (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
        for s in body
    ) if body else True
    decos = [ast.unparse(d) for d in func.decorator_list]

    idents = set()
    for n in nodes:
        if isinstance(n, ast.Name):
            idents.add(n.id)
        elif isinstance(n, ast.Attribute):
            idents.add(n.attr)
        elif isinstance(n, ast.arg):
            idents.add(n.arg)
    body_tokens = set()
    for i in idents:
        body_tokens.update(split_name(i))
    name_toks = split_name(name)
    overlap = (sum(1 for t in name_toks if t in body_tokens) / len(name_toks)) if name_toks else 0.0

    return {
        "loc": (func.end_lineno or func.lineno) - func.lineno + 1,
        "n_params": len(params),
        "n_defaults": len(a.defaults) + sum(1 for d in a.kw_defaults if d is not None),
        "has_varargs": int(a.vararg is not None),
        "has_kwargs": int(a.kwarg is not None),
        "n_returns": sum(isinstance(n, ast.Return) for n in nodes),
        "n_calls": sum(isinstance(n, ast.Call) for n in nodes),
        "cyclomatic": 1 + sum(isinstance(n, BRANCHES) for n in nodes)
                        + sum(len(n.values) - 1 for n in nodes if isinstance(n, ast.BoolOp)),
        "max_nesting": max_depth(func),
        "has_docstring": int(bool(doc)),
        "doc_len": len(doc),
        "is_stub": int(only_stub),
        "is_async": int(isinstance(func, ast.AsyncFunctionDef)),
        "n_decorators": len(decos),
        "is_property": int(any("property" in d for d in decos)),
        "is_static_or_class": int(any(d in ("staticmethod", "classmethod") for d in decos)),
        "name_body_overlap": round(overlap, 3),
    }


# ---------------------------------------------------------------- file parsing
def parse(path):
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return text, ast.parse(text)
    except (SyntaxError, ValueError):
        return None, None


def collect(tree):
    """{(class_path, name): (func_node, class_n_methods, class_depth)}"""
    out = {}

    def visit(node, class_path, depth):
        funcs = [c for c in ast.iter_child_nodes(node) if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, f"{class_path}.{child.name}" if class_path else child.name, depth + 1)
        for f in funcs:
            out[(class_path, f.name)] = (f, len(funcs) if class_path else 0, depth)

    visit(tree, "", 0)
    return out


def is_dunder(n):
    return n.startswith("__") and n.endswith("__")


# ---------------------------------------------------------------- main
pos = pd.read_csv(POS_CSV)
pos["class"] = pos["class"].fillna("")
pos_keys = {(r.commit, r.file, r["class"], r.old_name) for _, r in pos.iterrows()}
pos_files = pos.groupby("commit")["file"].apply(set).to_dict()

rows, missing_pos = [], set(pos_keys)

for commit, files in pos_files.items():
    cdir = COMMITS / commit
    before_texts = {}
    parsed = {}
    for f in files:
        bpath, apath = cdir / "Before" / f.replace("/", "#"), cdir / "After" / f.replace("/", "#")
        btext, btree = parse(bpath)
        atext, atree = parse(apath)
        if btree is None or atree is None:
            continue
        before_texts[f] = btext
        parsed[f] = (btext, btree, atree)

    all_before_text = "\n".join(before_texts.values())

    for f, (btext, btree, atree) in parsed.items():
        before, after = collect(btree), collect(atree)
        file_loc = btext.count("\n") + 1
        file_styles = [style_of(k[1]) for k in before]
        snake_share = (sum(s in ("snake", "lower") for s in file_styles) / len(file_styles)) if file_styles else 1.0

        for (cls, name), (func, cls_n, depth) in before.items():
            if is_dunder(name) or name.startswith("test"):
                continue
            key = (commit, f, cls, name)
            if key in pos_keys:
                label = 1
                missing_pos.discard(key)
            elif (cls, name) in after:
                label = 0           # still there, same name -> not renamed
            else:
                continue            # removed but not a labelled rename -> ambiguous, skip

            feat = {"commit": commit, "file": f, "class": cls, "name": name, "renamed": label}
            feat.update(name_features(name))
            feat.update(code_features(func, name))
            feat.update({
                "in_class": int(bool(cls)),
                "class_depth": depth,
                "class_n_methods": cls_n,
                "file_n_methods": len(before),
                "file_loc": file_loc,
                "is_test_file": int("test" in f.lower()),
                "style_mismatch": int(style_of(name) == "camel" and snake_share > 0.5),
                "call_sites": len(re.findall(r"\b" + re.escape(name) + r"\s*\(", all_before_text)) - 1,
            })
            rows.append(feat)

df = pd.DataFrame(rows)
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)

print("Rows:            ", len(df))
print("Renamed (1):     ", int(df["renamed"].sum()))
print("Not renamed (0): ", int((df["renamed"] == 0).sum()))
print("Ratio 0:1        ", round((df["renamed"] == 0).sum() / max(1, df["renamed"].sum()), 1))
print("Commits:         ", df["commit"].nunique())
print("Positives not found in Before code:", len(missing_pos))
print("Features:", [c for c in df.columns if c not in ("commit", "file", "class", "name", "renamed")])
print("Saved:", OUT)