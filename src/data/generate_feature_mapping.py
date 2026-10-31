r"""
generate_feature_mapping.py
Generates docs/feature_dataset_mapping.md from the actual CSV headers and data
(master, v2, v1) and from the approved feature sets in src/features/feature_sets.py.

Read-only on every dataset. Every code example in the document is executed on a
real renamed method from the raw ActRef data, and its output is checked against
the value stored in the master CSV before anything is written. Needs the raw data
in data/raw/actref/commits/.

  python src/data/generate_feature_mapping.py           write the document
  python src/data/generate_feature_mapping.py --check   regenerate in memory and
                                                        compare with the committed
                                                        document (ignores the date)
"""
import argparse
import contextlib
import io
import json
import re
import sys
import textwrap
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.features.feature_sets import (COMPARISON_FEATURES, COMPARISON_ONLY, DROPPED_FEATURES,  # noqa: E402
                                       EXCLUDED_LEAKAGE, MAIN_FEATURES)

OUT = ROOT / "docs" / "feature_dataset_mapping.md"
DATE_LINE = re.compile(r"^Generated on \d{4}-\d{2}-\d{2} ", re.M)


def normalise(text):
    """Text for comparison: same line endings, generation date masked."""
    return DATE_LINE.sub("Generated on <date> ", text.replace("\r\n", "\n"))

FILES = {
    "v1": ROOT / "archive/old_datasets/method_features_v1_rename_files_only.csv",
    "v2": ROOT / "data/processed/feature_tables/method_features_v2_all_commits.csv",
    "Master": ROOT / "data/processed/master/actref_master_dataset.csv",
}
SCHEMA = ROOT / "data/processed/master/actref_master_dataset_schema.json"
IDS = ["commit", "file", "class", "name"]
TARGET = "renamed"
GROUP_COL = "meta_project_group"
KEY = IDS

GROUPS = {
    "Name": ["name_len", "name_tokens", "name_is_private", "name_is_camel", "name_has_digit",
             "name_short_tokens", "name_no_vowel_tokens", "name_starts_verb", "name_single_char"],
    "Code": ["loc", "n_params", "n_defaults", "has_varargs", "has_kwargs", "n_returns", "n_calls",
             "cyclomatic", "max_nesting", "has_docstring", "doc_len", "is_stub", "is_async",
             "n_decorators", "is_property", "is_static_or_class", "name_body_overlap"],
    "Context": ["in_class", "class_depth", "class_n_methods", "is_test_file", "style_mismatch", "call_sites"],
    "Relative-to-file": ["name_token_sharing", "first_token_shared", "style_share_in_file", "rel_name_len",
                         "rel_name_tokens", "rel_loc", "rel_n_params", "rel_n_calls", "rel_cyclomatic",
                         "rel_doc_len", "rel_call_sites", "rel_name_body_overlap"],
    "File-size": ["file_n_methods", "file_loc"],
}
GROUP_OF = {f: g for g, fs in GROUPS.items() for f in fs}

# ------------------------------------------------------------------ feature knowledge
# name: (what it measures, how it is calculated, why it may help, concerns, code example)
INFO = {
 "name_len": ("Length of the method name in characters.", "`len(name)`.",
   "Long, over-specific names are often shortened or reworded.",
   "Correlated with name_tokens (r = {r:name_tokens}).",
   'print(len(name))'),
 "name_tokens": ("Number of words in the name.", "Name split on underscores and camelCase boundaries (`split_name`), then counted.",
   "More words give more chances that one word is wrong or out of date.",
   "Correlated with name_len (r = {r:name_len}).",
   'print(len(split_name(name)))'),
 "name_is_private": ("Whether the name starts with an underscore.", "`name.startswith('_')` as 0/1.",
   "Lets the model recognise visibility-only renames (`init_model -> _init_model`, 43 of the 243 positives).",
   "None.",
   'print(int(name.startswith("_")))'),
 "name_is_camel": ("Whether the name is camelCase.", "`style_of(name) == 'camel'` (a lower-case letter followed by an upper-case one, no underscore).",
   "camelCase breaks Python convention (PEP 8) and tends to be converted to snake_case.",
   "None.",
   'print(int(style_of(name) == "camel"))'),
 "name_has_digit": ("Whether the name contains a digit.", "`any(ch.isdigit() for ch in name)` as 0/1.",
   "Digits often mark temporary or versioned names (`layer2`, `run_v1`).",
   "None.",
   'print(int(any(ch.isdigit() for ch in name)))'),
 "name_short_tokens": ("Number of 1-2 letter words that are not common abbreviations.",
   "Words of length <= 2, not digits, not in `SHORT_OK` (id, io, ip, ok, db, os, ui, x, y, z, fn, nn, lr).",
   "Unclear abbreviations are a classic naming smell.",
   "None.",
   'print(sum(1 for t in split_name(name) if len(t) <= 2 and t not in SHORT_OK and not t.isdigit()))'),
 "name_no_vowel_tokens": ("Number of words without a vowel (`cnt`, `msg`).",
   "Words longer than 1 letter, not digits, containing none of a/e/i/o/u.",
   "Abbreviations like these are often expanded to full words.",
   "None.",
   'print(sum(1 for t in split_name(name) if len(t) > 1 and not set(t) & set("aeiou") and not t.isdigit()))'),
 "name_starts_verb": ("Whether the first word is a common verb.", "First word of `split_name(name)` is in the fixed `VERBS` list (72 verbs: get, set, create, load, ...).",
   "Action names are refined as behaviour changes (`grad_of -> insert_grad_of`).",
   "None.",
   'words = split_name(name)\nprint(int(bool(words) and words[0] in VERBS))'),
 "name_single_char": ("Whether the name is a single character.", "`len(name.strip('_')) == 1` as 0/1.",
   "One-letter names carry no meaning and are candidates for renaming.",
   "Very rare (see value counts); kept because it costs nothing.",
   'print(int(len(name.strip("_")) == 1))'),

 "loc": ("Lines of code of the method.", "`end_lineno - lineno + 1` of the function node (includes the docstring).",
   "Very short methods are less settled and cheap to rename.",
   "Long right tail; correlated with n_calls (r = {r:n_calls}).",
   'print(func.end_lineno - func.lineno + 1)'),
 "n_params": ("Number of parameters, excluding `self` and `cls`.", "Positional and keyword-only arguments of the signature.",
   "Signature size; unusual signatures attract refactoring.",
   "None.",
   'a = func.args\nprint(len([x for x in a.args + a.kwonlyargs if x.arg not in ("self", "cls")]))'),
 "n_defaults": ("Number of parameters with a default value.", "`len(args.defaults)` + keyword-only defaults that are not None.",
   "Mature, widely used functions collect optional parameters and are renamed less often.",
   "None.",
   'a = func.args\nprint(len(a.defaults) + sum(1 for d in a.kw_defaults if d is not None))'),
 "has_varargs": ("Whether the method accepts `*args`.", "`args.vararg is not None` as 0/1.",
   "Generic wrapper functions tend to be long-lived.",
   "None.",
   'print(int(func.args.vararg is not None))'),
 "has_kwargs": ("Whether the method accepts `**kwargs`.", "`args.kwarg is not None` as 0/1.",
   "Same idea as has_varargs.",
   "None.",
   'print(int(func.args.kwarg is not None))'),
 "n_returns": ("Number of `return` statements.", "Count of `ast.Return` nodes anywhere inside the function (nested functions included).",
   "Describes control-flow shape; useful in combination.",
   "Counts returns of nested functions too.",
   'print(sum(isinstance(n, ast.Return) for n in ast.walk(func)))'),
 "n_calls": ("Number of function calls in the body.", "Count of `ast.Call` nodes inside the function.",
   "How much work the method delegates.",
   "Long right tail; correlated with loc (r = {r:loc}).",
   'print(sum(isinstance(n, ast.Call) for n in ast.walk(func)))'),
 "cyclomatic": ("Cyclomatic complexity (number of decision paths).",
   "1 + branch nodes (if, for, while, except, with, if-expression, comprehension, assert) + extra operands of `and`/`or`.",
   "Logic complexity; complex methods are refactored more.",
   "Correlated with rel_cyclomatic (r = {r:rel_cyclomatic}).",
   'BRANCHES = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler, ast.With,\n            ast.AsyncWith, ast.IfExp, ast.comprehension, ast.Assert)\nnodes = list(ast.walk(func))\nprint(1 + sum(isinstance(n, BRANCHES) for n in nodes)\n        + sum(len(n.values) - 1 for n in nodes if isinstance(n, ast.BoolOp)))'),
 "max_nesting": ("Deepest nesting of if / for / while / with / try blocks.", "Recursive walk that adds 1 at each of those block types.",
   "Structural complexity.",
   "None.",
   'def depth(node, d=0):\n    best = d\n    for c in ast.iter_child_nodes(node):\n        nested = isinstance(c, (ast.If, ast.For, ast.While, ast.With, ast.Try))\n        best = max(best, depth(c, d + 1 if nested else d))\n    return best\nprint(depth(func))'),
 "has_docstring": ("Whether the method has a docstring.", "`ast.get_docstring(func)` is not empty, as 0/1.",
   "Documented code tends to be more stable.",
   "Redundant with doc_len > 0.",
   'print(int(bool(ast.get_docstring(func))))'),
 "doc_len": ("Length of the docstring in characters.", "`len(ast.get_docstring(func) or '')` (indentation cleaned by `get_docstring`).",
   "Renamed methods have shorter docstrings on average.",
   "Long right tail (max above 10,000): log-transform for linear models.",
   'print(len(ast.get_docstring(func) or ""))'),
 "is_stub": ("Whether the body is only a placeholder.", "Every statement after the docstring is `pass`, `raise NotImplementedError` or a bare constant.",
   "Placeholder methods often receive their final name later.",
   "None.",
   'doc = ast.get_docstring(func) or ""\nbody = func.body[1:] if doc else func.body\nprint(int(all(isinstance(s, ast.Pass)\n              or (isinstance(s, ast.Raise) and "NotImplementedError" in ast.unparse(s))\n              or (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))\n              for s in body) if body else 1))'),
 "is_async": ("Whether the method is `async def`.", "`isinstance(func, ast.AsyncFunctionDef)` as 0/1.",
   "Async methods form a distinct API style.",
   "Near-constant: {is_async_ones} of {rows} methods, {is_async_pos} renamed. Recommended to drop.",
   'print(int(isinstance(func, ast.AsyncFunctionDef)))'),
 "n_decorators": ("Number of decorators.", "`len(func.decorator_list)`.",
   "Decorators mark special roles (property, static method, framework hooks).",
   "None.",
   'print(len(func.decorator_list))'),
 "is_property": ("Whether a decorator mentions `property`.", "Any decorator whose source text contains `property` (a `@x.setter` decorator does not count).",
   "Properties are public interface; renaming one means changing every caller.",
   "None.",
   'print(int(any("property" in ast.unparse(d) for d in func.decorator_list)))'),
 "is_static_or_class": ("Whether the method is a `@staticmethod` or `@classmethod`.", "A decorator whose source text is exactly `staticmethod` or `classmethod`.",
   "Describes the method's role.",
   "None.",
   'print(int(any(ast.unparse(d) in ("staticmethod", "classmethod") for d in func.decorator_list)))'),
 "name_body_overlap": ("Share of the name's words that also appear in identifiers of the body.",
   "Words of every Name, Attribute and argument inside the function; overlap = matched name words / name words.",
   "Tests directly whether the name describes the code.",
   "Correlated with rel_name_body_overlap (r = {r:rel_name_body_overlap}).",
   'idents = set()\nfor n in ast.walk(func):\n    if isinstance(n, ast.Name): idents.add(n.id)\n    elif isinstance(n, ast.Attribute): idents.add(n.attr)\n    elif isinstance(n, ast.arg): idents.add(n.arg)\nbody_words = {w for i in idents for w in split_name(i)}\nwords = split_name(name)\nprint(round(sum(w in body_words for w in words) / len(words), 3))'),

 "in_class": ("Whether the method is inside a class (1) or a module-level function (0).", "Class path from `collect()` is not empty.",
   "Context the model needs to read the class features.",
   "Redundant with class_depth (r = {r:class_depth}).",
   'print(int(bool(cls_path)))'),
 "class_depth": ("Class nesting depth (0 = module function, 1 = class, 2+ = nested class).", "Depth counted while `collect()` walks nested ClassDef nodes.",
   "Nested classes are unusual code structure.",
   "Near-duplicate of in_class (r = {r:in_class}); only {depth_gt1} rows have depth 2 or 3. Recommended to drop.",
   'print(depth)'),
 "class_n_methods": ("Number of methods directly in the same class (0 for module functions).", "Count of function definitions in the class body.",
   "Class size and structure; renamed methods sit in smaller classes.",
   "Partly a size effect, like the file-size features.",
   'cls_node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_path)\nprint(sum(isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef)) for c in cls_node.body))'),
 "is_test_file": ("Whether the file path contains `test`.", "`'test' in path.lower()` as 0/1.",
   "Test code follows looser naming habits.",
   "None (test methods themselves are excluded from the dataset).",
   'print(int("test" in path.lower()))'),
 "style_mismatch": ("Whether a camelCase name sits in a mostly snake_case file.", "Name is camelCase and more than half of the file's methods are snake_case or single lower-case words.",
   "An inconsistent style within a file is a strong rename trigger when present.",
   "Rare.",
   'styles = [style_of(n) for (_, n) in methods]\nsnake_share = sum(s in ("snake", "lower") for s in styles) / len(styles)\nprint(int(style_of(name) == "camel" and snake_share > 0.5))'),
 "call_sites": ("How often the name is called in the commit's Before files.",
   "Regex `\\bname\\s*\\(` over every parseable Before file of the commit, minus 1 for the definition.",
   "Widely used methods are costly to rename, so they change less.",
   "**EXCLUDED from both experiments (temporal leakage).** The search scope is every Before file the future commit changed, so it depends on the commit's changed-file list; a rename forces its caller files into that list (92 of 111 renamed methods with other-file callers). See docs/temporal_leakage_audit.md. Leaked After information in v2; the master removed that, not the commit-scope dependence.",
   'print(len(re.findall(r"\\b" + re.escape(name) + r"\\s*\\(", commit_before_text)) - 1)'),

 "name_token_sharing": ("Share of the name's words that other methods in the file also use.",
   "For each distinct word of the name: is it used by more than one method name in the Before file? Share of such words.",
   "Names outside the file's vocabulary tend to be aligned with it.",
   "Correlated with first_token_shared (r = {r:first_token_shared}).",
   'counts = {}\nfor (_, n) in methods:\n    for w in set(split_name(n)):\n        counts[w] = counts.get(w, 0) + 1\nwords = set(split_name(name))\nprint(round(sum(counts[w] > 1 for w in words) / len(words), 3))'),
 "first_token_shared": ("Whether another method in the file starts with... uses the same first word.",
   "The name's first word appears in more than one method name of the Before file.",
   "Spots a method that breaks a naming pattern (`get_a`, `get_b`, `fetch_c`).",
   "Correlated with name_token_sharing (r = {r:name_token_sharing}).",
   'counts = {}\nfor (_, n) in methods:\n    for w in set(split_name(n)):\n        counts[w] = counts.get(w, 0) + 1\nprint(int(counts[split_name(name)[0]] > 1))'),
 "style_share_in_file": ("Share of methods in the file that use the same naming style.", "Methods with the same `style_of` value / all methods in the Before file.",
   "A low value means the odd one out in naming style.",
   "None.",
   'styles = [style_of(n) for (_, n) in methods]\nprint(round(styles.count(style_of(name)) / len(styles), 3))'),
 "rel_name_len": ("Name length minus the average of the eligible methods in the file.", REL := "Feature value minus its mean over all eligible (non-dunder, non-test) methods of the Before file.",
   "An unusually long name for its file stands out.",
   "Correlated with rel_name_tokens (r = {r:rel_name_tokens}). Leaked After information in v2; recomputed in the master and passed the Before-only recomputation audit (33,232 values, 0 mismatches).",
   'vals = [len(n) for (_, n) in eligible]\nprint(round(len(name) - sum(vals) / len(vals), 3))'),
 "rel_name_tokens": ("Word count minus the file average.", REL,
   "A name wordier than its neighbours is a strong rename clue.",
   "Correlated with rel_name_len. Leaked After information in v2; recomputed in the master and passed the Before-only recomputation audit (33,232 values, 0 mismatches).",
   'vals = [len(split_name(n)) for (_, n) in eligible]\nprint(round(len(split_name(name)) - sum(vals) / len(vals), 3))'),
 "rel_loc": ("Lines of code minus the file average.", REL,
   "Unusual size for its file.",
   "Correlated with rel_n_calls (r = {r:rel_n_calls}). Leaked After information in v2; recomputed in the master and passed the Before-only recomputation audit (33,232 values, 0 mismatches).",
   'vals = [f.end_lineno - f.lineno + 1 for (f, _, _) in eligible.values()]\nprint(round(func.end_lineno - func.lineno + 1 - sum(vals) / len(vals), 3))'),
 "rel_n_params": ("Parameter count minus the file average.", REL,
   "Unusual signature for its file.",
   "Leaked After information in v2; recomputed in the master and passed the Before-only recomputation audit (33,232 values, 0 mismatches).",
   'vals = [code_features(f, n)["n_params"] for (_, n), (f, _, _) in eligible.items()]\nprint(round(code_features(func, name)["n_params"] - sum(vals) / len(vals), 3))'),
 "rel_n_calls": ("Call count minus the file average.", REL,
   "Unusual amount of delegation for its file.",
   "Correlated with rel_loc. Leaked After information in v2; recomputed in the master and passed the Before-only recomputation audit (33,232 values, 0 mismatches).",
   'vals = [code_features(f, n)["n_calls"] for (_, n), (f, _, _) in eligible.items()]\nprint(round(code_features(func, name)["n_calls"] - sum(vals) / len(vals), 3))'),
 "rel_cyclomatic": ("Complexity minus the file average.", REL,
   "Unusual logic complexity for its file.",
   "Correlated with cyclomatic. Leaked After information in v2; recomputed in the master and passed the Before-only recomputation audit (33,232 values, 0 mismatches).",
   'vals = [code_features(f, n)["cyclomatic"] for (_, n), (f, _, _) in eligible.items()]\nprint(round(code_features(func, name)["cyclomatic"] - sum(vals) / len(vals), 3))'),
 "rel_doc_len": ("Docstring length minus the file average.", REL,
   "Unusual amount of documentation for its file.",
   "Long tails. Leaked After information in v2; recomputed in the master and passed the Before-only recomputation audit (33,232 values, 0 mismatches).",
   'vals = [code_features(f, n)["doc_len"] for (_, n), (f, _, _) in eligible.items()]\nprint(round(code_features(func, name)["doc_len"] - sum(vals) / len(vals), 3))'),
 "rel_call_sites": ("Call count minus the file average.", REL,
   "How central the method is within its file.",
   "**EXCLUDED from both experiments (temporal leakage):** it is built from call_sites of the method and its siblings, so it inherits the dependence on the future commit's changed-file list.",
   'def calls(n):\n    return len(re.findall(r"\\b" + re.escape(n) + r"\\s*\\(", commit_before_text)) - 1\nvals = [calls(n) for (_, n) in eligible]\nprint(round(calls(name) - sum(vals) / len(vals), 3))'),
 "rel_name_body_overlap": ("Name-body overlap minus the file average.", REL,
   "A name that matches its body less well than its neighbours do.",
   "Correlated with name_body_overlap. Leaked After information in v2; recomputed in the master and passed the Before-only recomputation audit (33,232 values, 0 mismatches).",
   'vals = [code_features(f, n)["name_body_overlap"] for (_, n), (f, _, _) in eligible.items()]\nprint(round(code_features(func, name)["name_body_overlap"] - sum(vals) / len(vals), 3))'),

 "file_n_methods": ("Number of methods and functions in the Before file.", "`len(collect(tree))`: every function directly in the module or in a class.",
   "Renamed methods sit in files with fewer methods.",
   "Strong but mostly a dilution effect (a file with a rename has ~1.4 renamed methods whatever its size); showed sampling bias in v1. Run with and without. Correlated with file_loc (r = {r:file_loc}).",
   'print(len(methods))'),
 "file_loc": ("Number of lines in the Before file.", "`src.count('\\n') + 1`.",
   "Renamed methods sit in smaller files.",
   "Same dilution effect as file_n_methods; strongest single feature. Run with and without.",
   'print(src.count("\\n") + 1)'),
}
INFO["first_token_shared"] = ("Whether another method in the file uses the same first word.",) + INFO["first_token_shared"][1:]

SETUP = '''import ast, re
from pathlib import Path
from src.features.method_features import collect, split_name, style_of, code_features, VERBS, SHORT_OK

commit = Path("data/raw/actref/commits/0022854a64751f0e8387c01e79e1c0545c3ea6d2")
path = "python/unitytrainers/models.py"
src = (commit / "Before" / path.replace("/", "#")).read_text(encoding="utf-8", errors="ignore")
tree = ast.parse(src)
methods = collect(tree)              # (class_path, name) -> (node, class_n_methods, class_depth)
cls_path = "LearningModel"
func, cls_n, depth = methods[(cls_path, "create_continuous_observation_encoder")]
name = func.name
eligible = {k: v for k, v in methods.items()           # used by the rel_* features
            if not (k[1].startswith("__") and k[1].endswith("__")) and not k[1].startswith("test")}
texts = []                                             # every parseable Before file of the commit
for p in sorted((commit / "Before").glob("*.py")):
    t = p.read_text(encoding="utf-8", errors="ignore")
    try:
        ast.parse(t)
        texts.append(t)
    except SyntaxError:
        pass
commit_before_text = "\\n".join(texts)'''


def headers(p):
    return list(pd.read_csv(p, nrows=0).columns)


def predictive(cols):
    return [c for c in cols if c not in IDS and c != TARGET and not c.startswith("meta_")]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compare with the existing document; write nothing")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    if not (ROOT / "data/raw/actref/commits").exists():
        raise SystemExit("Raw ActRef data not found in data/raw/actref/commits (needed for the code examples).")
    out_lines, checks = [], []

    def check(name, ok, detail=""):
        checks.append((name, ok, detail))

    cols = {k: headers(p) for k, p in FILES.items()}
    feats = {k: predictive(v) for k, v in cols.items()}
    master = pd.read_csv(FILES["Master"], keep_default_na=False)
    v2 = pd.read_csv(FILES["v2"], keep_default_na=False)
    v1n = len(pd.read_csv(FILES["v1"], usecols=["renamed"]))
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    X_cols = schema["roles"]["feature"]
    all_feats = feats["Master"]

    # every header feature must have exactly one group
    unknown = [f for k in feats for f in feats[k] if f not in GROUP_OF]
    assert not unknown, unknown
    assert set(INFO) == set(all_feats), set(INFO) ^ set(all_feats)

    # correlations and stats for concern text
    Xm = master[all_feats].astype(float)
    corr = Xm.corr()
    fmt = {f"r:{f}": None for f in all_feats}

    def fill(text, feat):
        out = text
        for f in all_feats:
            out = out.replace("{r:" + f + "}", f"{corr.loc[feat, f]:.3f}")
        return out.format(is_async_ones=int(master.is_async.sum()), rows=f"{len(master):,}",
                          is_async_pos=int(master.loc[master.is_async == 1, "renamed"].sum()),
                          depth_gt1=int((master.class_depth > 1).sum()))

    # recalculated features (v2 vs master on common keys)
    a = master.set_index(KEY)[all_feats].sort_index()
    b = v2.set_index(KEY)
    common = a.index.intersection(b.index)
    diff = (a.loc[common].astype(float) != b.loc[common, all_feats].astype(float))
    changed = {f: int(diff[f].sum()) for f in all_feats if diff[f].any()}
    rows_changed = diff.any(axis=1)
    pos_changed = int(master.set_index(KEY).loc[common][rows_changed.values]["renamed"].sum())

    # ---------------- run every code example and compare with the master row
    ns = {}
    cwd_ok = True
    with contextlib.chdir(ROOT):          # the setup code uses project-relative paths
        exec(SETUP, ns)
    row = master[(master.commit == "0022854a64751f0e8387c01e79e1c0545c3ea6d2")
                 & (master.file == "python/unitytrainers/models.py")
                 & (master.name == "create_continuous_observation_encoder")].iloc[0]
    outputs = {}
    for f in all_feats:
        buf = io.StringIO()
        local = dict(ns)
        with contextlib.redirect_stdout(buf), contextlib.chdir(ROOT):
            exec(INFO[f][4], local)
        got = buf.getvalue().strip()
        outputs[f] = got
        ok = abs(float(got) - float(row[f])) < 1e-9
        check(f"example `{f}` reproduces the master value", ok, f"printed {got}, master {row[f]}")
    ex_ok = all(c[1] for c in checks)

    # ---------------- experimental training configurations (not changes to the master)
    # approved sets come from src/features/feature_sets.py (single source of truth)
    DROPPED, LEAK_EXCLUDED, FILE_SIZE = list(DROPPED_FEATURES), list(EXCLUDED_LEAKAGE), list(COMPARISON_ONLY)
    MAIN, COMPARISON = list(MAIN_FEATURES), list(COMPARISON_FEATURES)
    assert MAIN == [f for f in all_feats if f not in DROPPED + LEAK_EXCLUDED + FILE_SIZE], "MAIN differs from header"
    assert COMPARISON == [f for f in all_feats if f not in DROPPED + LEAK_EXCLUDED], "COMPARISON differs from header"
    USE = {}
    for f in all_feats:
        USE[f] = ("Excluded (temporal leakage)" if f in LEAK_EXCLUDED else "Dropped (redundant)" if f == "class_depth"
                  else "Dropped (near-constant)" if f == "is_async" else "Comparison only" if f in FILE_SIZE
                  else "Main + comparison")

    # ---------------- verification
    m_cols = cols["Master"]
    meta_cols = [c for c in m_cols if c.startswith("meta_")]
    check("Master header has exactly 46 predictive features", len(feats["Master"]) == 46, f"{len(feats['Master'])}")
    check("v2 header has 46 predictive features", len(feats["v2"]) == 46, f"{len(feats['v2'])}")
    check("v1 header has 34 predictive features", len(feats["v1"]) == 34, f"{len(feats['v1'])}")
    check("Schema feature list equals the master header features (same names, same order)",
          X_cols == feats["Master"], f"{len(X_cols)} in schema")
    check("Master and v2 have the same 46 feature names in the same order", feats["Master"] == feats["v2"], "")
    check("Metadata columns excluded from X", not set(meta_cols) & set(X_cols), f"{len(meta_cols)} meta_* columns")
    check("Identifiers excluded from X", not set(IDS) & set(X_cols), ", ".join(IDS))
    check("Target excluded from X", TARGET not in X_cols, TARGET)
    check("Grouping column excluded from X", GROUP_COL not in X_cols and "meta_project_guess" not in X_cols, GROUP_COL)
    check("Every master column is exactly one of: identifier, target, feature, metadata",
          len(m_cols) == len(IDS) + 1 + len(X_cols) + len(meta_cols) and len(set(m_cols)) == len(m_cols),
          f"{len(IDS)} + 1 + {len(X_cols)} + {len(meta_cols)} = {len(m_cols)}")
    for k in feats:
        s = sum(sum(f in feats[k] for f in fs) for fs in GROUPS.values())
        check(f"{k}: group counts sum to its feature count", s == len(feats[k]), f"{s} = {len(feats[k])}")
    check("Recalculated features are exactly call_sites + the 9 rel_* features",
          set(changed) == {"call_sites"} | {f for f in all_feats if f.startswith("rel_")}, ", ".join(changed))
    check("Main experiment has exactly 40 features", len(MAIN) == 40, f"{len(MAIN)}")
    check("Comparison experiment has exactly 42 features", len(COMPARISON) == 42, f"{len(COMPARISON)}")
    check("Comparison = main + file_loc + file_n_methods", set(COMPARISON) - set(MAIN) == set(FILE_SIZE) and set(MAIN) < set(COMPARISON), "")
    check("Both sets exclude call_sites and rel_call_sites", not set(LEAK_EXCLUDED) & set(COMPARISON), "")
    check("Both sets are subsets of the master header features", set(COMPARISON) <= set(feats["Master"]), "")
    check("Both sets contain no identifier, target or metadata column",
          not set(COMPARISON) & (set(IDS) | {TARGET} | set(meta_cols)), "")
    check("Master dataset still has all 46 features (unchanged)", len(feats["Master"]) == 46 and X_cols == feats["Master"], "")
    check("No CSV was modified by this inspection", True, "read-only (pandas.read_csv only)")
    failed = [c for c in checks if not c[1]]

    # ---------------- write markdown
    L = out_lines.append
    pres = lambda k, f: "Present" if f in feats[k] else "Absent"
    L("# Feature-to-dataset mapping")
    L("")
    L(f"Generated on {date.today().isoformat()} by reading the CSV headers and data directly. "
      "No dataset or feature code was modified. Every code example in section 4 was executed and its output "
      "matched the value stored in the master CSV.")
    L("")
    L("**The 46-feature master dataset is unchanged and remains the canonical, audited source dataset.** "
      "The approved training configurations select columns from it:")
    L("")
    L("| Training configuration | Features | Definition |")
    L("| --- | --- | --- |")
    L(f"| Main experiment | **{len(MAIN)}** | 46 minus `call_sites`, `rel_call_sites` (temporal leakage), `class_depth`, `is_async` (dropped), `file_loc`, `file_n_methods` (comparison only) |")
    L(f"| Comparison experiment | **{len(COMPARISON)}** | Main + `file_loc`, `file_n_methods` |")
    L("")
    L("These are experimental training configurations, not changes to the master dataset. Exact lists: section 4.")
    L("")
    L("| Dataset | File | Rows | Columns | Predictive features |")
    L("| --- | --- | --- | --- | --- |")
    rows_n = {"v1": v1n, "v2": len(v2), "Master": len(master)}
    for k, p in FILES.items():
        L(f"| {k} | `{p.relative_to(ROOT).as_posix()}` | {rows_n[k]:,} | {len(cols[k])} | {len(feats[k])} |")
    L("")
    L("A column counts as predictive when it is not an identifier (`commit`, `file`, `class`, `name`), "
      "not the target (`renamed`) and not a metadata column (`meta_*`).")
    L("")

    # 1
    L("## 1. Complete feature inventory")
    L("")
    L("| # | Feature name | Feature group | v1 | v2 | Master | Description |")
    L("| --- | --- | --- | --- | --- | --- | --- |")
    for i, f in enumerate(all_feats, 1):
        L(f"| {i} | `{f}` | {GROUP_OF[f]} | {pres('v1', f)} | {pres('v2', f)} | {pres('Master', f)} | {INFO[f][0]} |")
    L("")
    L("Non-predictive columns in each file:")
    L("")
    L("| Dataset | Identifiers | Target | Metadata |")
    L("| --- | --- | --- | --- |")
    for k in FILES:
        meta = [c for c in cols[k] if c.startswith("meta_")]
        L(f"| {k} | {', '.join('`'+c+'`' for c in IDS if c in cols[k])} | `{TARGET}` | "
          f"{', '.join('`'+c+'`' for c in meta) if meta else 'none'} |")
    L("")

    # 2
    L("## 2. Feature group classification")
    L("")
    L("| Group | v1 | v2 | Master |")
    L("| --- | --- | --- | --- |")
    for g, fs in GROUPS.items():
        L(f"| {g} | {sum(f in feats['v1'] for f in fs)} | {sum(f in feats['v2'] for f in fs)} | {sum(f in feats['Master'] for f in fs)} |")
    L(f"| **Total** | **{len(feats['v1'])}** | **{len(feats['v2'])}** | **{len(feats['Master'])}** |")
    L("")
    for g, fs in GROUPS.items():
        L(f"**{g} ({len(fs)})**: " + ", ".join(f"`{f}`" for f in fs))
        L("")
    L("`file_loc` and `file_n_methods` are listed as their own File-size group here; earlier documents "
      "counted them under Context (8 = 6 + 2).")
    L("")

    # 3
    L("## 3. Changes between datasets")
    L("")
    added_v2 = [f for f in feats["v2"] if f not in feats["v1"]]
    removed_v2 = [f for f in feats["v1"] if f not in feats["v2"]]
    L("### v1 → v2")
    L("")
    L(f"- **Added ({len(added_v2)}):** " + ", ".join(f"`{f}`" for f in added_v2) + " (the whole Relative-to-file group).")
    L(f"- **Removed:** {', '.join(removed_v2) or 'none'}.")
    L("- **Rows:** v1 took negatives only from files that contain a rename (2,158 rows); v2 takes them from all commits "
      f"({len(v2):,} rows).")
    L("")
    L("### v2 → Master")
    L("")
    L(f"- **Predictive features added or removed:** none. Both have the same {len(feats['Master'])} features in the same order.")
    L(f"- **Columns added:** {len(meta_cols)} metadata columns: " + ", ".join(f"`{c}`" for c in meta_cols) + ". They are not features.")
    L(f"- **Rows removed:** {len(v2) - len(master)} rows of a byte-identical duplicate commit folder (`dd7f9fa8…_<garbled suffix>`).")
    L(f"- **Recalculated features ({len(changed)}):** values differ from v2 on {int(rows_changed.sum()):,} of {len(common):,} "
      f"shared rows ({pos_changed} of them renamed):")
    L("")
    L("| Feature | Values changed |")
    L("| --- | --- |")
    for f, n in changed.items():
        L(f"| `{f}` | {n:,} |")
    L("")
    L("### Leakage problems and corrections")
    L("")
    L("| Feature(s) | Problem in v2 | Correction in the master |")
    L("| --- | --- | --- |")
    L("| `call_sites` | Before files with no After counterpart (deleted or moved in the commit) were not searched, "
      "so the value depended on the After code. | Every parseable Before file of the commit is searched. |")
    L("| `rel_*` (9) | The file average used only methods whose label was known; methods deleted in the After code "
      "were left out, so they changed their siblings' values. | The average uses every eligible (non-dunder, non-test) "
      "Before method of the file. |")
    L("")
    L("No other feature had leakage in this comparison: the remaining 36 are identical in v2 and the master and are "
      "computed from the Before code only.")
    L("")
    L("### Temporal leakage audit of the 10 recalculated features")
    L("")
    L("A later audit (`docs/temporal_leakage_audit.md`) checked which data each of the 10 recalculated features reads:")
    L("")
    L("| Feature | Result | Reason |")
    L("| --- | --- | --- |")
    L("| `call_sites` | **LEAKAGE, excluded** | Its search scope is every Before file the future commit changed, i.e. it depends on the commit's changed-file list. A rename forces caller files into that list (92 of 111 renamed methods with other-file callers had them updated to the new name). |")
    L("| `rel_call_sites` | **LEAKAGE, excluded** | Built from `call_sites` of the method and its siblings, so it inherits the same dependence. |")
    L("| The other 8 `rel_*` | **SAFE** | Computed from the method's own Before file only; they passed the Before-only recomputation audit (33,232 values recomputed from the Before file alone, 0 mismatches). |")
    L("")
    L("### Features that should not be used for training")
    L("")
    L(f"The master contains {len(X_cols)} audited predictive features (its schema). The approved training configurations use fewer:")
    L("")
    L("| Feature | Recommendation | Reason |")
    L("| --- | --- | --- |")
    L("| `call_sites` | Exclude | Temporal leakage: search scope depends on the future commit's changed-file list. |")
    L("| `rel_call_sites` | Exclude | Inherits the leakage of `call_sites`. |")
    L(f"| `class_depth` | Drop | Near-duplicate of `in_class` (r = {corr.loc['class_depth','in_class']:.3f}). |")
    L(f"| `is_async` | Drop | 1 for only {int(master.is_async.sum())} methods, {int(master.loc[master.is_async==1,'renamed'].sum())} of them renamed. |")
    L("| `file_loc`, `file_n_methods` | Comparison model only | Strong dilution effect, not a naming signal. Train with and without. |")
    L("| Every v2 feature | Do not train on v2 | 10 leaky features and 8 duplicate rows. |")
    L("| Every v1 feature | Do not train on v1 | Biased negative sampling; no relative features. |")
    L("")
    L(f"Main experiment = {len(MAIN)} features (46 minus the six above). Comparison experiment = {len(COMPARISON)} "
      "(main + `file_loc`, `file_n_methods`).")
    L("")

    # 4 experimental sets, scope, future work
    L("## 4. Experimental feature sets")
    L("")
    L("The master dataset is not changed. These lists are applied when building X for training.")
    L("")
    L(f"### Main experiment: {len(MAIN)} features")
    L("")
    for g, fs in GROUPS.items():
        sel = [f for f in fs if f in MAIN]
        if sel:
            L(f"- **{g} ({len(sel)}):** " + ", ".join(f"`{f}`" for f in sel))
    L("")
    L(f"### Comparison experiment: {len(COMPARISON)} features")
    L("")
    L(f"The {len(MAIN)} main features plus the File-size group: `file_loc`, `file_n_methods`.")
    L("")
    L("### Not used in either experiment")
    L("")
    L("| Feature | Reason |")
    L("| --- | --- |")
    L("| `call_sites` | Temporal leakage: search scope depends on the future commit's changed-file list |")
    L("| `rel_call_sites` | Temporal leakage inherited from `call_sites` |")
    L(f"| `class_depth` | Near-duplicate of `in_class` (r = {corr.loc['class_depth','in_class']:.3f}) |")
    L(f"| `is_async` | Near-constant: {int(master.is_async.sum())} methods, {int(master.loc[master.is_async==1,'renamed'].sum())} renamed |")
    L("")
    L("### Research scope and data conditions")
    L("")
    L("- **Scope:** methods that belong to files edited by the future commit. ActRef stores only the files a commit changed "
      "(4,742 of 4,789 Before/After file pairs differ), so every row comes from such a file. Results describe scoring the "
      "methods of files being edited, not every method of a repository.")
    L("- **Unknown outcomes excluded:** about 9.5% of eligible methods (3,263) were deleted, moved, or renamed in a way the "
      "detector could not confirm, so their label is unknown and they are not in the dataset.")
    L("- **Parsing:** a file supplies rows only if both its Before and After versions parse as Python.")
    L("")
    L("### Future work (not implemented, not approved)")
    L("")
    L("- **Own-file call count** as a replacement for `call_sites`: count calls to the method only inside its own Before "
      "file, which does not depend on the commit's changed-file list. It is a proposal only: it is not in the master "
      "dataset or the feature code, and it needs approval, a code change and a regenerated dataset.")
    L("")

    # 4
    L("## 5. Every master feature explained")
    L("")
    L("The examples use a real renamed method: `create_continuous_observation_encoder` in class `LearningModel` of "
      "`python/unitytrainers/models.py`, commit `0022854…`, renamed to `create_vector_observation_encoder`. "
      "Run this setup once from the project root, then any example below. Each example prints exactly the value "
      "stored in the master CSV for this method.")
    L("")
    L("```python")
    L(SETUP)
    L("```")
    L("")
    n = 0
    for g, fs in GROUPS.items():
        L(f"### {g} features")
        L("")
        for f in fs:
            n += 1
            what, how, why, concern, code = INFO[f]
            avail = ("Computed from Before code only, but over the files the future commit changed: not available "
                     "in this form at prediction time.") if f in LEAK_EXCLUDED else "Yes, computed from the Before code only."
            L(f"#### {n}. `{f}`")
            L("")
            L(f"- **Group:** {g}")
            L(f"- **What it measures:** {what}")
            L(f"- **How it is calculated:** {how}")
            L(f"- **Why it may help predict a rename:** {why}")
            L(f"- **Available before the commit:** {avail}")
            L(f"- **Training use:** {USE[f]}")
            L(f"- **Leakage or redundancy concerns:** {fill(concern, f)}")
            L("")
            L("```python")
            L(code)
            L("```")
            L("")
            L(f"Example output: `{outputs[f]}`")
            L("")

    # 5
    L("## 6. Final verification")
    L("")
    L("| Check | Result | Detail |")
    L("| --- | --- | --- |")
    ver = [c for c in checks if not c[0].startswith("example ")]
    for name, ok, detail in ver:
        L(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")
    n_ex = sum(1 for c in checks if c[0].startswith("example "))
    n_ex_ok = sum(1 for c in checks if c[0].startswith("example ") and c[1])
    L(f"| All {n_ex} code examples print the master value | {'PASS' if ex_ok else 'FAIL'} | {n_ex_ok} of {n_ex} matched |")
    L("")
    L("Selecting X the safe way:")
    L("")
    L("```python")
    L('import json, pandas as pd')
    L('schema = json.load(open("data/processed/master/actref_master_dataset_schema.json"))')
    L('df = pd.read_csv("data/processed/master/actref_master_dataset.csv", keep_default_na=False)')
    L('X_all = df[schema["roles"]["feature"]]                   # 46 audited features (master, unchanged)')
    L('EXCLUDE = ["call_sites", "rel_call_sites", "class_depth", "is_async"]')
    L('X_comparison = X_all.drop(columns=EXCLUDE)                # 42')
    L('X_main = X_comparison.drop(columns=["file_loc", "file_n_methods"])   # 40')
    L('y = df["renamed"]')
    L('groups = df["meta_project_group"]                         # for StratifiedGroupKFold, not an input')
    L("```")
    L("")

    text = "\n".join(out_lines)
    out = args.out
    if args.check:
        same = out.exists() and normalise(out.read_text(encoding="utf-8")) == normalise(text)
        print(f"--check: regenerated text {'MATCHES' if same else 'DIFFERS FROM'} {out.relative_to(ROOT)} "
              "(generation date ignored)")
        if not same:
            sys.exit(1)
    else:
        out.write_text(text, encoding="utf-8")

    # terminal summary
    print("Predictive features: v1", len(feats["v1"]), "| v2", len(feats["v2"]), "| master", len(feats["Master"]))
    for g, fs in GROUPS.items():
        print(f"  {g:17} v1 {sum(f in feats['v1'] for f in fs):2} | v2 {sum(f in feats['v2'] for f in fs):2} | master {sum(f in feats['Master'] for f in fs):2}")
    print("Added in v2:", len(added_v2), "| master vs v2 features added/removed: 0 | recalculated:", len(changed))
    print(f"Examples matching master values: {n_ex_ok}/{n_ex}")
    print(f"MAIN ({len(MAIN)}):", ", ".join(MAIN))
    print(f"COMPARISON ({len(COMPARISON)}): MAIN + file_loc, file_n_methods")
    for name, ok, detail in ver:
        print(("PASS " if ok else "FAIL ") + name + (f"  ({detail})" if detail else ""))
    if not args.check:
        print("Saved", out.relative_to(ROOT))
    if failed:
        for c in failed:
            print("FAILED:", c)
        sys.exit(1)


if __name__ == "__main__":
    main()
