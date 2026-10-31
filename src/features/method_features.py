"""
Method-level feature helpers.

The functions below are copied unchanged from the original v2 builder
(archive/legacy_scripts/Build_Datasetv2.py, lines 30-180) so that the
regenerated dataset is identical to data/processed/validated/train_features_v2.csv.
Do not change their behaviour without regenerating and re-validating the dataset.

Every function here receives only Before-commit code (an AST node, a name,
or Before file text). Nothing from the After version is used for features.
"""
import ast
import re
import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)

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
    """Read and parse a file. Returns (None, None) if the file is missing
    (e.g. deleted or moved in the commit) or is not valid Python."""
    if not path.exists():
        return None, None
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return text, ast.parse(text)
    except (SyntaxError, ValueError, OSError, RecursionError):
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


REL_COLS = ["name_len", "name_tokens", "loc", "n_params", "n_calls", "cyclomatic",
            "doc_len", "call_sites", "name_body_overlap"]

