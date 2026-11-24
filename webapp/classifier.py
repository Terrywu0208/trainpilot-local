"""Local, bounded text-classification workflow. No LLM or external execution."""
from __future__ import annotations

import csv
import io
import json
import re
import uuid
from collections import Counter
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

MAX_BYTES = 5_000_000
MAX_ROWS = 10_000
ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")


def _folder(root: Path, run_id: str) -> Path:
    if not ID_PATTERN.fullmatch(run_id):
        raise ValueError("Invalid experiment ID.")
    folder = root / run_id
    if not folder.is_dir():
        raise ValueError("Experiment not found. Upload your CSV again.")
    return folder


def parse_csv(content: str) -> tuple[list[str], list[dict]]:
    if len(content.encode("utf-8")) > MAX_BYTES:
        raise ValueError("CSV must be at most 5 MB.")
    try:
        reader = csv.DictReader(io.StringIO(content.lstrip("\ufeff")))
        fields = reader.fieldnames
        if not fields:
            raise ValueError("CSV needs a header with column names.")
        named = [name for name in fields if name and name.strip()]
        if len(named) != len(set(named)):
            raise ValueError("CSV needs unique column names.")
        used = set(named)
        generated = []
        normalized = []
        for index, name in enumerate(fields, 1):
            if not name or not name.strip():
                name = f'unnamed_column_{index}'
                while name in used:
                    name += '_'
                used.add(name)
                generated.append(name)
            normalized.append(name)
        fields = normalized
        reader.fieldnames = fields
        rows = []
        for row in reader:
            if None in row:
                raise ValueError("A row has more fields than the header.")
            if any((v or "").strip() for v in row.values()):
                rows.append({k: (v or "").strip() for k, v in row.items()})
            if len(rows) > MAX_ROWS:
                raise ValueError("CSV must have at most 10,000 rows.")
    except csv.Error as exc:
        raise ValueError(f"Could not parse CSV: {exc}") from exc
    if not rows:
        raise ValueError("The CSV has no data rows.")
    empty_generated = {name for name in generated if not any(row[name] for row in rows)}
    if empty_generated:
        fields = [name for name in fields if name not in empty_generated]
        rows = [{key: value for key, value in row.items() if key not in empty_generated} for row in rows]
    if not fields:
        raise ValueError("CSV needs at least one non-empty column.")
    return fields, rows


def inspect_csv(root: Path, content: str, goal: str = "") -> dict:
    if len(goal) > 500:
        raise ValueError("The goal must be at most 500 characters.")
    fields, rows = parse_csv(content)
    run_id = uuid.uuid4().hex
    folder = root / run_id
    folder.mkdir(parents=True)
    (folder / "input.csv").write_text(content, encoding="utf-8")
    (folder / "goal.txt").write_text(goal.strip(), encoding="utf-8")
    columns = [{"name": f, "nonempty": sum(bool(r[f]) for r in rows),
                "unique": len({r[f] for r in rows if r[f]})} for f in fields]
    return {"id": run_id, "goal": goal.strip(), "rows": len(rows), "columns": columns,
            "examples": [{f: r[f][:160] for f in fields} for r in rows[:3]],
            "message": "Review the columns and samples, then select the text and label columns. Suggestions are based on statistics and rules."}


def _load(root: Path, run_id: str):
    folder = _folder(root, run_id)
    fields, rows = parse_csv((folder / "input.csv").read_text(encoding="utf-8"))
    return folder, fields, rows


def plan(root: Path, run_id: str, text_column: str, label_column: str) -> dict:
    folder, fields, rows = _load(root, run_id)
    if text_column not in fields or label_column not in fields or text_column == label_column:
        raise ValueError("Select distinct, existing text and label columns.")
    valid = [(r[text_column], r[label_column]) for r in rows if r[text_column] and r[label_column]]
    counts = Counter(y for _, y in valid)
    warnings = []
    if len(valid) < 30:
        warnings.append("At least 30 valid labeled rows are required.")
    if not 2 <= len(counts) <= 20:
        warnings.append("Provide between 2 and 20 classes.")
    if counts and min(counts.values()) < 5:
        warnings.append("Each class needs at least 5 valid rows for training and evaluation.")
    if valid and len(counts) >= 2 and max(counts.values()) / min(counts.values()) >= 3:
        warnings.append("Class distribution is imbalanced; standard and class-weighted models will be compared.")
    seen = {}
    duplicates = 0
    for text, label in valid:
        key = " ".join(text.split()).casefold()
        if key in seen:
            if seen[key] != label:
                warnings.append("Identical text has conflicting labels; inspect and correct it first.")
                break
            duplicates += 1
        seen[key] = label
    if duplicates:
        warnings.append(f"Found {duplicates} duplicate texts; duplicates are removed before training to reduce leakage.")
    if len(seen) < 30 or (seen and min(Counter(seen.values()).values()) < 5):
        warnings.append("After deduplication at least 30 rows and 5 per class are required.")
    blocking = len(valid) < 30 or not 2 <= len(counts) <= 20 or len(seen) < 30 or (seen and min(Counter(seen.values()).values()) < 5) or any("conflicting labels" in w for w in warnings)
    return {"id": run_id, "goal": (folder / "goal.txt").read_text(encoding="utf-8"), "text_column": text_column, "label_column": label_column,
            "valid_rows": len(valid), "missing_rows": len(rows)-len(valid),
            "unique_rows": len(seen), "duplicates": duplicates, "class_counts": dict(counts),
            "warnings": warnings, "ready": not blocking,
            "recommendation": "Start with a character TF-IDF and Logistic Regression baseline on the local CPU; consider more data or fine-tuning based on validation.",
            "split": "Stratified split: 60% training, 20% validation, 20% test; select on validation and evaluate test once."}


def train(root: Path, run_id: str, text_column: str, label_column: str) -> dict:
    import joblib
    from sklearn.dummy import DummyClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline

    proposal = plan(root, run_id, text_column, label_column)
    if not proposal["ready"]:
        raise ValueError(" ".join(proposal["warnings"]))
    folder, _, rows = _load(root, run_id)
    if (folder / "result.json").exists():
        raise ValueError("This experiment is complete; upload data again to create a separate experiment.")
    unique = {}
    for row in rows:
        x, y = row[text_column], row[label_column]
        if x and y:
            unique.setdefault(" ".join(x.split()).casefold(), (x, y))
    values = list(unique.values())
    x, y = zip(*values)
    try:
        x_train, x_test, y_train, y_test = train_test_split(list(x), list(y), test_size=0.2, random_state=42, stratify=y)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42, stratify=y_train)
    except ValueError as exc:
        raise ValueError(f"Insufficient data for a stratified split; add examples in each class: {exc}") from exc
    attempts = []
    models = []
    for balanced in (False, True):
        model = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=30000),
                              LogisticRegression(max_iter=300, class_weight="balanced" if balanced else None, random_state=42))
        model.fit(x_train, y_train)
        score = f1_score(y_val, model.predict(x_val), average="macro", zero_division=0)
        attempts.append({"model": "balanced" if balanced else "standard", "validation_macro_f1": round(float(score), 4)})
        models.append(model)
    selected = max(range(len(attempts)), key=lambda i: attempts[i]["validation_macro_f1"])
    model = models[selected]
    dummy = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
    predictions = model.predict(x_test)
    labels = sorted(set(y))
    metrics = {"accuracy": float(accuracy_score(y_test, predictions)),
               "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
               "baseline_accuracy": float(accuracy_score(y_test, dummy.predict(x_test))),
               "per_class": classification_report(y_test, predictions, labels=labels, output_dict=True, zero_division=0),
               "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist(), "labels": labels}
    artifact = folder / "classifier.joblib"
    joblib.dump(model, artifact)
    reloaded = joblib.load(artifact)
    verified = bool((reloaded.predict(x_test) == predictions).all())
    if not verified:
        raise RuntimeError("Predictions differ after saving and reloading the model.")
    errors = [{"text": a[:200], "expected": b, "predicted": c} for a, b, c in zip(x_test, y_test, predictions) if b != c][:5]
    suggestions = []
    if proposal["warnings"]:
        suggestions.extend(proposal["warnings"])
    if metrics["macro_f1"] < 0.8:
        suggestions.append("Review errors and label consistency; add diverse examples for weaker classes and evaluate fine-tuning separately.")
    else:
        suggestions.append("Keep independent data for continued evaluation; compare fine-tuning costs and benefits on new validation data.")
    weakest = min(labels, key=lambda name: metrics["per_class"][name]["recall"])
    suggestions.append(f"The lowest-recall class on the test set is {weakest}; review its errors and labels and collect representative data.")
    result = {"id": run_id, "status": "completed", "model": attempts[selected]["model"],
              "training_type": "real local scikit-learn classifier", "attempts": attempts,
              "split_counts": {"train": len(x_train), "validation": len(x_val), "test": len(x_test)},
              "metrics": metrics, "errors": errors, "suggestions": suggestions,
              "checkpoint_reloaded": verified, "proposal": proposal}
    (folder / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def predict(root: Path, run_id: str, text: str) -> dict:
    import joblib
    folder = _folder(root, run_id)
    if not text.strip() or len(text) > 5000:
        raise ValueError("Enter 1 to 5,000 characters.")
    if not (folder / "result.json").is_file():
        raise ValueError("Complete training first.")
    model = joblib.load(folder / "classifier.joblib")
    probabilities = model.predict_proba([text])[0]
    return {"label": str(model.predict([text])[0]),
            "scores": {str(k): round(float(v), 4) for k, v in zip(model.classes_, probabilities)},
            "note": "Scores are model estimates and are not guaranteed to be calibrated or correct."}


def report(root: Path, run_id: str) -> str:
    folder = _folder(root, run_id)
    if not (folder / "result.json").is_file():
        raise ValueError("Complete training first.")
    r = json.loads((folder / "result.json").read_text(encoding="utf-8"))
    m = r["metrics"]
    lines = ["# Local classification research report", "", f"Experiment: {run_id}",
             "## Goal", r['proposal']['goal'] or "Not provided", "", "## Data and method", f"User-provided CSV; input column: {r['proposal']['text_column']}; label column: {r['proposal']['label_column']}.",
             f"Valid rows: {r['proposal']['valid_rows']}; unique rows: {r['proposal']['unique_rows']}; missing rows: {r['proposal']['missing_rows']}.",
             f"Class counts before deduplication: {r['proposal']['class_counts']}",
             "Training: TF-IDF character n-grams + logistic regression; CPU, scikit-learn.",
             f"Selection: {r['model']}; seed 42; selected using validation macro F1.",
             f"Validation comparisons: {r['attempts']}", f"Split: {r['split_counts']}", "", "## Held-out results", f"Held-out test accuracy: {m['accuracy']:.4f}",
             f"Held-out test macro F1: {m['macro_f1']:.4f}", f"Majority-class baseline accuracy: {m['baseline_accuracy']:.4f}",
             f"Checkpoint reloaded and predictions matched: {r['checkpoint_reloaded']}",
             f"Per-class metrics: { {k: m['per_class'][k] for k in m['labels']} }",
             f"Confusion matrix (actual rows, predicted columns; labels {m['labels']}): {m['confusion_matrix']}",
             f"Up to five errors: {r['errors']}",
             "", "## Improvement suggestions", *[f"- {s}" for s in r['suggestions']],
             "", "## Limitations", "- Metrics depend on this dataset and split; they do not prove production performance.",
             "- No LLM reasoning or external dataset suitability verification was performed.",
             "- The source, license, and representativeness of the uploaded CSV were not independently verified.",
             "- If these test results inform changes, evaluate the revised model on a fresh held-out test set.",
             "", "## Method references", "- https://scikit-learn.org/stable/modules/compose.html",
             "- https://scikit-learn.org/stable/common_pitfalls.html",
             "- https://huggingface.co/docs/hub/datasets-cards"]
    return "\n".join(lines) + "\n"


def search_datasets(query: str) -> list[dict]:
    if not query.strip() or len(query) > 100:
        raise ValueError("Enter a search term of 1 to 100 characters.")
    url = "https://huggingface.co/api/datasets?" + urlencode({"search": query.strip(), "limit": 8, "full": "true"})
    with urlopen(Request(url, headers={"User-Agent": "TrainPilot-Local/1.0"}), timeout=10) as response:
        data = json.load(response)
    return [{"id": d.get("id", ""), "url": "https://huggingface.co/datasets/" + d.get("id", ""),
             "downloads": d.get("downloads"), "tags": d.get("tags", [])[:12],
             "license": (d.get("cardData") or {}).get("license") or "not provided"} for d in data]
