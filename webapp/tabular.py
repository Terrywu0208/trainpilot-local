"""Extensible local tabular training adapters: classification, regression and clustering."""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from importlib.metadata import entry_points
from pathlib import Path

from webapp.classifier import _load, _folder

TASKS = {
    "classification": {"label": "Classification: predict a class", "needs_target": True, "model": "Logistic Regression"},
    "regression": {"label": "Regression: predict a number", "needs_target": True, "model": "Ridge regression"},
    "clustering": {"label": "Clustering: explore data", "needs_target": False, "model": "K-means"},
}


def _plugins() -> dict:
    """Load explicitly installed Python entry points, never uploaded code."""
    plugins = {}
    for entry in entry_points(group="trainpilot.training_adapters"):
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,40}", entry.name) or entry.name in TASKS:
            raise ValueError(f"Invalid or conflicting adapter ID: {entry.name}")
        adapter = entry.load()
        if any(not callable(getattr(adapter, name, None)) for name in ("describe", "analyze", "train", "predict", "report", "verify")):
            raise ValueError(f"Adapter {entry.name} does not implement the training interface")
        plugins[entry.name] = adapter
    return plugins


def catalog() -> dict:
    from webapp import model_backends
    extensions = []
    for name, adapter in _plugins().items():
        description = adapter.describe()
        extensions.append({"id": name, "label": str(description["label"]),
                           "needs_target": bool(description.get("needs_target", True)),
                           "model": str(description.get("model", name))})
    return {"tasks": [{"id": key, **value} for key, value in TASKS.items()] + extensions, 'model_backends': model_backends.catalog(),
            "notice": "Local baselines use installed capabilities; not every model, dataset, or task is supported."}


def _number(value):
    try:
        number = float(value)
        import math
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def analyze(root: Path, run_id: str, task: str, target: str = "", features: list | None = None) -> dict:
    folder, fields, rows = _load(root, run_id)
    if task not in TASKS and task in _plugins():
        return _plugins()[task].analyze(root, run_id, target, features)
    if task not in TASKS:
        raise ValueError("Classification, regression, and clustering are supported.")
    if TASKS[task]["needs_target"] and target not in fields:
        raise ValueError("Select a target column to predict.")
    if task == "clustering":
        target = ""
    features = features if features is not None else [f for f in fields if f != target]
    if not isinstance(features, list) or not features or len(features) != len(set(features)) or any(f not in fields or f == target for f in features):
        raise ValueError("Select at least one valid input column and exclude the target.")
    valid = [r for r in rows if not target or r[target]]
    warnings = []
    if len(valid) < 30:
        warnings.append("At least 30 valid rows are required.")
    types = {}
    for feature in features:
        values = [r[feature] for r in valid if r[feature]]
        if not values:
            raise ValueError(f"Input column {feature} is empty; remove it.")
        if re.search(r'(?i:^id$|_id$)|Id$|ID$', feature) and len(set(values)) >= len(values) * .9:
            warnings.append(f"{feature} may be a unique identifier; review whether it should be excluded.")
        if all(_number(value) is not None for value in values):
            kind = "numeric"
        elif len(set(values)) <= 30:
            kind = "categorical"
        else:
            kind = "text"
            warnings.append(f"{feature} has many distinct text values; review for identifiers and leakage.")
        types[feature] = {"type": kind, "missing": len(valid) - len(values), "unique": len(set(values))}
    if task == "classification":
        counts = Counter(r[target] for r in valid)
        if not 2 <= len(counts) <= 20 or (counts and min(counts.values()) < 5):
            warnings.append("Classification requires 2–20 classes and at least 5 rows per class.")
        target_summary = dict(counts)
        ready = len(valid) >= 30 and 2 <= len(counts) <= 20 and min(counts.values()) >= 5
    elif task == "regression":
        numeric_targets = [_number(r[target]) for r in valid]
        ready = len(valid) >= 30 and all(v is not None for v in numeric_targets)
        if not ready:
            warnings.append("Regression requires a numeric target and at least 30 rows.")
        target_summary = {"numeric_values": sum(v is not None for v in numeric_targets)}
    else:
        target_summary = {}
        ready = len(valid) >= 30
    return {"id": run_id, "goal": (folder / "goal.txt").read_text(encoding="utf-8"),
            "task": task, "target": target, "features": features, "types": types,
            "rows": len(valid), "target_summary": target_summary, "warnings": warnings,
            "ready": ready, "model": TASKS[task]["model"],
            "recommendation": "Choose the task and features for a lightweight local baseline. Review inferred types for identifiers, future information, and sensitive features."}


def train(root: Path, run_id: str, task: str, target: str = "", features: list | None = None, on_progress=None, options: dict | None = None) -> dict:
    if task not in TASKS and task in _plugins():
        folder = _folder(root, run_id)
        result = _plugins()[task].train(root, run_id, target, features)
        (folder / "tabular_adapter.json").write_text(json.dumps({"id": task}), encoding="utf-8")
        return result
    import joblib
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.compose import ColumnTransformer
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
    from sklearn.svm import SVC, SVR
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, log_loss, mean_absolute_error, mean_squared_error, r2_score, silhouette_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    proposal = analyze(root, run_id, task, target, features)
    if not proposal["ready"]:
        raise ValueError(" ".join(proposal["warnings"]))
    folder, _, rows = _load(root, run_id)
    if (folder / "tabular_result.json").exists():
        raise ValueError("This experiment is complete; upload the data again for a separate experiment.")
    names = proposal["features"]
    data = [r for r in rows if not proposal["target"] or r[proposal["target"]]]
    signatures = [tuple(r[f].strip().casefold() for f in names) for r in data]
    duplicates = len(signatures) - len(set(signatures))
    x = np.array([[(_number(r[f]) if r[f] else np.nan) if proposal['types'][f]['type'] == 'numeric' else (r[f] or '') for f in names] for r in data], dtype=object)
    def make_transform(selected_names):
        transformers = []
        for idx, name in enumerate(selected_names):
            kind = proposal["types"][name]["type"]
            if kind == "numeric":
                transformers.append((f"n{idx}", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), [idx]))
            elif kind == "categorical":
                transformers.append((f"c{idx}", OneHotEncoder(handle_unknown="ignore"), [idx]))
            else:
                transformers.append((f"t{idx}", TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=10000), idx))
        return ColumnTransformer(transformers)
    transform = make_transform(names)
    options = options or {}
    model_count = options.get('model_count', 2)
    if isinstance(model_count, bool) or not isinstance(model_count, int) or not 1 <= model_count <= 6:
        raise ValueError('Comparisons support one to six local models.')
    if not isinstance(options.get('auto_ablation', True), bool):
        raise ValueError('The feature ablation setting must be a boolean.')
    strength = float(options.get('regularization', 1.0))
    if not .01 <= strength <= 100:
        raise ValueError('The regularization parameter must be between 0.01 and 100.')
    def progress(stage, **details):
        if on_progress:
            on_progress({'stage': stage, **details})
    warnings = list(proposal["warnings"])
    source_path = _folder(root, run_id) / 'source.json'
    if source_path.is_file() and json.loads(source_path.read_text(encoding='utf-8')).get('kind') == 'competition':
        warnings.append('Scores use a local holdout, not the Kaggle leaderboard; no submission was made.')
    if 'regularization' in options:
        warnings.append('Regularization affects linear models only; Random Forest does not use it.')
    ablation = None
    if task == "clustering":
        transformed = transform.fit_transform(x)
        attempts = []
        for k in (2, 3, 4):
            candidate = KMeans(n_clusters=k, random_state=42, n_init=5)
            labels = candidate.fit_predict(transformed)
            unique = len(set(labels))
            if 1 < unique < len(labels):
                score = float(silhouette_score(transformed, labels, sample_size=min(len(labels), 500), random_state=42))
                attempts.append({"groups": k, "silhouette": score})
        if not attempts:
            raise ValueError("The data cannot form at least two distinct clusters; check the input features.")
        best = max(attempts, key=lambda a: a["silhouette"])
        model = Pipeline([("features", transform), ("model", KMeans(n_clusters=best["groups"], random_state=42, n_init=5))])
        model.fit(x)
        predictions = model.predict(x)
        metrics = {"silhouette": best["silhouette"], "cluster_counts": {str(k): int(v) for k, v in Counter(predictions).items()}}
        split = {"exploration": len(x)}
        suggestions = ["Clustering is exploratory, not ground-truth classification; review samples, scaling, and stability across cluster counts."]
    else:
        y = [r[proposal["target"]] for r in data]
        if task == "regression":
            y = [_number(v) for v in y]
        try:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42, stratify=y if task == "classification" else None)
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=.25, random_state=42, stratify=y_train if task == "classification" else None)
        except ValueError as exc:
            raise ValueError(f"Insufficient data for the evaluation split: {exc}") from exc
        from sklearn.base import clone
        candidates = ([('Logistic Regression', LogisticRegression(max_iter=300, class_weight='balanced', C=strength, random_state=42)),
                       ('Random Forest', RandomForestClassifier(n_estimators=40, max_depth=12, n_jobs=1, random_state=42)),
                       ('Extra Trees', ExtraTreesClassifier(n_estimators=40, max_depth=12, n_jobs=1, random_state=42)),
                       ('Support Vector Machine', SVC()), ('Decision Tree', DecisionTreeClassifier(max_depth=8, random_state=42)),
                       ('K Nearest Neighbors', KNeighborsClassifier(n_neighbors=3))]
                      if task == 'classification' else
                      [('Ridge', Ridge(alpha=strength)), ('Random Forest', RandomForestRegressor(n_estimators=40, max_depth=12, n_jobs=1, random_state=42)),
                       ('Extra Trees', ExtraTreesRegressor(n_estimators=40, max_depth=12, n_jobs=1, random_state=42)), ('Support Vector Regression', SVR()),
                       ('Decision Tree', DecisionTreeRegressor(max_depth=8, random_state=42)), ('K Nearest Neighbors', KNeighborsRegressor(n_neighbors=3))])[:model_count]
        from webapp import model_backends
        backend_id = options.get('model_backend', '')
        backend_info = None
        if backend_id:
            adapters = model_backends.registry()
            backend = adapters.get(backend_id)
            backend_info = backend.describe() if backend else None
            if not backend_info or not backend_info.get('available') or task not in backend_info.get('tasks', []):
                raise ValueError('The requested model adapter is unavailable or incompatible with this task.')
            candidates[-1] = (backend_info['label'], backend.create(task))
        explicit_keep_all = bool(re.search(r'(?:保留|使用|採用|采用)(?:所有|全部|全部的)(?:欄位|字段|特徵|特征)|(?:keep|use) all features', proposal['goal'], re.I))
        risky = []
        for idx, name in enumerate(names):
            observed = [str(v) for v in x_train[:, idx] if str(v) not in ('', 'nan')]
            unique = len(set(observed))
            is_id = bool(re.search(r'(?i:^id$|_id$)|Id$|ID$', name)) and observed and unique >= len(observed) * .9
            high_card_text = proposal['types'][name]['type'] == 'text' and unique > 30 and observed and unique >= len(observed) * .5
            if (is_id or high_card_text) and not re.search(r'(?:保留|keep)\s*' + re.escape(name) + r'(?![A-Za-z0-9_])', proposal['goal'], re.I):
                risky.append(name)
        cleaned = [name for name in names if name not in risky]
        profiles = [('baseline', names)]
        if options.get('auto_ablation', True) and not explicit_keep_all and risky and cleaned:
            profiles.append(('reduced', cleaned))
        profile_attempts, trained = {}, []
        total = len(candidates) * len(profiles)
        for profile_index, (profile_name, selected_names) in enumerate(profiles):
            indices = [names.index(name) for name in selected_names]
            profile_attempts[profile_name] = []
            for index, (name, estimator) in enumerate(candidates):
                current = profile_index * len(candidates) + index + 1
                progress('training', model=name, profile=profile_name, index=current, total=total, task=task)
                candidate = Pipeline([('features', make_transform(selected_names)), ('model', clone(estimator))])
                started = time.perf_counter()
                loss_curve = None
                if isinstance(estimator, (RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor)):
                    # Record actual ensemble growth, evaluating only a bounded training sample.
                    transformed = candidate.named_steps['features'].fit_transform(x_train[:, indices])
                    fitted = candidate.named_steps['model']
                    fitted.set_params(warm_start=True)
                    elapsed = time.perf_counter() - started
                    sample_count = min(len(y_train), 1000)
                    sample_y = np.asarray(y_train)[:sample_count]
                    points = []
                    for tree_count in (10, 20, 30, 40):
                        fitted.set_params(n_estimators=tree_count)
                        fit_started = time.perf_counter()
                        fitted.fit(transformed, y_train)
                        elapsed += time.perf_counter() - fit_started
                        if task == 'classification':
                            loss = log_loss(sample_y, fitted.predict_proba(transformed[:sample_count]), labels=fitted.classes_)
                        else:
                            loss = mean_squared_error(sample_y, fitted.predict(transformed[:sample_count]))
                        points.append({'step': tree_count, 'loss': float(loss)})
                    loss_curve = {'metric': 'log_loss' if task == 'classification' else 'mean_squared_error',
                                  'step_unit': 'trees', 'sample_count': sample_count, 'points': points}
                else:
                    try:
                        candidate.fit(x_train[:, indices], y_train)
                    except Exception as exc:
                        if not backend_info or name != backend_info['label']:
                            raise
                        warnings.append(f'{name} training failed ({type(exc).__name__}): {str(exc)[:200]}; excluded from successful comparisons.')
                        progress('model_failed', model=name, profile=profile_name, index=current, total=total)
                        continue
                    elapsed = time.perf_counter() - started
                prediction_started = time.perf_counter()
                try:
                    val_prediction = candidate.predict(x_val[:, indices])
                except Exception as exc:
                    if not backend_info or name != backend_info['label']:
                        raise
                    warnings.append(f'{name} inference failed ({type(exc).__name__}): {str(exc)[:200]}; excluded from successful comparisons.')
                    progress('model_failed', model=name, profile=profile_name, index=current, total=total)
                    continue
                prediction_seconds = time.perf_counter() - prediction_started
                val_score = float(f1_score(y_val, val_prediction, average='macro', zero_division=0)) if task == 'classification' else float(mean_absolute_error(y_val, val_prediction))
                entry = {'model': name, 'metric': 'macro_f1' if task == 'classification' else 'mae', 'validation_score': val_score,
                         'fit_seconds': elapsed, 'predict_ms_per_row': prediction_seconds * 1000 / len(x_val)}
                if loss_curve is not None:
                    entry['loss_curve'] = loss_curve
                if backend_info and name == backend_info['label']:
                    entry.update({'backend': backend_id, 'model_source': backend_info.get('repo'),
                                  'method': backend_info.get('method'), 'weight_finetuning': bool(backend_info.get('weight_finetuning'))})
                    entry['context_rows'] = getattr(candidate.named_steps['model'], 'context_rows_', None)
                    entry['context_features'] = getattr(candidate.named_steps['model'], 'context_features_', None)
                    backend_loss = getattr(candidate.named_steps['model'], 'loss_curve_', None)
                    if backend_loss:
                        entry['loss_curve'] = {'metric': 'cross_entropy' if task == 'classification' else 'bar_distribution_loss',
                                               'step_unit': 'gradient_step', 'sample_count': min(len(y_train), 80), 'points': backend_loss}
                if task == 'classification':
                    entry['validation_accuracy'] = float(accuracy_score(y_val, val_prediction))
                profile_attempts[profile_name].append(entry)
                trained.append((profile_name, selected_names, indices, candidate, entry))
                progress('model_complete', model=name, profile=profile_name, index=current, total=total, metric=entry['metric'], score=val_score)
        if not trained:
            raise ValueError('No model completed successfully. ' + ' '.join(warnings[-2:]))
        best = (max if task == 'classification' else min)(trained, key=lambda item: item[4]['validation_score'])
        selected_profile, selected_names, selected_indices, model, selected_entry = best
        attempts = profile_attempts[selected_profile]
        proposal['model'] = selected_entry['model']
        proposal['features'] = selected_names
        predictions = model.predict(x_test[:, selected_indices])
        if len(profiles) == 2:
            ablation = {'status': 'completed', 'removed_features': risky, 'selected_profile': selected_profile,
                        'metric': 'macro_f1' if task == 'classification' else 'mae',
                        'baseline': profile_attempts['baseline'], 'reduced': profile_attempts['reduced'],
                        'note': 'Both profiles use the same training and validation split. The holdout test evaluates only the chosen profile and fixed baseline; repeated validation can introduce selection bias.'}
            if selected_profile == 'reduced':
                def clarify_warning(warning):
                    return next((f'Original plan: {warning} Final plan excluded {name}。' for name in risky if warning.startswith(name + ' ')), warning)
                warnings = [clarify_warning(warning) for warning in warnings]
                proposal['warnings'] = [clarify_warning(warning) for warning in proposal['warnings']]
        selected_signatures = [tuple(r[f].strip().casefold() for f in selected_names) for r in data]
        duplicates = len(selected_signatures) - len(set(selected_signatures))
        if task == "classification":
            baseline = DummyClassifier(strategy="most_frequent").fit(x_train, y_train)
            labels = sorted(set(y))
            metrics = {"accuracy": float(accuracy_score(y_test, predictions)), "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
                       "baseline_accuracy": float(accuracy_score(y_test, baseline.predict(x_test))),
                       "labels": labels, "per_class": classification_report(y_test, predictions, labels=labels, output_dict=True, zero_division=0),
                       "confusion_matrix": confusion_matrix(y_test, predictions, labels=labels).tolist()}
        else:
            baseline = DummyRegressor(strategy="mean").fit(x_train, y_train)
            metrics = {"mae": float(mean_absolute_error(y_test, predictions)), "r2": float(r2_score(y_test, predictions)),
                       "baseline_mae": float(mean_absolute_error(y_test, baseline.predict(x_test)))}
        split = {"train": len(x_train), "validation": len(x_val), "test": len(x_test)}
        suggestions = ["Review errors and data representativeness; use new independent test data after further tuning."]
    if duplicates:
        warnings.append(f"Detected {duplicates} duplicate inputs; random splitting can inflate performance. Review and use a group-based split on new data.")
    artifact = folder / "tabular.joblib"
    joblib.dump(model, artifact)
    restored = joblib.load(artifact)
    check_x = x if task == "clustering" else x_test[:, selected_indices]
    if not np.array_equal(restored.predict(check_x), predictions):
        raise RuntimeError("Predictions differ after model reload.")
    has_backend = task != 'clustering' and any(a.get('backend') for a in attempts)
    result = {"id": run_id, "status": "completed", "proposal": proposal, "metrics": metrics,
              "split_counts": split, "attempts": attempts, "suggestions": warnings + suggestions,
              "checkpoint_reloaded": True, "training_type": "mixed local model comparison" if has_backend else "real local scikit-learn", "options": options, "ablation": ablation}
    (folder / "tabular_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def predict(root: Path, run_id: str, values: dict) -> dict:
    import joblib
    import numpy as np
    folder = _folder(root, run_id)
    marker = folder / "tabular_adapter.json"
    if marker.is_file():
        adapter_id = json.loads(marker.read_text(encoding="utf-8"))["id"]
        if adapter_id not in _plugins():
            raise ValueError("The model adapter used for training is unavailable.")
        return _plugins()[adapter_id].predict(root, run_id, values)
    if not (folder / "tabular_result.json").is_file():
        raise ValueError("Complete training first.")
    result = json.loads((folder / "tabular_result.json").read_text(encoding="utf-8"))
    proposal = result["proposal"]
    names = proposal["features"]
    if not isinstance(values, dict) or set(values) != set(names):
        raise ValueError("Provide the same input features used during training.")
    x = []
    for name in names:
        value = str(values[name]).strip()
        if len(value) > 5000:
            raise ValueError("Each field must be at most 5,000 characters.")
        if proposal["types"][name]["type"] == "numeric":
            if value and _number(value) is None:
                raise ValueError(f"{name} must be numeric.")
            x.append(_number(value) if value else np.nan)
        else:
            x.append(value)
    prediction = joblib.load(folder / "tabular.joblib").predict(np.array([x], dtype=object))[0]
    return {"value": prediction.item() if hasattr(prediction, "item") else prediction,
            "task": proposal["task"], "note": "Predictions are estimates; cluster labels are not verified classes."}


def report(root: Path, run_id: str) -> str:
    folder = _folder(root, run_id)
    marker = folder / "tabular_adapter.json"
    if marker.is_file():
        adapter_id = json.loads(marker.read_text(encoding="utf-8"))["id"]
        if adapter_id not in _plugins():
            raise ValueError("The model adapter used for training is unavailable.")
        return _plugins()[adapter_id].report(root, run_id)
    path = folder / "tabular_result.json"
    if not path.is_file():
        raise ValueError("Complete training first.")
    r = json.loads(path.read_text(encoding="utf-8"))
    p = r["proposal"]
    source_file = folder / "source.json"
    source = json.loads(source_file.read_text(encoding="utf-8")) if source_file.exists() else None
    return "\n".join(["# Local Model Experiment Report", "", f"Goal: {p['goal'] or 'Not provided'}", f"Data source: {source['url'] if source else 'User-provided, unverified source'}", f"Task: {p['task']}",
                      f"Model: {p['model']}", f"Input features: {p['features']}", f"Target: {p['target'] or 'None'}",
                      f"Rows: {p['rows']}", f"Data split: {r['split_counts']}", f"Metrics: {r['metrics']}",
                      f"Attempts: {r['attempts']}", f"Model reload verified: {r['checkpoint_reloaded']}", "", "## Suggestions",
                      *[f"- {s}" for s in r['suggestions']], "", "## Limitations", "- The system has not independently verified the source, license, representativeness, or column semantics of uploaded data.",
                      "- The implemented model selection does not support every possible task.", "- Avoid future information leakage and review identifier columns.", ""])
