"""Versioned local training and evidence-based comparison."""
from __future__ import annotations

import json
from pathlib import Path
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from webapp import classifier, tabular


class VariantState(TypedDict, total=False):
    id: str
    parent: str
    result: dict
    state: str


def create(root: Path, parent_id: str, features: list, options: dict) -> dict:
    parent = classifier._folder(root, parent_id)
    old = json.loads((parent / 'tabular_result.json').read_text(encoding='utf-8'))
    plan = old['proposal']
    if plan['task'] not in ('classification', 'regression'):
        raise ValueError('Version comparison supports classification and regression.')
    _, original_fields, _ = classifier._load(root, parent_id)
    if not isinstance(features, list) or not features or any(f not in original_fields or f == plan['target'] for f in features) or len(features) != len(set(features)):
        raise ValueError('The new version may only use original non-target columns.')
    if not isinstance(options, dict) or set(options) - {'regularization', 'model_count', 'auto_ablation', 'model_backend'}:
        raise ValueError('Unsupported parameter.')
    inherited = {key: value for key, value in old.get('options', {}).items() if key in {'regularization', 'model_count', 'model_backend'}}
    options = {**inherited, **options}
    model_backend = options.get('model_backend', '')
    if not isinstance(model_backend, str):
        raise ValueError('Invalid model adapter configuration.')
    if model_backend:
        from webapp import model_backends
        matching = [item for item in model_backends.catalog() if item['id'] == model_backend]
        if not matching or not matching[0].get('available') or plan['task'] not in matching[0].get('tasks', []):
            raise ValueError('The original model adapter is unavailable or does not support this task.')
    count = options.get('model_count', 2)
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 6:
        raise ValueError('Comparisons support one to six local models.')
    strength = options.get('regularization', 1.0)
    if isinstance(strength, bool) or not isinstance(strength, (int, float)) or not .01 <= strength <= 100:
        raise ValueError('The regularization parameter must be between 0.01 and 100.')
    if 'auto_ablation' in options and not isinstance(options['auto_ablation'], bool):
        raise ValueError('The feature ablation setting must be a boolean.')
    options.setdefault('auto_ablation', False)  # Respect explicitly chosen version features.
    preview = classifier.inspect_csv(root, (parent / 'input.csv').read_text(encoding='utf-8'),
                                     (parent / 'goal.txt').read_text(encoding='utf-8'))
    child_id = preview['id']
    child = classifier._folder(root, child_id)
    (child / 'lineage.json').write_text(json.dumps({'parent': parent_id, 'options': options}, ensure_ascii=False), encoding='utf-8')
    source = parent / 'source.json'
    if source.exists():
        (child / 'source.json').write_text(source.read_text(encoding='utf-8'), encoding='utf-8')

    def train_node(state):
        result = tabular.train(root, child_id, plan['task'], plan['target'], features, options=options)
        return {'result': result}

    def verify_node(state):
        if not state['result'].get('checkpoint_reloaded'):
            raise ValueError('The new model failed reload verification.')
        return {'state': 'completed'}

    graph = StateGraph(VariantState)
    graph.add_node('train', train_node)
    graph.add_node('verify', verify_node)
    graph.add_edge(START, 'train')
    graph.add_edge('train', 'verify')
    graph.add_edge('verify', END)
    state = graph.compile().invoke({'id': child_id, 'parent': parent_id})
    return {'state': state['state'], 'id': child_id, 'result': state['result'], 'comparison': compare(root, parent_id, child_id)}


def compare(root: Path, first_id: str, second_id: str) -> dict:
    first, second = classifier._folder(root, first_id), classifier._folder(root, second_id)
    a = json.loads((first / 'tabular_result.json').read_text(encoding='utf-8'))
    b = json.loads((second / 'tabular_result.json').read_text(encoding='utf-8'))
    _, fields_a, rows_a = classifier._load(root, first_id)
    _, fields_b, rows_b = classifier._load(root, second_id)
    same_data = fields_a == fields_b and rows_a == rows_b
    same_task = a['proposal']['task'] == b['proposal']['task'] and a['proposal']['target'] == b['proposal']['target']
    metric = 'macro_f1' if a['proposal']['task'] == 'classification' else 'mae' if a['proposal']['task'] == 'regression' else 'silhouette'
    def selected_attempt(result):
        selected = result.get('proposal', {}).get('model')
        return next((item for item in result.get('attempts', []) if item.get('model') == selected and item.get('metric') == metric and isinstance(item.get('validation_score'), (int, float))), {})
    first_attempt, second_attempt = selected_attempt(a), selected_attempt(b)
    first_score, second_score = first_attempt.get('validation_score'), second_attempt.get('validation_score')
    same_split = a.get('split_counts') == b.get('split_counts')
    comparable = same_data and same_task and same_split and first_score is not None and second_score is not None
    return {'comparable': comparable, 'same_dataset': same_data, 'same_task': same_task, 'metric': metric,
            'versions': [{'id': first_id, 'model': a['proposal']['model'], 'value': first_score, 'options': a.get('options', {}), 'features': a['proposal'].get('features', []), 'validation_accuracy': first_attempt.get('validation_accuracy'), 'fit_seconds': first_attempt.get('fit_seconds'), 'predict_ms_per_row': first_attempt.get('predict_ms_per_row')},
                         {'id': second_id, 'model': b['proposal']['model'], 'value': second_score, 'options': b.get('options', {}), 'features': b['proposal'].get('features', []), 'validation_accuracy': second_attempt.get('validation_accuracy'), 'fit_seconds': second_attempt.get('fit_seconds'), 'predict_ms_per_row': second_attempt.get('predict_ms_per_row')}],
            'note': 'Compared on the validation set with the same data and split. Repeated tuning can overfit; evaluate on new data.' if comparable else 'Data, task, or split differ, or validation metrics are missing; a fair comparison is unavailable.'}
