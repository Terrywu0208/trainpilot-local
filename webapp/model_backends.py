"""Installed model integrations; discovery never executes arbitrary Hub code."""
from importlib.metadata import entry_points
import importlib.util
import re


class PredictLMBackend:
    def describe(self):
        return {'id': 'predictlm', 'label': 'Hugging Face PredictLM 26M',
                'repo': 'zerooneresearch/predictlm-base-26m',
                'tasks': ['classification', 'regression'],
                'method': 'local_finetune_then_in_context', 'weight_finetuning': True,
                'available': all(importlib.util.find_spec(x) is not None for x in ('predictlm', 'torch', 'huggingface_hub'))}

    def create(self, task):
        from webapp.hf_tabular import PredictLMEstimator
        return PredictLMEstimator(task=task)


def registry():
    adapters = {'predictlm': PredictLMBackend()}
    for entry in entry_points(group='trainpilot.model_backends'):
        if not re.fullmatch(r'[a-z][a-z0-9_-]{0,40}', entry.name) or entry.name in adapters:
            raise ValueError(f'Invalid or conflicting model backend: {entry.name}')
        adapter = entry.load()
        if isinstance(adapter, type):
            adapter = adapter()
        if not callable(getattr(adapter, 'describe', None)) or not callable(getattr(adapter, 'create', None)):
            raise ValueError(f'Model backend {entry.name} must implement describe and create')
        adapters[entry.name] = adapter
    return adapters


def catalog():
    out = []
    for key, backend in registry().items():
        detail = dict(backend.describe())
        detail['id'] = key
        out.append(detail)
    return out
