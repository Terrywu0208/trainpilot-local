import csv
import io

from fastapi.testclient import TestClient
from sklearn.datasets import load_diabetes, load_iris
from webapp.main import create_app
from webapp import tabular


def make_csv(dataset, target=True):
    out = io.StringIO()
    writer = csv.writer(out)
    names = ['feature_' + str(i) for i in range(dataset.data.shape[1])]
    writer.writerow(names + (['target'] if target else []))
    for x, y in zip(dataset.data, dataset.target):
        writer.writerow(list(x) + ([y] if target else []))
    return out.getvalue(), names


def test_iris_classification_and_clustering(tmp_path):
    client = TestClient(create_app(runs_root=tmp_path))
    csv_text, features = make_csv(load_iris())
    for task in ('classification', 'clustering'):
        run = client.post('/api/classifier/inspect', json={'csv': csv_text, 'goal': 'test models'}).json()
        body = {'id': run['id'], 'task': task, 'target': 'target' if task == 'classification' else '', 'features': features}
        plan = client.post('/api/training/plan', json=body)
        assert plan.status_code == 200 and plan.json()['ready']
        result = client.post('/api/training/train', json=body)
        assert result.status_code == 200, result.text
        data = result.json()
        assert data['checkpoint_reloaded']
        if task == 'classification':
            assert data['metrics']['accuracy'] > .8
            curve = next(a['loss_curve'] for a in data['attempts'] if a['model'] == 'Random Forest')
            assert curve['metric'] == 'log_loss' and [p['step'] for p in curve['points']] == [10, 20, 30, 40]
            assert all(p['loss'] >= 0 for p in curve['points'])
            report = client.get('/api/training/' + run['id'] + '/report').text
            assert 'Training loss: log_loss' in report and 'No stepwise loss recorded for: Logistic Regression' in report
        else:
            assert 2 <= len(data['metrics']['cluster_counts']) <= 4
        response = client.post('/api/training/predict', json={'id': run['id'], 'values': {name: float(value) for name, value in zip(features, load_iris().data[0])}})
        assert response.status_code == 200
        assert 'test models' in client.get('/api/training/' + run['id'] + '/report').text


def test_diabetes_regression_and_invalid_target(tmp_path):
    client = TestClient(create_app(runs_root=tmp_path))
    csv_text, features = make_csv(load_diabetes())
    run = client.post('/api/classifier/inspect', json={'csv': csv_text}).json()
    body = {'id': run['id'], 'task': 'regression', 'target': 'target', 'features': features}
    assert client.post('/api/training/plan', json=body).json()['ready']
    result = client.post('/api/training/train', json=body)
    assert result.status_code == 200, result.text
    assert result.json()['metrics']['mae'] < result.json()['metrics']['baseline_mae']
    curve = next(a['loss_curve'] for a in result.json()['attempts'] if a['model'] == 'Random Forest')
    assert curve['metric'] == 'mean_squared_error' and len(curve['points']) == 4
    assert client.post('/api/training/plan', json={**body, 'target': 'missing'}).status_code == 400


def test_installed_model_backend_is_compared_on_same_split(tmp_path, monkeypatch):
    from sklearn.dummy import DummyClassifier
    from webapp import classifier, model_backends
    class Backend:
        def describe(self):
            return {'label': 'Extension baseline', 'tasks': ['classification'], 'available': True,
                    'repo': 'example/backend', 'method': 'fit', 'weight_finetuning': True}
        def create(self, task):
            return DummyClassifier(strategy='most_frequent')
    monkeypatch.setattr(model_backends, 'registry', lambda: {'extension': Backend()})
    content, features = make_csv(load_iris())
    run = classifier.inspect_csv(tmp_path, content, '比較模型')
    result = tabular.train(tmp_path, run['id'], 'classification', 'target', features,
                           options={'model_count': 2, 'model_backend': 'extension'})
    assert len(result['attempts']) == 2 and result['checkpoint_reloaded']
    assert result['attempts'][1]['backend'] == 'extension'
    assert result['training_type'] == 'mixed local model comparison'


def test_followup_runs_multiple_real_variants_with_inherited_backend(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from sklearn.dummy import DummyClassifier
    from webapp import classifier, data_agent, model_backends, variants
    class Backend:
        def describe(self):
            return {'label': 'Extension baseline', 'tasks': ['classification'], 'available': True,
                    'repo': 'example/backend', 'method': 'fit', 'weight_finetuning': False}
        def create(self, task):
            return DummyClassifier(strategy='most_frequent')
    monkeypatch.setattr(model_backends, 'registry', lambda: {'extension': Backend()})
    content, features = make_csv(load_iris())
    run = classifier.inspect_csv(tmp_path, content, '分類比較')
    parent = tabular.train(tmp_path, run['id'], 'classification', 'target', features,
                           options={'model_count': 2, 'model_backend': 'extension'})
    def reply(tool, args):
        call = SimpleNamespace(name=tool, args=args)
        return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[SimpleNamespace(function_call=call, text=None, thought=False)]))])
    class Models:
        def generate_content(self, **kwargs):
            return reply('retrain_variant', {'features': features, 'regularization': .5})
    history = [{'role': 'assistant', 'text': '可以試試將正則化參數設為 0.5，再比較驗證表現。'}]
    action = data_agent.respond(tmp_path, run['id'], '請依建議重訓', client=SimpleNamespace(models=Models()), history=history)
    assert action['options']['model_backend'] == 'extension'
    first = variants.create(tmp_path, run['id'], action['features'], action['options'])
    assert first['state'] == 'completed' and first['result']['checkpoint_reloaded']
    assert first['result']['options']['model_backend'] == 'extension'
    assert first['comparison']['metric'] == 'macro_f1' and first['comparison']['comparable']
    assert all(version.get('validation_accuracy') is not None and version.get('fit_seconds') is not None and version.get('predict_ms_per_row') is not None for version in first['comparison']['versions'])
    selected = first['result']['proposal']['model']
    assert first['comparison']['versions'][1]['value'] == next(x['validation_score'] for x in first['result']['attempts'] if x['model'] == selected)
    second = variants.create(tmp_path, first['id'], features, {'regularization': 2.0})
    assert second['state'] == 'completed' and second['result']['options']['model_backend'] == 'extension'
    assert second['result']['options']['model_count'] == 2
    assert parent['options']['model_backend'] == 'extension'


def test_installed_adapter_is_discovered_and_routed(tmp_path, monkeypatch):
    class Adapter:
        def describe(self): return {'label': 'custom task', 'needs_target': False, 'model': 'custom'}
        def analyze(self, root, run_id, target, features): return {'id': run_id, 'task': 'custom', 'ready': True}
        def train(self, root, run_id, target, features): return {'status': 'completed'}
        def predict(self, root, run_id, values): return {'value': 7}
        def report(self, root, run_id): return 'custom report'
        def verify(self, root, run_id, result): return True
    class Entry:
        name = 'custom'
        def load(self): return Adapter()
    monkeypatch.setattr(tabular, 'entry_points', lambda **kwargs: [Entry()])
    client = TestClient(create_app(runs_root=tmp_path))
    csv_text, features = make_csv(load_iris())
    run = client.post('/api/classifier/inspect', json={'csv': csv_text}).json()
    assert any(t['id'] == 'custom' for t in client.get('/api/training/capabilities').json()['tasks'])
    body = {'id': run['id'], 'task': 'custom', 'target': '', 'features': features}
    assert client.post('/api/training/plan', json=body).json()['ready']
    assert client.post('/api/training/train', json=body).json()['status'] == 'completed'
    assert client.post('/api/training/predict', json={'id': run['id'], 'values': {}}).json()['value'] == 7
    assert client.get('/api/training/' + run['id'] + '/report').text == 'custom report'


def test_variant_and_version_comparison(tmp_path):
    client = TestClient(create_app(runs_root=tmp_path))
    data, features = make_csv(load_iris())
    run = client.post('/api/classifier/inspect', json={'csv': data, 'goal': '分類'}).json()
    body = {'id': run['id'], 'task': 'classification', 'target': 'target', 'features': features}
    first = client.post('/api/training/train', json=body)
    assert first.status_code == 200, first.text
    result = client.post('/api/training/variant', json={'parent_id': run['id'], 'features': features[:-1], 'options': {'regularization': 0.5, 'model_count': 5}})
    assert result.status_code == 200, result.text
    variant = result.json()
    assert variant['state'] == 'completed' and variant['comparison']['comparable']
    assert variant['result']['options']['regularization'] == 0.5
    assert len(variant['result']['attempts']) == 5
    assert client.get('/api/training/compare', params={'first': run['id'], 'second': variant['id']}).json()['same_dataset']
    restored = client.post('/api/training/variant', json={'parent_id': variant['id'], 'features': features, 'options': {}})
    assert restored.status_code == 200 and restored.json()['result']['proposal']['features'] == features


def test_multiple_sessions_are_isolated_and_report_has_template(tmp_path):
    client = TestClient(create_app(runs_root=tmp_path))
    csv_text, features = make_csv(load_iris())
    headers = [{'X-Training-Session': c * 32} for c in 'abc']
    runs = []
    for i, header in enumerate(headers):
        response = client.post('/api/classifier/inspect', headers=header, json={'csv': csv_text, 'goal': f'session {i} <script>alert(1)</script>'})
        assert response.status_code == 200
        runs.append(response.json()['id'])
    body = {'id': runs[0], 'task': 'classification', 'target': 'target', 'features': features}
    assert client.post('/api/training/train', headers=headers[0], json=body).status_code == 200
    second = {**body, 'id': runs[1]}
    assert client.post('/api/training/train', headers=headers[1], json=second).status_code == 200
    assert client.post('/api/training/plan', headers=headers[1], json=body).status_code == 400
    assert client.post('/api/training/predict', headers=headers[2], json={'id': runs[0], 'values': {}}).status_code == 400
    report = client.get(f'/api/training/{runs[0]}/report', params={'session': 'a' * 32})
    assert report.status_code == 200 and 'text/html' in report.headers['content-type']
    assert 'Training experiment report' in report.text and 'Model comparison' in report.text and 'confusion matrix' in report.text
    assert '<script>alert(1)</script>' not in report.text and '&lt;script&gt;' in report.text
    assert client.get(f'/api/training/{runs[0]}/report', params={'session': 'b' * 32}).status_code == 400
    other_report = client.get(f'/api/training/{runs[1]}/report', params={'session': 'b' * 32})
    assert other_report.status_code == 200 and 'session 1' in other_report.text and 'session 0' not in other_report.text
    assert client.get('/api/training/agent-budget', headers={'X-Training-Session': '../bad'}).status_code == 400


def test_session_budget_is_shared(tmp_path):
    from webapp import data_agent
    root = tmp_path / 'classifier'
    data_agent.budget(root / 'sessions' / ('a' * 32), consume=True, service='openai')
    data_agent.budget(root / 'sessions' / ('b' * 32), consume=True, service='openai')
    assert data_agent.budget(root, service='openai')['used'] == 2


def test_real_feature_ablation_compares_models_and_reloads(tmp_path, monkeypatch):
    import csv
    import io
    from webapp import classifier, tabular
    data = load_iris()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['RecordId', 'high_card_text', 'length', 'width', 'target'])
    for i, (values, target) in enumerate(zip(data.data, data.target)):
        writer.writerow([i, f'text-{i}', values[0], values[1], int(target)])
    writer.writerow([150, 'text-150', data.data[0][0], data.data[0][1], int(data.target[0])])
    import sklearn.metrics
    real_f1 = sklearn.metrics.f1_score
    evaluated_sizes = []
    def counted_f1(actual, predicted, **kwargs):
        evaluated_sizes.append(len(actual))
        return real_f1(actual, predicted, **kwargs)
    monkeypatch.setattr(sklearn.metrics, 'f1_score', counted_f1)
    run = classifier.inspect_csv(tmp_path, output.getvalue(), '分類並比較模型')
    result = tabular.train(tmp_path, run['id'], 'classification', 'target', ['RecordId', 'high_card_text', 'length', 'width'])
    ablation = result['ablation']
    assert ablation['status'] == 'completed' and set(ablation['removed_features']) == {'RecordId', 'high_card_text'}
    assert len(ablation['baseline']) == len(ablation['reduced']) == 2
    assert all(a['fit_seconds'] >= 0 and 0 <= a['validation_score'] <= 1 for a in ablation['baseline'] + ablation['reduced'])
    assert result['checkpoint_reloaded'] and (tmp_path / run['id'] / 'tabular.joblib').is_file()
    assert result['split_counts'] == {'train': 90, 'validation': 30, 'test': 31}
    assert evaluated_sizes.count(31) == 1
    assert 'test evaluates only the chosen profile' in ablation['note']
    assert 'Feature ablation comparison' in __import__('webapp.report_page', fromlist=['render']).render(tmp_path, run['id'])
    keep = classifier.inspect_csv(tmp_path, output.getvalue(), '分類，保留所有欄位')
    kept = tabular.train(tmp_path, keep['id'], 'classification', 'target', ['RecordId', 'high_card_text', 'length', 'width'])
    assert kept['ablation'] is None and kept['proposal']['features'] == ['RecordId', 'high_card_text', 'length', 'width']
