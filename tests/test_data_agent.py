from types import SimpleNamespace
import csv
import io

import pytest
from sklearn.datasets import load_iris

from webapp import classifier, data_agent
from webapp import tabular


def make_csv(dataset):
    output = io.StringIO()
    writer = csv.writer(output)
    names = ['feature_' + str(i) for i in range(dataset.data.shape[1])]
    writer.writerow(names + ['target'])
    for x, y in zip(dataset.data, dataset.target):
        writer.writerow(list(x) + [y])
    return output.getvalue(), names


def response(tool=None, args=None, text=None):
    call = SimpleNamespace(name=tool, args=args) if tool else None
    part = SimpleNamespace(function_call=call, text=text, thought=False)
    return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))])


class FakeModels:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.requests = []

    def generate_content(self, **kwargs):
        self.requests.append(kwargs)
        if str(getattr(kwargs.get('config'), 'system_instruction', '')).startswith('SPECIALIST:'):
            return response(text='請檢查資料品質並比較模型。')
        return next(self.outputs)


def test_agent_inspects_then_finalizes_without_raw_rows(tmp_path):
    content, features = make_csv(load_iris())
    run = classifier.inspect_csv(tmp_path, content, '預測類別')
    args = {'task': 'classification', 'target': 'target', 'features': features}
    fake = FakeModels([response('inspect_plan', args), response('finalize_plan', args)])
    result = data_agent.propose(tmp_path, run['id'], client=SimpleNamespace(models=fake))
    assert result['state'] == 'ready' and result['plan']['ready']
    assert [x['tool'] for x in result['trace']] == ['inspect_plan', 'finalize_plan']
    assert data_agent.budget(tmp_path)['used'] == 2
    initial = fake.requests[0]['contents'][0].parts[0].text
    assert 'feature_0' in initial and '5.1,3.5' not in initial


def test_repeated_identical_valid_plan_does_not_loop(tmp_path):
    content, features = make_csv(load_iris())
    run = classifier.inspect_csv(tmp_path, content, '請預測類別並訓練')
    args = {'task': 'classification', 'target': 'target', 'features': features}
    fake = FakeModels([response('inspect_plan', args), response('inspect_plan', args)])
    result = data_agent.propose(tmp_path, run['id'], client=SimpleNamespace(models=fake), execute=True)
    assert result['state'] == 'completed' and result['result']['checkpoint_reloaded']
    assert len(fake.requests) == 3  # two specialists and one validated plan
    assert result['trace'][-1]['status'] == 'auto_finalized'


def test_repeated_valid_inspection_finishes_without_starting_training(tmp_path):
    content, features = make_csv(load_iris())
    run = classifier.inspect_csv(tmp_path, content, 'Predict the target')
    args = {'task': 'classification', 'target': 'target', 'features': features}
    fake = FakeModels([response('inspect_plan', args), response('inspect_plan', args)])
    result = data_agent.propose(tmp_path, run['id'], client=SimpleNamespace(models=fake), execute=False)
    assert result['state'] == 'ready' and result['plan']['ready']
    assert len(fake.requests) == 2
    assert not (tmp_path / run['id'] / 'tabular.joblib').exists()


def test_dataset_name_as_task_is_repaired_only_for_explicit_binary_goal(tmp_path):
    content, features = make_csv(load_iris())
    content = '\n'.join(content.splitlines()[:101]) + '\n'
    run = classifier.inspect_csv(tmp_path, content, '預測目標是否成立')
    args = {'task': 'some_dataset', 'target': 'target', 'features': features}
    fake = FakeModels([response('inspect_plan', args)])
    result = data_agent.propose(tmp_path, run['id'], client=SimpleNamespace(models=fake), execute=True)
    assert result['state'] == 'completed' and result['plan']['task'] == 'classification'
    assert any(item['status'] == 'normalized_binary_task' for item in result['trace'])


def test_binary_survival_prediction_repairs_dataset_name_task(tmp_path):
    content, features = make_csv(load_iris())
    content = '\n'.join(content.splitlines()[:101]) + '\n'
    run = classifier.inspect_csv(tmp_path, content, 'Predict the survival rate and compare models.')
    args = {'task': 'titanic', 'target': 'target', 'features': features}
    fake = FakeModels([response('inspect_plan', args), response('finalize_plan', args)])
    result = data_agent.propose(tmp_path, run['id'], client=SimpleNamespace(models=fake))
    assert result['state'] == 'ready' and result['plan']['task'] == 'classification'


def test_repeated_identical_planning_error_stops_early_with_reason(tmp_path):
    content, features = make_csv(load_iris())
    run = classifier.inspect_csv(tmp_path, content, 'Train a regression model')
    args = {'task': 'unknown-dataset', 'target': 'target', 'features': features}
    fake = FakeModels([response('inspect_plan', args)] * 4)
    result = data_agent.propose(tmp_path, run['id'], client=SimpleNamespace(models=fake))
    assert result['state'] == 'needs_input'
    assert 'could not be validated' in result['message']
    assert len(fake.requests) == 2


def test_planner_can_use_hugging_face_mcp_before_inspection(tmp_path, monkeypatch):
    from webapp import hf_mcp
    monkeypatch.setattr(hf_mcp, 'search', lambda *args: '### example/model\nlicense:apache-2.0')
    content, features = make_csv(load_iris())
    run = classifier.inspect_csv(tmp_path, content, '預測類別')
    args = {'task': 'classification', 'target': 'target', 'features': features}
    fake = FakeModels([response('hf_mcp_search', {'query': 'tabular', 'repo_type': 'model'}), response('inspect_plan', args)])
    result = data_agent.propose(tmp_path, run['id'], client=SimpleNamespace(models=fake), execute=True)
    assert result['state'] == 'completed'
    assert any(item['tool'] == 'hf_mcp_search' and item['status'] == 'completed' for item in result['trace'])


def test_intake_can_search_models_via_mcp(tmp_path, monkeypatch):
    from webapp import hf_mcp
    monkeypatch.setattr(hf_mcp, 'search', lambda *args: '### example/table-model\n')
    fake = FakeModels([response('hf_mcp_search', {'query': 'table', 'repo_type': 'model'})])
    result = data_agent.converse(tmp_path, [{'role': 'user', 'text': '找 Hugging Face table 模型'}], client=SimpleNamespace(models=fake))
    assert result['datasets'][0]['url'] == 'https://huggingface.co/example/table-model'
    assert 'MCP' in result['message']


def test_uninspected_final_plan_is_never_accepted_and_planning_is_bounded(tmp_path):
    content, features = make_csv(load_iris())
    run = classifier.inspect_csv(tmp_path, content, '預測類別')
    args = {'task': 'classification', 'target': 'target', 'features': features}
    fake = FakeModels([response('finalize_plan', args)] * data_agent.MAX_MODEL_CALLS)
    result = data_agent.propose(tmp_path, run['id'], client=SimpleNamespace(models=fake))
    assert result['state'] == 'needs_input' and data_agent.budget(tmp_path)['used'] == 2


def test_provider_requests_have_no_local_daily_or_task_quota(tmp_path):
    import sqlite3
    data_agent.budget(tmp_path, consume=True, run_id='run', service='openai')
    with sqlite3.connect(tmp_path / '.openai-budget.sqlite3') as db:
        db.execute('UPDATE usage SET calls=200')
        db.execute('UPDATE task_usage SET calls=80')
    result = data_agent.budget(tmp_path, consume=True, run_id='run', service='openai')
    assert (result['used'], result['task_used']) == (201, 81)
    assert result['daily_limit'] is None and result['per_task_limit'] is None


def test_provider_requests_are_spaced_across_calls(tmp_path, monkeypatch):
    delays = []
    monkeypatch.setattr(data_agent.time, 'time', lambda: 100.0)
    monkeypatch.setattr(data_agent.time, 'sleep', delays.append)
    data_agent.budget(tmp_path, consume=True, pace=True)
    data_agent.budget(tmp_path, consume=True, pace=True)
    assert delays == [5.0]


def test_provider_503_switches_model_once_and_counts_both_requests(tmp_path):
    from google.genai.errors import ServerError
    class Models:
        def __init__(self): self.names = []
        def generate_content(self, **kwargs):
            self.names.append(kwargs['model'])
            if len(self.names) == 1:
                raise ServerError(503, {'error': {'message': 'unavailable'}})
            return response(text='ok')
    models = Models()
    data_agent._generate(tmp_path, SimpleNamespace(models=models), False, contents='hello', config=None)
    assert len(models.names) == 2 and models.names[0] != models.names[1]
    assert data_agent.budget(tmp_path)['used'] == 2


def test_fallback_third_model_and_model_specific_pacing(tmp_path, monkeypatch):
    from google.genai.errors import ServerError
    class Models:
        def __init__(self): self.names = []
        def generate_content(self, **kwargs):
            self.names.append(kwargs['model'])
            if len(self.names) < 3:
                raise ServerError(503, {'error': {'message': 'unavailable'}})
            return response(text='ok')
    models = Models()
    data_agent._generate(tmp_path, SimpleNamespace(models=models), False, contents='hello', config=None)
    assert models.names[-1] == 'gemini-3.6-flash'
    assert data_agent.budget(tmp_path)['used'] == 3
    delays = []
    monkeypatch.setattr(data_agent.time, 'time', lambda: 100.0)
    monkeypatch.setattr(data_agent.time, 'sleep', delays.append)
    data_agent.budget(tmp_path, consume=True, pace=True, model='gemini-3.6-flash')
    data_agent.budget(tmp_path, consume=True, pace=True, model='gemini-3.6-flash')
    assert delays == [13.0]


def test_graph_runs_model_tool_execution_and_verification(tmp_path):
    content, features = make_csv(load_iris())
    run = classifier.inspect_csv(tmp_path, content, '請依特徵預測分類')
    args = {'task': 'classification', 'target': 'target', 'features': features}
    fake = FakeModels([response('inspect_plan', args), response('finalize_plan', args)])
    outcome = data_agent.propose(tmp_path, run['id'], client=SimpleNamespace(models=fake), execute=True)
    assert outcome['state'] == 'completed' and outcome['result']['checkpoint_reloaded']
    assert {x['role'] for x in outcome['specialist_findings'] if x['status'] == 'completed'} == {'data', 'model'}
    assert len(outcome['result']['attempts']) == 2
    assert [e['node'] for e in outcome['events'] if e['status'] in ('complete', 'passed')][-2:] == ['execute', 'verify']
    assert (tmp_path / run['id'] / 'agent_events.jsonl').exists()


def test_explicit_huggingface_model_request_runs_discovery(tmp_path, monkeypatch):
    from webapp import hf_models, model_backends
    monkeypatch.setattr(model_backends, 'catalog', lambda: [])
    monkeypatch.setattr(hf_models, 'search_mcp', lambda task: (_ for _ in ()).throw(OSError('offline')))
    content, features = make_csv(load_iris())
    run = classifier.inspect_csv(tmp_path, content, '請搜尋 Hugging Face 模型並預測類別')
    found = [{'id': 'example/model', 'url': 'https://huggingface.co/example/model', 'license': 'mit'}]
    seen = []
    monkeypatch.setattr(hf_models, 'search', lambda task: seen.append(task) or found)
    args = {'task': 'classification', 'target': 'target', 'features': features}
    fake = FakeModels([response('inspect_plan', args), response('finalize_plan', args)])
    result = data_agent.propose(tmp_path, run['id'], client=SimpleNamespace(models=fake), execute=True)
    assert result['state'] == 'completed' and seen == ['classification']
    assert result['model_discovery']['results'] == found
    assert (tmp_path / run['id'] / 'model_discovery.json').is_file()


def test_huggingface_search_failure_is_visible_but_training_continues(tmp_path, monkeypatch):
    from webapp import hf_models, model_backends
    monkeypatch.setattr(model_backends, 'catalog', lambda: [])
    monkeypatch.setattr(hf_models, 'search_mcp', lambda task: (_ for _ in ()).throw(OSError('offline')))
    content, features = make_csv(load_iris())
    run = classifier.inspect_csv(tmp_path, content, '搜尋 Hugging Face 模型並預測類別')
    def fail(task): raise OSError('network unavailable')
    monkeypatch.setattr(hf_models, 'search', fail)
    args = {'task': 'classification', 'target': 'target', 'features': features}
    fake = FakeModels([response('inspect_plan', args), response('finalize_plan', args)])
    result = data_agent.propose(tmp_path, run['id'], client=SimpleNamespace(models=fake), execute=True)
    assert result['state'] == 'completed' and result['model_discovery']['status'] == 'failed'
    assert result['model_discovery']['results'] == []


def test_explicit_unavailable_huggingface_repo_is_not_silently_replaced(tmp_path, monkeypatch):
    from webapp import hf_models, model_backends
    monkeypatch.setattr(hf_models, 'search_mcp', lambda task: [])
    monkeypatch.setattr(model_backends, 'catalog', lambda: [{'id': 'other', 'label': 'Other', 'repo': 'x/other', 'available': True, 'tasks': ['classification']}])
    content, features = make_csv(load_iris())
    run = classifier.inspect_csv(tmp_path, content, '用 Hugging Face 模型 author/requested 比較分類')
    args = {'task': 'classification', 'target': 'target', 'features': features}
    fake = FakeModels([response('inspect_plan', args)])
    result = data_agent.propose(tmp_path, run['id'], client=SimpleNamespace(models=fake), execute=True)
    assert result['state'] == 'completed'
    assert 'no compatible installed adapter' in result['model_discovery']['decision']
    assert not any(item.get('backend') for item in result['result']['attempts'])


def test_kaggle_dataset_id_is_not_mistaken_for_huggingface_model(tmp_path, monkeypatch):
    import json
    from sklearn.dummy import DummyClassifier
    from webapp import hf_models, model_backends
    class Backend:
        def describe(self):
            return {'label': 'Example HF', 'repo': 'hf/model', 'tasks': ['classification'], 'available': True, 'method': 'fit'}
        def create(self, task):
            return DummyClassifier(strategy='most_frequent')
    monkeypatch.setattr(model_backends, 'registry', lambda: {'example': Backend()})
    monkeypatch.setattr(hf_models, 'search_mcp', lambda task: [])
    content, features = make_csv(load_iris())
    run = classifier.inspect_csv(tmp_path, content, '使用 Kaggle example/dataset 並比較 Hugging Face 模型')
    (tmp_path / run['id'] / 'source.json').write_text(json.dumps({'kind': 'dataset', 'ref': 'example/dataset', 'url': 'https://www.kaggle.com/datasets/example/dataset'}))
    args = {'task': 'classification', 'target': 'target', 'features': features}
    fake = FakeModels([response('inspect_plan', args)])
    result = data_agent.propose(tmp_path, run['id'], client=SimpleNamespace(models=fake), execute=True)
    assert result['model_discovery']['selected_backend'] == 'example'
    assert any(item.get('backend') == 'example' for item in result['result']['attempts'])


def test_explicit_four_model_comparison_is_honored(tmp_path):
    content, features = make_csv(load_iris())
    run = classifier.inspect_csv(tmp_path, content, '請比較四個模型預測分類')
    args = {'task': 'classification', 'target': 'target', 'features': features, 'model_count': 4}
    fake = FakeModels([response('inspect_plan', args), response('finalize_plan', args)])
    outcome = data_agent.propose(tmp_path, run['id'], client=SimpleNamespace(models=fake), execute=True)
    assert outcome['state'] == 'completed'
    assert len(outcome['result']['attempts']) == 4


def test_agent_does_not_ask_user_to_pick_models_for_a_count(tmp_path):
    content, features = make_csv(load_iris())
    run = classifier.inspect_csv(tmp_path, content, '請比較四個模型預測分類')
    args = {'task': 'classification', 'target': 'target', 'features': features, 'model_count': 4}
    question = {'question': '請選四個模型', 'options': ['Logistic Regression', 'Random Forest']}
    fake = FakeModels([response('ask_clarification', question), response('inspect_plan', args), response('finalize_plan', args)])
    result = data_agent.propose(tmp_path, run['id'], client=SimpleNamespace(models=fake), execute=True)
    assert result['state'] == 'completed' and len(result['result']['attempts']) == 4


@pytest.mark.parametrize('count', [3, 5])
def test_requested_model_count_is_not_fixed(tmp_path, count):
    content, features = make_csv(load_iris())
    run = classifier.inspect_csv(tmp_path, content, f'請比較 {count} 個模型預測分類')
    args = {'task': 'classification', 'target': 'target', 'features': features, 'model_count': count}
    fake = FakeModels([response('inspect_plan', args), response('finalize_plan', args)])
    outcome = data_agent.propose(tmp_path, run['id'], client=SimpleNamespace(models=fake), execute=True)
    assert outcome['state'] == 'completed'
    assert len(outcome['result']['attempts']) == count


def test_graph_replans_once_after_training_error(tmp_path, monkeypatch):
    content, features = make_csv(load_iris())
    run = classifier.inspect_csv(tmp_path, content, '請依特徵預測分類')
    first = {'task': 'classification', 'target': 'target', 'features': features}
    second = {'task': 'classification', 'target': 'target', 'features': features[1:]}
    fake = FakeModels([response('inspect_plan', first), response('finalize_plan', first), response('inspect_plan', second), response('finalize_plan', second)])
    real_train = tabular.train
    count = []
    def fail_once(*args, **kwargs):
        count.append(1)
        if len(count) == 1:
            raise ValueError('simulated local failure')
        return real_train(*args, **kwargs)
    monkeypatch.setattr(tabular, 'train', fail_once)
    outcome = data_agent.propose(tmp_path, run['id'], client=SimpleNamespace(models=fake), execute=True)
    assert outcome['state'] == 'completed' and len(count) == 2
    assert any(e['node'] == 'critic' and e['status'] == 'replanning' for e in outcome['events'])


def test_graph_asks_for_clarification_without_training(tmp_path):
    content, _ = make_csv(load_iris())
    run = classifier.inspect_csv(tmp_path, content, '幫我分析')
    fake = FakeModels([response(text='請問要預測哪一欄？')])
    outcome = data_agent.propose(tmp_path, run['id'], client=SimpleNamespace(models=fake), execute=True)
    assert outcome['state'] == 'needs_input' and '哪一欄' in outcome['message']
    assert not (tmp_path / run['id'] / 'tabular.joblib').exists()


def test_agent_clarification_returns_dynamic_choices(tmp_path):
    fake = FakeModels([response('ask_clarification', {'question': '要使用哪份資料？', 'options': ['上傳 CSV', '搜尋公開資料']}), response(text='{"question":"Which data source?","options":["Upload CSV","Search public datasets"]}')])
    result = data_agent.converse(tmp_path, [{'role': 'user', 'text': '我想訓練模型'}], client=SimpleNamespace(models=fake))
    assert result['options'] == ['Upload CSV', 'Search public datasets']


def test_clarifications_persist_and_task_budget_counts_across_turns(tmp_path):
    content, _ = make_csv(load_iris())
    run = classifier.inspect_csv(tmp_path, content, '幫我分析')
    first = FakeModels([response(text='請問目標欄位？')])
    data_agent.propose(tmp_path, run['id'], '希望做分類', client=SimpleNamespace(models=first))
    second = FakeModels([response(text='還需要更多說明')])
    data_agent.propose(tmp_path, run['id'], '目標欄位是 target', client=SimpleNamespace(models=second))
    sent = second.requests[0]['contents'][0].parts[0].text
    assert '希望做分類' in sent and '目標欄位是 target' in sent
    assert data_agent.budget(tmp_path, run_id=run['id'])['task_used'] == 2


def test_unsupported_tool_cannot_execute(tmp_path):
    content, _ = make_csv(load_iris())
    run = classifier.inspect_csv(tmp_path, content, '測試')
    fake = FakeModels([response('delete_files', {'task': 'classification', 'target': 'target', 'features': []}), response(text='無法執行')])
    result = data_agent.propose(tmp_path, run['id'], client=SimpleNamespace(models=fake), execute=True)
    assert result['state'] == 'needs_input'
    assert not (tmp_path / run['id'] / 'tabular.joblib').exists()


def test_open_conversation_without_a_dataset(tmp_path):
    fake = FakeModels([response(text='請告訴我你想用什麼資料，預測什麼結果。')])
    result = data_agent.converse(tmp_path, [{'role': 'user', 'text': '我想預測銷售量'}], client=SimpleNamespace(models=fake))
    assert '資料' in result['message']
    assert data_agent.budget(tmp_path)['used'] == 1


def test_followup_approval_executes_prior_improvement_without_reasking(tmp_path):
    data = load_iris()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['PassengerId', 'Name', 'Ticket', 'Cabin', 'length', 'width', 'target'])
    for i, (values, label) in enumerate(zip(data.data, data.target)):
        writer.writerow([i + 1, f'person-{i}', f'ticket-{i}', f'cabin-{i}', values[0], values[1], int(label)])
    run = classifier.inspect_csv(tmp_path, output.getvalue(), '分類')
    features = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'length', 'width']
    tabular.train(tmp_path, run['id'], 'classification', 'target', features, options={'model_count': 4, 'auto_ablation': False})
    history = [{'role': 'assistant', 'text': '建議排除 PassengerId、Name、Ticket、Cabin，然後重新訓練比較。'},
               {'role': 'user', 'text': '可以直接幫我去實作嗎'}, {'role': 'assistant', 'text': '是否要排除這些欄位？'}]
    fake = FakeModels([response(text='請再確認欄位。')])
    answer = data_agent.respond(tmp_path, run['id'], '你直接去幫我測試', client=SimpleNamespace(models=fake), history=history)
    assert answer['type'] == 'variant_request'
    assert answer['features'] == ['length', 'width'] and answer['options']['model_count'] == 4
    from webapp import variants
    version = variants.create(tmp_path, run['id'], answer['features'], answer['options'])
    assert version['state'] == 'completed' and version['comparison']['comparable']
    assert len(version['result']['attempts']) == 4 and version['result']['proposal']['features'] == ['length', 'width']


def test_followup_requires_action_and_keeps_other_columns():
    features = ['PassengerId', 'Name', 'Age', 'Fare']
    history = [{'role': 'assistant', 'text': '建議排除 PassengerId、Name，並重新訓練。'}]
    assert data_agent._approved_feature_improvement('可以', history, features) == ['Age', 'Fare']
    assert data_agent._approved_feature_improvement('先不要實作，只分析', history, features) is None
    assert data_agent._approved_feature_improvement('可以', [], features) is None
    assert data_agent._approved_feature_improvement('可以分析結果嗎？', history, features) is None
    assert data_agent._approved_feature_improvement('直接測試，但保留 Name', history, features) == ['Name', 'Age', 'Fare']


def test_dataset_search_requires_explicit_user_request(tmp_path, monkeypatch):
    fake = FakeModels([response('search_datasets', {'query': 'sentiment'})])
    monkeypatch.setattr(classifier, 'search_datasets', lambda query: [{'id': 'example/public', 'url': 'https://huggingface.co/datasets/example/public', 'license': 'unknown'}])
    result = data_agent.converse(tmp_path, [{'role': 'user', 'text': '請搜尋 sentiment 資料集'}], client=SimpleNamespace(models=fake))
    assert result['datasets'][0]['id'] == 'example/public'
    unauthorized = FakeModels([response('search_datasets', {'query': 'sentiment'})])
    with pytest.raises(ValueError, match='explicit'):
        data_agent.converse(tmp_path, [{'role': 'user', 'text': '介紹 sentiment 分類'}], client=SimpleNamespace(models=unauthorized))


def test_agent_uses_kaggle_tool_only_for_a_resolved_source(tmp_path, monkeypatch):
    from webapp import kaggle_tools
    monkeypatch.setattr(kaggle_tools, 'search', lambda query, kind: [{'ref': 'titanic', 'kind': 'competition', 'url': 'https://www.kaggle.com/competitions/titanic'}])
    monkeypatch.setattr(kaggle_tools, 'prepare', lambda root, kind, ref, goal, filename: {'state': 'prepared', 'dataset': {'id': 'example'}, 'source': {'ref': ref}})
    fake = FakeModels([response('prepare_kaggle', {'query': 'titanic', 'kind': 'competition', 'filename': ''})])
    result = data_agent.converse(tmp_path, [{'role': 'user', 'text': '想解 Kaggle Titanic'}], client=SimpleNamespace(models=fake))
    assert result['prepared']['source']['ref'] == 'titanic'


def test_explicit_kaggle_training_precedes_hf_research(tmp_path, monkeypatch):
    from webapp import kaggle_tools
    captured = {}
    def prepare(root, kind, ref, goal, filename):
        captured.update(kind=kind, ref=ref, goal=goal)
        return {'state': 'prepared', 'dataset': {'id': 'example'}, 'source': {'ref': ref}}
    monkeypatch.setattr(kaggle_tools, 'prepare', prepare)
    request = '使用 Kaggle 資料集 example/cancer-data，透過 Hugging Face MCP 搜尋模型並比較四個模型預測。'
    result = data_agent.converse(tmp_path, [{'role': 'user', 'text': request}], client=SimpleNamespace(models=FakeModels([])))
    assert result['prepared']['source']['ref'] == 'example/cancer-data'
    assert captured['kind'] == 'dataset' and captured['goal'] == request


def test_csv_blank_header_keeps_values_and_discards_empty_column():
    fields, rows = classifier.parse_csv('diagnosis,,\nM,7,\nB,8,\n')
    assert fields == ['diagnosis', 'unnamed_column_2']
    assert rows == [{'diagnosis': 'M', 'unnamed_column_2': '7'}, {'diagnosis': 'B', 'unnamed_column_2': '8'}]
    with pytest.raises(ValueError, match='unique'):
        classifier.parse_csv('x,x\n1,2\n')


def test_kaggle_ambiguity_gives_choices_and_preserves_original_goal(tmp_path, monkeypatch):
    from webapp import kaggle_tools
    sources = [{'ref': 'one/titanic', 'kind': 'dataset', 'title': 'Titanic', 'url': 'https://www.kaggle.com/datasets/one/titanic'}, {'ref': 'two/titanic', 'kind': 'dataset', 'title': 'Titanic', 'url': 'https://www.kaggle.com/datasets/two/titanic'}]
    monkeypatch.setattr(kaggle_tools, 'search', lambda query, kind: sources)
    captured = {}
    def prepare(root, kind, ref, goal, filename):
        captured['goal'] = goal
        return {'state': 'prepared', 'dataset': {'id': 'example'}, 'source': {'ref': ref}}
    monkeypatch.setattr(kaggle_tools, 'prepare', prepare)
    user_goal = '使用 Kaggle Titanic 比較四個模型'
    fake = FakeModels([response('prepare_kaggle', {'query': 'titanic', 'kind': 'dataset', 'filename': ''})])
    answer = data_agent.converse(tmp_path, [{'role': 'user', 'text': user_goal}], client=SimpleNamespace(models=fake))
    assert len(answer['options']) == 2
    fake = FakeModels([response('prepare_kaggle', {'query': 'one/titanic', 'kind': 'dataset', 'filename': ''})])
    result = data_agent.converse(tmp_path, [{'role': 'user', 'text': user_goal}, {'role': 'assistant', 'text': answer['message']}, {'role': 'user', 'text': answer['options'][0]}], client=SimpleNamespace(models=fake))
    assert result['prepared']['source']['ref'] == 'one/titanic' and user_goal in captured['goal']


def test_kaggle_csv_choice_is_english_and_resumes_selected_source(tmp_path, monkeypatch):
    from webapp import kaggle_tools
    calls = []
    def prepare(root, kind, ref, goal, filename):
        calls.append((kind, ref, filename, goal))
        if not filename:
            return {'state': 'needs_file', 'candidates': ['train_clean.csv', 'test_clean.csv']}
        return {'state': 'prepared', 'dataset': {'id': 'example'}, 'source': {'ref': ref}}
    monkeypatch.setattr(kaggle_tools, 'prepare', prepare)
    history = [{'role': 'user', 'text': 'Do not train yet; inspect the data.'}, {'role': 'user', 'text': 'Use Kaggle dataset owner/data'}]
    answer = data_agent.converse(tmp_path, history, client=SimpleNamespace(models=FakeModels([])))
    assert answer['options'] == ['Use Kaggle dataset owner/data file train_clean.csv', 'Use Kaggle dataset owner/data file test_clean.csv']
    result = data_agent.converse(tmp_path, history + [{'role': 'assistant', 'text': answer['message']}, {'role': 'user', 'text': answer['options'][0]}], client=SimpleNamespace(models=FakeModels([])))
    assert result['prepared']['source']['ref'] == 'owner/data'
    assert calls[-1][2] == 'train_clean.csv' and 'Do not train yet' in calls[-1][3]


def test_inspect_only_goal_does_not_start_training(tmp_path):
    run = classifier.inspect_csv(tmp_path, 'x,target\n1,0\n2,1\n', 'Do NOT train any models yet.')
    answer = data_agent.propose(tmp_path, run['id'], execute=True)
    assert answer['state'] == 'needs_input'
    assert 'not started' in answer['message']
    answer = data_agent.propose(tmp_path, run['id'], clarification='Run 5-fold cross-validation with LightGBM', execute=True)
    assert answer['state'] == 'failed' and 'No experiment was started' in answer['message']


def test_inspection_returns_local_samples_without_sending_rows_to_model(tmp_path):
    run = classifier.inspect_csv(tmp_path, 'Cabin,target\nB/0/P,1\n,0\n', 'Do not train yet')
    fake = FakeModels([response(text='Two potential plans.')])
    result = data_agent.inspect_only(tmp_path, run['id'], client=SimpleNamespace(models=fake))
    assert result['examples'][0]['Cabin'] == 'B/0/P'
    assert result['columns'][0]['missing_percent'] == 50
    assert 'B/0/P' not in fake.requests[0]['contents']
    assert result['message'] == 'Two potential plans.'


def test_english_model_count_survives_target_clarification_without_reasking(tmp_path):
    content, features = make_csv(load_iris())
    run = classifier.inspect_csv(tmp_path, content, 'Compare four models and include a compatible Hugging Face model.')
    args = {'task': 'classification', 'target': 'target', 'features': features, 'model_count': 2}
    fake = FakeModels([response('ask_clarification', {'question': '要我用四個模型並包含 Hugging Face 嗎？', 'options': ['是', '否']}), response('inspect_plan', args), response('finalize_plan', args)])
    result = data_agent.propose(tmp_path, run['id'], clarification='target', client=SimpleNamespace(models=fake))
    assert result['state'] == 'ready'
    assert result['plan']['model_count'] == 4
    assert any(x.get('status') == 'answered_by_capabilities' for x in result['trace'])


def test_chinese_clarification_is_translated_to_english(tmp_path):
    import json
    fake = FakeModels([response(text=json.dumps({'question': 'Which target column?', 'options': ['target', 'outcome']}))])
    call = SimpleNamespace(args={'question': '請選擇目標欄位', 'options': ['目標', '結果']})
    result = data_agent._clarification(call, tmp_path, SimpleNamespace(models=fake))
    assert result == {'message': 'Which target column?', 'options': ['target', 'outcome']}


def test_english_message_translates_assistant_text_without_changing_data(tmp_path, monkeypatch):
    monkeypatch.setattr(data_agent, '_client', lambda: object())
    monkeypatch.setattr(data_agent, '_generate', lambda *args, **kwargs: response(text='Please confirm the target column.'))
    result = data_agent.english_message(tmp_path, {'message': '請確認目標欄位', 'dataset': {'name': '中文資料'}})
    assert result == {'message': 'Please confirm the target column.', 'dataset': {'name': '中文資料'}}
