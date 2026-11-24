import csv
import io
import json
from types import SimpleNamespace

import pytest
from sklearn.datasets import load_iris
from webapp import classifier, data_agent, tabular


@pytest.fixture
def experiment(tmp_path):
    iris = load_iris()
    stream = io.StringIO()
    writer = csv.writer(stream)
    features = [f'feature_{i}' for i in range(4)]
    writer.writerow([*features, 'target'])
    for x, y in zip(iris.data, iris.target):
        writer.writerow([*x, y])
    run = classifier.inspect_csv(tmp_path, stream.getvalue(), 'Classify flowers')
    tabular.train(tmp_path, run['id'], 'classification', 'target', features)
    return tmp_path, run['id'], features


def response(name, args):
    part = SimpleNamespace(function_call=SimpleNamespace(name=name, args=args), text=None, thought=False)
    return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))])


@pytest.mark.parametrize('message', [
    'I need a detail intro for Logistic Regression, Hugging Face PredictLM 26M',
    'Explain how to train a classifier', '現在還有哪裡可以優化？',
    'Should I train a different model?', 'What model should I train?',
    'Explain the training and test performance',
    '請介紹這個模型是怎麼訓練的',
    'Why did you train this model?',
    'Explain how you train this model',
])
def test_read_only_question_never_exposes_training_tool(experiment, monkeypatch, message):
    root, run_id, _ = experiment
    seen = {}
    def generate(*args, **kwargs):
        seen['input'] = json.loads(kwargs['contents'])
        seen['tools'] = [item.name for group in kwargs['config'].tools for item in group.function_declarations]
        return response('answer_user', {'message': 'Here is the requested explanation.'})
    monkeypatch.setattr(data_agent, '_generate', generate)
    before = set(root.iterdir())
    result = data_agent.respond(root, run_id, message, client=object())
    assert result['type'] == 'answer' and 'retrain_variant' not in seen['tools']
    assert set(root.iterdir()) == before
    assert 'predict_ms_per_row' in seen['input']['experiment']['validation_models'][0]
    assert 'test_confusion_matrix' in seen['input']['experiment']


def test_training_confirmation_and_combined_intent_are_actionable(experiment, monkeypatch):
    root, run_id, features = experiment
    def generate(*args, **kwargs):
        tools = [item.name for group in kwargs['config'].tools for item in group.function_declarations]
        assert 'retrain_variant' in tools
        return response('retrain_variant', {'features': features, 'regularization': .5, 'explanation': 'Logistic regression is a linear classifier.'})
    monkeypatch.setattr(data_agent, '_generate', generate)
    for message, history in [
        ('Explain the model and then tune it', []),
        ('好，就照你建議的方案幫我跑一次驗證看看', []),
        ('yes', [{'role': 'assistant', 'text': 'Would you like me to run a variant with regularization 0.5?'}]),
    ]:
        result = data_agent.respond(root, run_id, message, client=object(), history=history)
        assert result['type'] == 'variant_request'
        assert result['message'].startswith('Logistic regression is a linear classifier.')


def test_read_only_guard_rejects_unexpected_training_call(experiment, monkeypatch):
    root, run_id, features = experiment
    monkeypatch.setattr(data_agent, '_generate', lambda *args, **kwargs: response('retrain_variant', {'features': features, 'regularization': 1.0}))
    with pytest.raises(ValueError, match='not authorized'):
        data_agent.respond(root, run_id, 'Explain Logistic Regression', client=object())


def test_malformed_options_fall_back_to_free_text():
    call = SimpleNamespace(args={'question': 'What would you like?', 'options': ['valid', None, '', 'x' * 121]})
    assert data_agent._clarification(call) == {'message': 'What would you like?', 'options': ['valid']}
    call.args['options'] = None
    assert data_agent._clarification(call)['options'] == []
