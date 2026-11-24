import csv
import io

from fastapi.testclient import TestClient
from webapp.main import create_app


def csv_data():
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["text", "label"])
    for label, subject in [("refund", "refund billing"), ("shipping", "delivery parcel"), ("product", "product damaged")]:
        for i in range(35):
            writer.writerow([f"{subject} question number {i}", label])
    return output.getvalue()


def test_classifier_end_to_end(tmp_path):
    client = TestClient(create_app(runs_root=tmp_path))
    preview = client.post('/api/classifier/inspect', json={'csv': csv_data()}).json()
    assert preview['rows'] == 105 and len(preview['examples']) == 3
    body = {'id': preview['id'], 'text_column': 'text', 'label_column': 'label'}
    proposal = client.post('/api/classifier/plan', json=body).json()
    assert proposal['ready'] and proposal['unique_rows'] == 105
    response = client.post('/api/classifier/train', json=body)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['checkpoint_reloaded'] and result['metrics']['macro_f1'] > .8
    assert sum(result['split_counts'].values()) == 105
    prediction = client.post('/api/classifier/predict', json={'id': preview['id'], 'text': 'refund for billing'}).json()
    assert prediction['label'] == 'refund'
    assert 'Held-out test macro F1' in client.get('/api/classifier/' + preview['id'] + '/report').text
    assert client.post('/api/classifier/train', json=body).status_code == 400


def test_rejects_insufficient_conflicting_and_invalid_input(tmp_path):
    client = TestClient(create_app(runs_root=tmp_path))
    assert client.post('/api/classifier/inspect', json={'csv': 'x,x\na,b'}).status_code == 400
    preview = client.post('/api/classifier/inspect', json={'csv': csv_data() + 'refund billing question number 0,shipping\n'}).json()
    plan = client.post('/api/classifier/plan', json={'id': preview['id'], 'text_column': 'text', 'label_column': 'label'}).json()
    assert not plan['ready'] and any('conflicting labels' in w for w in plan['warnings'])
    assert client.post('/api/classifier/predict', json={'id': '../../outside', 'text': 'hello'}).status_code == 400
