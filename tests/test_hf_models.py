import json
from webapp import hf_models


def test_hub_search_reads_metadata_only(monkeypatch):
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, n):
            return json.dumps([{'id': 'example/tabular', 'pipeline_tag': 'tabular-classification',
                                'library_name': 'sklearn', 'tags': ['license:mit']}]).encode()
    def fake_open(request, timeout):
        assert 'filter=tabular-classification' in request.full_url and timeout == 12
        return Response()
    monkeypatch.setattr(hf_models, 'urlopen', fake_open)
    found = hf_models.search('classification')
    assert found[0]['id'] == 'example/tabular' and found[0]['license'] == 'mit'
    assert found[0]['url'] == 'https://huggingface.co/example/tabular'
