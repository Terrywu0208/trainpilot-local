import pytest
from webapp import hf_mcp, hf_models


def test_mcp_search_arguments_and_result_parsing(monkeypatch):
    seen = []
    def fake(tool, args):
        seen.append((tool, args))
        return '### example/table-model\n**Tags:** license:apache-2.0\n'
    monkeypatch.setattr(hf_mcp, 'call', fake)
    result = hf_models.search_mcp('classification')
    assert seen[0][0] == 'hub_repo_search'
    assert seen[0][1]['filters'] == ['tabular-classification']
    assert result[0]['id'] == 'example/table-model' and result[0]['license'] == 'apache-2.0'


def test_mcp_only_allows_read_only_tools_and_valid_ids():
    with pytest.raises(ValueError):
        hf_mcp.call('create_job', {})
    with pytest.raises(ValueError):
        hf_mcp.details('../private')


def test_generic_empty_model_search_retries_with_task_tag(monkeypatch):
    requests = []
    def fake(tool, args):
        requests.append(args)
        return 'Found 0 repositories' if len(requests) == 1 else '### example/tabular\n'
    monkeypatch.setattr(hf_mcp, 'call', fake)
    result = hf_mcp.search('tabular classification', 'model')
    assert 'example/tabular' in result
    assert requests[1]['filters'] == ['tabular-classification'] and 'query' not in requests[1]
