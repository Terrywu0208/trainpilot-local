from webapp.report_page import _chart, _paired_chart, _split_chart


def test_signed_chart_and_missing_values():
    chart = _chart('分群', [('負值', -0.5), ('正值', 0.25), ('缺失', None)], fixed_range=(-1, 1))
    assert 'left:50.00%' in chart
    assert 'left:25.00%;width:25.00%' in chart
    assert '-0.500' in chart and '0.250' in chart
    assert '缺失' not in chart


def test_paired_chart_matches_models_and_escapes_labels():
    chart = _paired_chart('比較', [{'model': '<model>', 'value': 0.4}, {'model': 'missing', 'value': 1}], [{'model': '<model>', 'value': 0.6}])
    assert '&lt;model&gt; · Before' in chart
    assert '&lt;model&gt; · After' in chart
    assert 'missing' not in chart


def test_split_chart_uses_available_counts():
    chart = _split_chart({'train': 60, 'validation': 20, 'test': 20})
    assert 'Training set 60 rows（60.0%）' in chart
    assert 'Test set 20 rows（20.0%）' in chart
    assert _split_chart({}) == ''
