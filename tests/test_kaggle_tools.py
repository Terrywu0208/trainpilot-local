import json
from pathlib import Path
import pytest
from webapp import kaggle_tools


def test_competition_search_and_prepare(tmp_path, monkeypatch):
    def fake_run(*args):
        if args[:2] == ('competitions', 'list'):
            return json.dumps([{'ref': 'https://www.kaggle.com/competitions/titanic'}])
        if args[:2] == ('competitions', 'files'):
            return json.dumps([{'name': 'train.csv', 'size': 100}])
        if args[:3] == ('competitions', 'pages', 'list'):
            return json.dumps([{'content': 'Predict passenger survival'}])
        if args[:2] == ('competitions', 'download'):
            destination = Path(args[args.index('-p') + 1])
            (destination / 'train.csv').write_text('x,y\n1,yes\n2,no\n')
            return ''
        raise AssertionError(args)
    monkeypatch.setattr(kaggle_tools, '_run', fake_run)
    assert kaggle_tools.search('titanic')[0]['ref'] == 'titanic'
    data = kaggle_tools.prepare(tmp_path, 'competition', 'titanic', 'predict survival')
    assert data['state'] == 'prepared' and data['dataset']['rows'] == 2
    assert (tmp_path / data['dataset']['id'] / 'source.json').exists()
    with pytest.raises(ValueError):
        kaggle_tools.prepare(tmp_path, 'competition', '../secret', 'x')
