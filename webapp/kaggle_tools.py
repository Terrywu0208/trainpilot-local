"""Read-only, bounded Kaggle discovery and CSV download helpers."""
from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import zipfile
from difflib import SequenceMatcher
from pathlib import Path

from webapp import classifier

ROOT = Path(__file__).resolve().parent.parent
MAX_DOWNLOAD = 2_000_000


def _environment() -> dict:
    env = os.environ.copy()
    if not env.get('KAGGLE_API_TOKEN'):
        path = ROOT / '.kaggle.local'
        if path.is_file():
            for line in path.read_text(encoding='utf-8-sig').splitlines():
                name, sep, value = line.partition('=')
                if sep and name.strip() == 'KAGGLE_API_TOKEN':
                    env['KAGGLE_API_TOKEN'] = value.strip().strip('"').strip("'")
                    break
    if not env.get('KAGGLE_API_TOKEN'):
        raise ValueError('尚未設定 Kaggle API token，無法讀取 Kaggle。')
    return env


def _run(*args: str) -> str:
    binary = Path(sys.executable).parent / ('kaggle.exe' if os.name == 'nt' else 'kaggle')
    try:
        result = subprocess.run([str(binary), *args], capture_output=True, text=True,
                                timeout=50, env=_environment(), check=True)
    except subprocess.TimeoutExpired as exc:
        raise ValueError('Kaggle 連線逾時，請稍後再試。') from exc
    except subprocess.CalledProcessError as exc:
        # Do not expose CLI errors: they may include credentials or private paths.
        raise ValueError('Kaggle 拒絕或無法完成讀取。該題可能需要先登入並在網站接受規則。') from exc
    except FileNotFoundError as exc:
        raise ValueError('尚未安裝 Kaggle 官方 CLI。') from exc
    return result.stdout


def search(query: str, kind: str = 'competition') -> list[dict]:
    if not query.strip() or len(query) > 100 or kind not in ('competition', 'dataset'):
        raise ValueError('請提供有效的 Kaggle 搜尋詞與類型。')
    command = ['competitions', 'list'] if kind == 'competition' else ['datasets', 'list']
    output = _run(*command, '-s', query.strip(), '--format', 'json')
    try:
        results = json.loads(output)
    except json.JSONDecodeError as exc:
        raise ValueError('無法解析 Kaggle 搜尋結果。') from exc
    found = []
    for item in results[:10]:
        ref = item.get('ref') or item.get('id')
        if not isinstance(ref, str):
            continue
        ref = ref.rstrip('/').split('/competitions/')[-1] if kind == 'competition' else ref
        valid = r'[a-zA-Z0-9][a-zA-Z0-9_-]*' if kind == 'competition' else r'[a-zA-Z0-9][a-zA-Z0-9_-]*/[a-zA-Z0-9][a-zA-Z0-9_-]*'
        if re.fullmatch(valid, ref):
            found.append({'ref': ref, 'kind': kind, 'title': item.get('title') or ref,
                          'url': ('https://www.kaggle.com/competitions/' if kind == 'competition' else 'https://www.kaggle.com/datasets/') + ref})
    return found


def describe_competition(ref: str) -> dict:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', ref):
        raise ValueError('無效的 Kaggle 題目 ID。')
    files = json.loads(_run('competitions', 'files', ref, '--format', 'json'))
    pages = json.loads(_run('competitions', 'pages', 'list', '--format', 'json', '--content', '--page-name', 'description', ref))
    text = ' '.join(re.sub(r'<[^>]+>', ' ', str(p.get('content', ''))) for p in pages)
    return {'ref': ref, 'kind': 'competition', 'url': 'https://www.kaggle.com/competitions/' + ref,
            'description': text[:3000], 'files': [{'name': f.get('name'), 'size': f.get('size')} for f in files]}


def prepare(root: Path, kind: str, ref: str, goal: str, filename: str = '') -> dict:
    if kind not in ('competition', 'dataset'):
        raise ValueError('Kaggle 類型必須為 competition 或 dataset。')
    pattern = r'[A-Za-z0-9][A-Za-z0-9_-]*' if kind == 'competition' else r'[A-Za-z0-9][A-Za-z0-9_-]*/[A-Za-z0-9][A-Za-z0-9_-]*'
    if not re.fullmatch(pattern, ref):
        raise ValueError('無效的 Kaggle 資料 ID。')
    if kind == 'competition':
        details = describe_competition(ref)
        files = details['files']
    else:
        raw = json.loads(_run('datasets', 'files', ref, '--format', 'json'))
        files = raw if isinstance(raw, list) else raw.get('files', [])
        details = {'ref': ref, 'kind': kind, 'url': 'https://www.kaggle.com/datasets/' + ref,
                   'description': '', 'files': files}
    candidates = [f for f in files if isinstance(f.get('name'), str) and re.fullmatch(r'[A-Za-z0-9_. -]+\.csv', f['name'], re.I)]
    selected = next((f for f in candidates if f['name'] == filename), None) if filename else next((f for f in candidates if f['name'].lower() == 'train.csv'), None)
    if not selected and not filename and len(candidates) == 1:
        selected = candidates[0]
    if not selected:
        return {'state': 'needs_file', 'source': details, 'candidates': [f['name'] for f in candidates[:20]]}
    if selected.get('size') is not None and int(selected['size']) > MAX_DOWNLOAD:
        raise ValueError('所選 CSV 超過 2 MB 本機測試上限，請選擇較小的資料。')
    import tempfile
    with tempfile.TemporaryDirectory(prefix='trainpilot-kaggle-', dir=str(root.parent)) as tmp:
        args = ['competitions', 'download', ref] if kind == 'competition' else ['datasets', 'download', ref]
        _run(*args, '-f', selected['name'], '-p', tmp, '-q')
        path = Path(tmp) / selected['name']
        if path.is_file():
            raw_data = path.read_bytes()
        else:
            archives = list(Path(tmp).glob('*.zip'))
            if len(archives) != 1:
                raise ValueError('Kaggle 沒有回傳預期的 CSV。')
            with zipfile.ZipFile(archives[0]) as z:
                matching = [entry for entry in z.infolist() if entry.filename == selected['name'] and entry.file_size <= MAX_DOWNLOAD]
                if len(matching) != 1:
                    raise ValueError('壓縮檔沒有單一且符合大小限制的 CSV。')
                raw_data = z.read(matching[0])
        if len(raw_data) > MAX_DOWNLOAD:
            raise ValueError('資料超過 2 MB 本機測試上限。')
        try:
            content = raw_data.decode('utf-8-sig')
        except UnicodeDecodeError as exc:
            raise ValueError('目前只支援 UTF-8 CSV。') from exc
    result = classifier.inspect_csv(root, content, goal)
    source = {'kind': kind, 'ref': ref, 'url': details['url'], 'filename': selected['name'],
              'description': details['description']}
    (root / result['id'] / 'source.json').write_text(json.dumps(source, ensure_ascii=False), encoding='utf-8')
    return {'state': 'prepared', 'dataset': result, 'source': source}
