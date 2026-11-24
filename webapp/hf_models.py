"""Read-only, bounded Hugging Face Hub model discovery."""
from __future__ import annotations

import json
import re
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def search_mcp(task: str) -> list[dict]:
    from webapp import hf_mcp
    tags = {'classification': 'tabular-classification', 'regression': 'tabular-regression'}
    if task not in tags:
        return []
    text = hf_mcp.search(repo_type='model', task=tags[task], limit=5)
    results = []
    for chunk in re.split(r'(?m)^### ', text)[1:]:
        model_id = chunk.splitlines()[0].strip()
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*', model_id):
            continue
        license_match = re.search(r'license:([A-Za-z0-9_.-]+)', chunk)
        results.append({'id': model_id, 'url': 'https://huggingface.co/' + model_id,
                        'task': tags[task], 'library': 'Not specified',
                        'license': license_match.group(1) if license_match else 'Not specified'})
    return results[:5]


def search(task: str) -> list[dict]:
    tags = {'classification': 'tabular-classification', 'regression': 'tabular-regression'}
    if task not in tags:
        return []
    query = urlencode({'filter': tags[task], 'sort': 'downloads', 'direction': '-1', 'limit': 8, 'full': 'true'})
    request = Request('https://huggingface.co/api/models?' + query, headers={'User-Agent': 'TrainPilot-Local/1.0'})
    with urlopen(request, timeout=12) as response:
        raw = response.read(500_001)
    if len(raw) > 500_000:
        raise ValueError('Model search results exceed the size limit.')
    items = json.loads(raw)
    if not isinstance(items, list):
        raise ValueError('Could not parse the Hugging Face model list.')
    results = []
    for item in items:
        if not isinstance(item, dict):
            continue
        model_id = item.get('id') or item.get('modelId')
        if not isinstance(model_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*', model_id):
            continue
        tags_found = item.get('tags') if isinstance(item.get('tags'), list) else []
        license_tag = next((tag[8:] for tag in tags_found if isinstance(tag, str) and tag.startswith('license:')), 'Not specified')
        results.append({'id': model_id, 'url': 'https://huggingface.co/' + model_id,
                        'task': str(item.get('pipeline_tag') or tags[task])[:100],
                        'library': str(item.get('library_name') or 'Not specified')[:100], 'license': license_tag[:100]})
    return results[:5]
