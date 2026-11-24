"""Responses API adapter for the existing LangGraph tool protocol."""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

from google.genai import types
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
MODEL = os.environ.get('TRAINPILOT_OPENAI_MODEL', 'gpt-5-nano')
_MARKER = b'trainpilot-openai-response:'


def _key():
    value = os.environ.get('OPENAI_API_KEY', '').strip()
    if value:
        return value
    path = ROOT / '.env.local'
    if path.is_file():
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            name, sep, value = line.partition('=')
            if sep and name.strip() == 'OPENAI_API_KEY':
                value = value.strip().strip('"').strip("'")
                if value:
                    return value
    raise ValueError('尚未設定 OpenAI API 金鑰。')


def _schema(value):
    if isinstance(value, dict):
        return {k: (v.lower() if k == 'type' and isinstance(v, str) else _schema(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [_schema(x) for x in value]
    return value


def _inputs(contents):
    if isinstance(contents, str):
        return [{'role': 'user', 'content': contents}]
    items, pending = [], []
    for content in contents:
        parts = content.parts or []
        raw = next((p.thought_signature for p in parts if p.thought_signature and p.thought_signature.startswith(_MARKER)), None)
        if raw:
            output = json.loads(raw[len(_MARKER):].decode('utf-8'))
            items.extend(output)
            pending.extend((x['name'], x['call_id']) for x in output if x.get('type') == 'function_call')
            continue
        texts = []
        for part in parts:
            if part.text:
                texts.append(part.text)
            if part.function_response:
                call = part.function_response
                idx = next((i for i, pair in enumerate(pending) if pair[0] == call.name), None)
                if idx is None:
                    raise ValueError('工具回覆沒有對應的模型呼叫。')
                _, call_id = pending.pop(idx)
                items.append({'type': 'function_call_output', 'call_id': call_id,
                              'output': json.dumps(call.response, ensure_ascii=False)})
        if texts:
            items.append({'role': 'assistant' if content.role == 'model' else 'user', 'content': '\n'.join(texts)})
    return items


class OpenAIAdapter:
    def __init__(self, api_key=None, sdk=None):
        self.sdk = sdk or OpenAI(api_key=api_key or _key(), timeout=30, max_retries=0)
        self.models = self

    def generate_content(self, *, model, contents, config):
        tools = []
        for group in (config.tools or []):
            for declaration in (group.function_declarations or []):
                schema = declaration.parameters.model_dump(mode='json', exclude_none=True)
                tools.append({'type': 'function', 'name': declaration.name,
                              'description': declaration.description or '', 'parameters': _schema(schema), 'strict': False})
        kwargs = {'model': MODEL, 'input': _inputs(contents), 'store': False,
                  'max_output_tokens': max(2048, int(config.max_output_tokens or 768)),
                  'reasoning': {'effort': 'minimal'}, 'include': ['reasoning.encrypted_content']}
        if config.system_instruction:
            kwargs['instructions'] = str(config.system_instruction)
        if tools:
            kwargs['tools'] = tools
            kwargs['parallel_tool_calls'] = False
            if any(tool['name'] == 'ask_clarification' for tool in tools):
                kwargs['tool_choice'] = 'required'
        reply = self.sdk.responses.create(**kwargs)
        if reply.status != 'completed':
            reason = getattr(getattr(reply, 'incomplete_details', None), 'reason', None)
            raise ValueError('OpenAI 回覆未完成' + (f'（{reason}）' if reason in ('max_output_tokens', 'content_filter') else '') + '，請稍後再試或縮小需求。')
        output = [x.model_dump(mode='json', exclude_none=True) for x in reply.output]
        parts = []
        for item in output:
            if item.get('type') == 'function_call':
                try:
                    args = json.loads(item['arguments'])
                except (ValueError, TypeError) as exc:
                    raise ValueError('模型工具參數不是有效 JSON。') from exc
                parts.append(types.Part(function_call=types.FunctionCall(name=item['name'], args=args)))
            elif item.get('type') == 'message':
                for block in item.get('content', []):
                    if block.get('type') == 'output_text' and block.get('text'):
                        parts.append(types.Part(text=block['text']))
        if not parts:
            raise ValueError('OpenAI 沒有回傳文字或工具呼叫。')
        parts[0].thought_signature = _MARKER + json.dumps(output, ensure_ascii=False).encode('utf-8')
        return SimpleNamespace(candidates=[SimpleNamespace(content=types.Content(role='model', parts=parts))])
