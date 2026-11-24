"""Read-only Hugging Face MCP access for the platform's agent."""
import asyncio
import re

ENDPOINT = 'https://hf.co/mcp'
ALLOWED_TOOLS = {'hub_repo_search', 'hub_repo_details'}


async def _call(tool, arguments):
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    async with asyncio.timeout(25):
        async with streamable_http_client(ENDPOINT) as streams:
            async with ClientSession(streams[0], streams[1], read_timeout_seconds=15) as session:
                await session.initialize()
                available = {item.name for item in (await session.list_tools()).tools}
                if tool not in available:
                    raise ValueError('The Hugging Face MCP does not provide the required tool.')
                result = await session.call_tool(tool, arguments, read_timeout_seconds=15)
                if getattr(result, 'isError', False) or getattr(result, 'is_error', False):
                    raise ValueError('The Hugging Face MCP tool returned an error.')
                return '\n'.join(getattr(item, 'text', '') for item in result.content)[:10000]


def call(tool, arguments):
    if tool not in ALLOWED_TOOLS:
        raise ValueError('This MCP tool is not on the read-only allowlist.')
    return asyncio.run(_call(tool, arguments))


def search(query='', repo_type='model', task='', limit=5):
    if repo_type not in ('model', 'dataset', 'space') or not isinstance(query, str) or len(query) > 160:
        raise ValueError('Invalid Hugging Face search query.')
    if not isinstance(task, str) or len(task) > 80:
        raise ValueError('Invalid Hugging Face task tag.')
    task = re.sub(r'\s+', '-', task.strip().lower())
    if task and not re.fullmatch(r'[a-z0-9-]+', task):
        raise ValueError('Invalid Hugging Face task tag.')
    arguments = {'repo_types': [repo_type], 'limit': min(max(int(limit), 1), 5), 'sort': 'downloads'}
    if query:
        arguments['query'] = query
    if task:
        arguments['filters'] = [task]
    result = call('hub_repo_search', arguments)
    # A task description is often not a repository name. If an exact generic
    # task query produced no repos, search by the corresponding Hub tag once.
    generic = re.sub(r'[\s_-]+', ' ', query.strip().casefold())
    task_aliases = {'tabular classification': 'tabular-classification',
                    'tabular regression': 'tabular-regression'}
    if repo_type == 'model' and generic in task_aliases and not re.search(r'(?m)^###\s+\S+/\S+', result):
        fallback = {'repo_types': ['model'], 'filters': [task_aliases[generic]], 'limit': arguments['limit'], 'sort': 'downloads'}
        return 'The query was adjusted to a task tag.\n' + call('hub_repo_search', fallback)
    return result


def details(repo_id, repo_type='model'):
    if repo_type not in ('model', 'dataset', 'space') or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*', repo_id):
        raise ValueError('Invalid Hugging Face resource ID.')
    return call('hub_repo_details', {'repo_ids': [repo_id], 'repo_type': repo_type, 'operations': ['overview']})
