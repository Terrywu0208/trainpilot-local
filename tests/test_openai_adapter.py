from types import SimpleNamespace

from google.genai import types
from webapp.openai_adapter import OpenAIAdapter


class Output:
    def __init__(self, item): self.item = item
    def model_dump(self, **kwargs): return self.item


def test_responses_tool_call_and_output_are_replayed():
    class Responses:
        def __init__(self): self.requests = []
        def create(self, **kwargs):
            self.requests.append(kwargs)
            if len(self.requests) == 1:
                item = {'type': 'function_call', 'name': 'inspect_plan', 'call_id': 'call_1', 'arguments': '{"task":"classification"}'}
            else:
                item = {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': '完成'}]}
            return SimpleNamespace(status='completed', output=[Output(item)])
    responses = Responses()
    adapter = OpenAIAdapter(sdk=SimpleNamespace(responses=responses))
    config = types.GenerateContentConfig(system_instruction='Follow instructions', max_output_tokens=512,
        tools=[types.Tool(function_declarations=[types.FunctionDeclaration(name='inspect_plan', description='Inspect',
            parameters={'type':'OBJECT', 'properties':{'task':{'type':'STRING'}}, 'required':['task']})])])
    first = adapter.generate_content(model='', contents=[types.Content(role='user', parts=[types.Part.from_text(text='分析')])], config=config)
    content = first.candidates[0].content
    assert content.parts[0].function_call.args == {'task': 'classification'}
    tool_output = types.Content(role='user', parts=[types.Part.from_function_response(name='inspect_plan', response={'ready': True})])
    second = adapter.generate_content(model='', contents=[types.Content(role='user', parts=[types.Part.from_text(text='分析')]), content, tool_output], config=config)
    sent = responses.requests[1]
    assert sent['store'] is False and sent['parallel_tool_calls'] is False
    assert sent['tools'][0]['parameters']['type'] == 'object'
    assert sent['input'][-1]['type'] == 'function_call_output' and sent['input'][-1]['call_id'] == 'call_1'
    assert second.candidates[0].content.parts[0].text == '完成'


def test_choice_tool_requires_a_tool_action():
    class Responses:
        def create(self, **kwargs):
            assert kwargs['tool_choice'] == 'required'
            return SimpleNamespace(status='completed', output=[Output({'type': 'function_call', 'name': 'ask_clarification', 'call_id': 'c', 'arguments': '{"question":"請問目標？","options":["A","B"]}'})])
    adapter = OpenAIAdapter(sdk=SimpleNamespace(responses=Responses()))
    tool = types.FunctionDeclaration(name='ask_clarification', description='Ask', parameters={'type':'OBJECT','properties':{'question':{'type':'STRING'},'options':{'type':'ARRAY','items':{'type':'STRING'}}},'required':['question','options']})
    result = adapter.generate_content(model='', contents='請協助', config=types.GenerateContentConfig(tools=[types.Tool(function_declarations=[tool])]))
    assert result.candidates[0].content.parts[0].function_call.name == 'ask_clarification'
