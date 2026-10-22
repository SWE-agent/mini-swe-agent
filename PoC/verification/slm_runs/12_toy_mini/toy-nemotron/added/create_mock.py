"""Create a minimal mock anthropic module."""
import sys
import types

# Create a minimal anthropic module that only has what we need
anthropic_mock = types.ModuleType('anthropic')
anthropic_mock.Anthropic = type('Anthropic', (), {
    '__init__': lambda self, **kwargs: setattr(self, '_call_count', 0),
    'messages': property(lambda self: self._Messages()),
})

class _Messages:
    def create(self, model, max_tokens, system, messages, **kwargs):
        anthropic_mock.Anthropic._call_count += 1
        if anthropic_mock.Anthropic._call_count == 1:
            return types.SimpleNamespace(content=[
                {"type": "tool_use", "id": "tool_1", "name": "bash", "input": {"command": "echo hello"}}
            ])
        else:
            return types.SimpleNamespace(content=[
                {"type": "text", "text": "Task complete: echo hello ran successfully"}
            ])

anthropic_mock.AsyncAnthropic = anthropic_mock.Anthropic
anthropic_mock.Client = anthropic_mock.Anthropic
anthropic_mock.AsyncClient = anthropic_mock.Anthropic
anthropic_mock.Timeout = Exception
anthropic_mock.RequestOptions = type('RequestOptions', (), {})

sys.modules['anthropic'] = anthropic_mock
sys.modules['anthropic._client'] = types.ModuleType('anthropic._client')
sys.modules['anthropic._client'].Anthropic = anthropic_mock.Anthropic
sys.modules['anthropic._client'].AsyncAnthropic = anthropic_mock.AsyncAnthropic
sys.modules['anthropic._base_client'] = types.ModuleType('anthropic._base_client')
sys.modules['anthropic._base_client'].Anthropic = anthropic_mock.Anthropic

print("Mock anthropic module created successfully")
