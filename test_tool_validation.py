"""Source-copy and bounded regex checks; synthetic cases, not model scores."""

from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch
import io
import json
import subprocess

from tool_validation import copy_issues, regex_issues, retry_feedback, verbatim_sources


def cli_checks():
    import orbi

    source = 'Keep "this" punctuation 🌐.'
    prompt = 'Remember this exact fact globally: ' + source
    def call(text):
        return dict(role='assistant', content='', tool_calls=[dict(id='call-1', type='function',
            function=dict(name='remember', arguments=json.dumps(dict(text=text, scope='global', tier='L1'))))])
    for replies, succeeds in (([call(source[:-1]), call(source), dict(content='Saved.')], True),
                               ([call(source[:-1]), call(source[:-1])], False),
                               ([call(source[:-1]), dict(content='Saved.')], False)):
        memory = Mock()
        memory.retrieve.return_value = dict(text='')
        memory.add.return_value = 1
        task = Mock()
        with patch.object(orbi, 'Task', return_value=task), patch.object(orbi, 'history', return_value=[]), \
                patch.object(orbi, 'ensure_runtime'), patch.object(orbi, 'fit_messages', return_value=[]), \
                patch.object(orbi, 'stream_reply', side_effect=replies), redirect_stdout(io.StringIO()):
            try:
                orbi._run_turn(dict(paths=dict(db_path=Path('unused.db'))), memory, 'session', str(Path.cwd()), prompt)
            except ValueError as error:
                assert not succeeds and 'Verbatim' in str(error)
            else:
                assert succeeds
        if succeeds:
            assert memory.add.call_args_list[0].args[0] == source and memory.add.call_count == 2
        else:
            memory.add.assert_not_called()
        feedback = [c.args[0] for c in task.message.call_args_list if c.args[0].get('role') == 'tool']
        assert len(feedback) == (2 if succeeds else 1)
        assert sum('not executed' in item['content'] for item in feedback) == 1


def main():
    import orbi

    messages = [dict(role='system', content='Original instructions.'), dict(role='user', content='Keep me.')]
    effective = orbi.system_messages(messages)
    assert effective[0]['content'] == messages[0]['content'] + '\n\n' + orbi.SYSTEM_RULES
    assert messages[0]['content'] == 'Original instructions.' and effective[1] == messages[1]
    assert orbi.system_messages(effective) == effective
    assert orbi.system_messages(messages[1:])[0] == dict(role='system', content=orbi.SYSTEM_RULES)
    with patch.object(orbi, 'json_request', side_effect=[dict(prompt='rendered'), dict(tokens=[1, 2])]) as request:
        fitted = orbi.fit_messages(orbi.settings(), 'Original instructions.', 'Memory.', [], messages[1:])
    counted = request.call_args_list[0].args[1]['messages']
    assert counted == fitted and counted[0]['content'].endswith(orbi.SYSTEM_RULES)

    source = 'Exact "quotes", a backslash \\, tiếng Việt 🌐.\nSecond line. '
    prompt = 'Remember this exact fact globally: ' + source
    assert verbatim_sources(prompt, 'remember') == {'text': source}
    assert not copy_issues(prompt, 'remember', {'text': source})
    issues = copy_issues(prompt, 'remember', {'text': source.rstrip()})
    assert len(issues) == 1 and issues[0]['source'] == source
    assert 'not executed' in retry_feedback(issues) and 'diff' in retry_feedback(issues)
    assert not copy_issues('Remember that I enjoy tea.', 'remember', {'text': 'I enjoy tea.'})
    assert verbatim_sources('Save this. Preserve the newline:\n' + source, 'remember')['text'] == source
    regex_prompt = r'Use this regex to search lib/*.py: ^def [a-z_]+\(. Search case-sensitively.'
    pattern = verbatim_sources(regex_prompt, 'search_files')['query']
    assert pattern == r'^def [a-z_]+\('
    assert copy_issues(regex_prompt, 'search_files', {'query': pattern + '.'})
    examples = [('def hello(', True), ('class Hello(', False), ('def 9bad(', False)]
    assert not regex_issues(pattern, examples)
    assert regex_issues(pattern + '.', examples)
    assert regex_issues('(', examples)
    assert regex_issues(pattern, [])
    assert regex_issues(pattern, [('def hello(', 1)])
    assert regex_issues(pattern, ['malformed'])
    assert not regex_issues('abc', [('ABC', True)], case_sensitive=False)
    with patch('tool_validation.subprocess.run', side_effect=subprocess.TimeoutExpired('regex', 1)):
        assert regex_issues(pattern, examples)[0]['error'].endswith('TimeoutExpired')
    cli_checks()
    print('PASS: verbatim source preservation, diff feedback, regex compile/match and timeout checks.')


if __name__ == '__main__':
    main()
