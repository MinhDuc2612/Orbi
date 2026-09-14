"""Validate explicit source copies before execution; never repair emitted values."""

import difflib
import json
import re
import subprocess
import sys


def verbatim_sources(prompt, function):
    # ponytail: explicit delimiters only; require structured source spans for other prose.
    if function == "remember":
        marker = re.search(r"\bexact (?:fact|text)\b[^:\n]*:[ \t]?", prompt, re.I)
        if marker is None:
            marker = re.search(r"\bPreserve the newline:[ \t]*\r?\n", prompt, re.I)
        return {"text": prompt[marker.end():]} if marker else {}
    patterns = {
        "read_file": ("path", r"\bexact path is (.+?)\. (?:Use|Read)\b"),
        "search_files": ("query", r"\bregex to search [^:\n]+:[ \t]*(.+?)\. (?:Search|Match)\b"),
        "recall": ("query", r'\bexact query "([^"]+)"|\bexact query (.+?)(?: with the default result limit|, returning\b|\.$)'),
    }
    if function not in patterns:
        return {}
    field, pattern = patterns[function]
    match = re.search(pattern, prompt)
    return {field: next(value for value in match.groups() if value is not None)} if match else {}


def copy_issues(prompt, function, arguments):
    issues = []
    for field, source in verbatim_sources(prompt, function).items():
        actual = arguments.get(field)
        if actual != source:
            before = json.dumps(source, ensure_ascii=False)
            after = json.dumps(actual, ensure_ascii=False)
            issues.append(dict(kind="verbatim", field=field, source=source, emitted=actual,
                               diff="\n".join(difflib.ndiff([before], [after]))))
    return issues


def retry_feedback(issues):
    diagnostics = [issue["diagnostic"] for issue in issues if issue.get("diagnostic")]
    return ("The tool call was rejected and was not executed. Retry exactly once with one corrected "
            "tool call. Preserve source characters, punctuation and whitespace exactly. "
            "In a verbatim diff, '-' is the required source and '+' is your emitted argument.\n"
            + ("\n".join(diagnostics) + "\nRe-emit the pattern that matches the intended targets.\n"
               if diagnostics else "")
            + json.dumps(issues, ensure_ascii=False))


def tool_issues(prompt, function, arguments, regex_examples=None):
    issues = copy_issues(prompt, function, arguments)
    if regex_examples is not None and function == "search_files" and arguments.get("literal") is False:
        issues += regex_issues(arguments.get("query"), regex_examples,
                               case_sensitive=arguments.get("case_sensitive", True))
    return issues


def regex_issues(pattern, examples, *, case_sensitive=True):
    """Compile and check caller-supplied (text, should_match) examples in a bounded child."""
    if (type(pattern) is not str or type(case_sensitive) is not bool
            or not isinstance(examples, (list, tuple)) or not examples
            or any(not isinstance(item, (list, tuple)) or len(item) != 2
                   or type(item[0]) is not str or type(item[1]) is not bool for item in examples)):
        return [dict(kind="regex", error="A pattern and explicit boolean match examples are required")]
    payload = json.dumps(dict(pattern=pattern, examples=examples, case_sensitive=case_sensitive))
    if len(payload) > 65536:
        return [dict(kind="regex", error="Regex validation input exceeds 65536 characters")]
    program = """import json,re,sys
from re import _parser,_constants
data=json.load(sys.stdin)
try:
    pattern=re.compile(data['pattern'],0 if data['case_sensitive'] else re.IGNORECASE)
    # ponytail: pinned Python3.12 parser; explain terminal wildcards only, retain observations otherwise.
    parsed=_parser.parse(data['pattern'],pattern.flags)
    terminal_wildcard=bool(parsed) and parsed[-1][0]==_constants.ANY
    errors=[]
    for text,wanted in data['examples']:
        actual=bool(pattern.search(text))
        if actual != wanted:
            diagnostic='Your pattern '+data['pattern']
            diagnostic+=(' does not match required target ' if wanted else ' incorrectly matches excluded target ')+json.dumps(text)+'.'
            witness=pattern.search(text+'x') if wanted and terminal_wildcard else None
            if witness is not None and witness.end()==len(text)+1:
                diagnostic+=' The trailing "." requires one more character'
                diagnostic+=(' after '+json.dumps(text[-1]) if text else '')+'. This target ends there.'
                diagnostic+=' The unchanged pattern matches '+json.dumps(text+'x')+' instead.'
            errors.append(dict(kind='regex',pattern=data['pattern'],text=text,expected=wanted,
                               actual=actual,diagnostic=diagnostic))
except re.error as error:
    errors=[dict(kind='regex',error=str(error))]
print(json.dumps(errors))
"""
    try:
        result = subprocess.run([sys.executable, "-I", "-c", program], input=payload,
                                text=True, capture_output=True, timeout=1, check=True)
        return json.loads(result.stdout)
    except (subprocess.SubprocessError, OSError, ValueError) as error:
        return [dict(kind="regex", error=f"Regex validation failed: {type(error).__name__}")]
