"""Check benchmark code in time-limited subprocesses with restricted builtins."""
import ast
import copy
import itertools
import json
from pathlib import Path
import random
import re
import subprocess
import sys


def check(payload):
    source = payload['source'].strip()
    fence = re.fullmatch(r'```(?:python)?\s*\n(.*?)\n```', source, re.S)
    if fence:
        source = fence[1]
    tree = ast.parse(source)
    expected = 'merge_intervals' if payload['task'] == 0 else 'topological_sort'
    assert len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef)
    assert tree.body[0].name == expected and not tree.body[0].decorator_list
    assert not tree.body[0].args.defaults and not any(tree.body[0].args.kw_defaults)
    forbidden = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.ClassDef)
    methods = {'append', 'pop', 'add', 'discard', 'items', 'values', 'keys', 'get',
               'copy', 'update', 'setdefault', 'remove', 'sort', 'extend'}
    for node in ast.walk(tree):
        assert not isinstance(node, forbidden), 'unsupported syntax'
        if isinstance(node, ast.Name):
            assert not node.id.startswith('__'), 'dunder access'
        if isinstance(node, ast.Attribute):
            assert node.attr in methods, 'unsupported attribute'
    allowed = {name: value for name, value in vars(__import__('builtins')).items()
               if name in {'list', 'tuple', 'dict', 'set', 'str', 'int', 'float',
                           'bool', 'len', 'range', 'sorted', 'enumerate', 'zip',
                           'min', 'max', 'sum', 'any', 'all', 'ValueError'}}
    ns = {'__builtins__': allowed}
    exec(compile(tree, '<candidate>', 'exec'), ns)
    function = ns[expected]
    passed, failures = 0, []
    rng = random.Random(7429)
    if payload['task'] == 0:
        cases = [[], [(1, 2), (3, 4)], [(1, 3), (3, 4)], [(1, 8), (2, 3)],
                 [(2, 2), (2, 2)], [(-5, -2), (-1, 1)],
                 [(5, 7), (1, 3), (3, 4)]]
        for _ in range(100):
            cases.append([tuple(sorted((rng.randrange(-6, 7), rng.randrange(-6, 7))))
                          for _ in range(rng.randrange(10))])
        for case in cases:
            original = copy.deepcopy(case)
            try:
                result = function(case)
                assert case == original, 'mutated input'
                assert isinstance(result, list) and result is not case, 'expected new list'
                assert all(isinstance(x, tuple) and len(x) == 2 and x[0] <= x[1]
                           for x in result), 'expected valid interval tuples'
                assert all(a[1] < b[0] for a, b in zip(result, result[1:])), 'not merged/sorted'
                # Half-integer sample points distinguish adjacent integer intervals
                # from overlapping/touching continuous CLOSED intervals.
                expected_cover = {x for x in range(-14, 19) if any(2*a <= x <= 2*b for a,b in original)}
                actual_cover = {x for x in range(-14, 19) if any(2*a <= x <= 2*b for a,b in result)}
                assert expected_cover == actual_cover, 'changed covered set (filled a gap or lost coverage)'
                passed += 1
            except Exception as exc:
                failures.append({'input':original, 'error':str(exc)})
    else:
        cases = [([], []), (['a'], [('a', 'a')]),
                 (['c','b','a'], [('a','c'), ('a','c')]),
                 (['a','b'], [('a','b'), ('b','a')])]
        for _ in range(100):
            nodes = list('abcdef'[:rng.randrange(1, 7)])
            rng.shuffle(nodes)
            edges = [(a,b) for a in nodes for b in nodes if rng.random() < .16]
            if edges:
                edges += edges[:2]
            cases.append((nodes, edges))
        for nodes, edges in cases:
            original = copy.deepcopy((nodes, edges))
            try:
                # First valid lexicographic permutation is an independent oracle.
                expected_order = next((list(p) for p in itertools.permutations(sorted(nodes))
                    if all(p.index(a) < p.index(b) for a,b in edges)), None)
                try:
                    result = function(nodes, edges)
                except ValueError:
                    assert expected_order is None, 'raised on an acyclic graph'
                else:
                    assert expected_order is not None, 'did not reject cycle'
                    assert result == expected_order, f'expected {expected_order}, got {result}'
                assert (nodes, edges) == original, 'mutated input'
                passed += 1
            except Exception as exc:
                failures.append({'input':original, 'error':str(exc)})
    return {'passed':passed, 'total':passed+len(failures), 'all_passed':not failures,
            'failures':failures[:5], 'stripped_markdown_fence':bool(fence)}


if len(sys.argv) > 1 and sys.argv[1] == '--worker':
    try:
        print(json.dumps(check(json.load(sys.stdin))))
    except Exception as exc:
        print(json.dumps({'all_passed':False, 'error':repr(exc)}))
else:
    directory = Path(sys.argv[1])
    data = json.loads((directory/'results.json').read_text())
    results = []
    for run in data['runs']:
        for row in run['measurements']:
            content = row['response']['choices'][0]['message']['content']
            payload = {'source':content, 'task':row['repeat'] % 2}
            try:
                child = subprocess.run([sys.executable, '-I', '-S', __file__, '--worker'],
                    input=json.dumps(payload), capture_output=True, text=True, timeout=5,
                    creationflags=subprocess.CREATE_NO_WINDOW)
                verdict = json.loads(child.stdout)
            except Exception as exc:
                verdict = {'all_passed':False, 'error':repr(exc)}
            result = {'draft':run['draft'], 'context':row['context_label'],
                      'task':payload['task'], 'output_tokens':row['response']['usage']['completion_tokens'],
                      'finish_reason':row['response']['choices'][0].get('finish_reason'),
                      'verdict':verdict}
            results.append(result)
    (directory/'code-checks.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))
