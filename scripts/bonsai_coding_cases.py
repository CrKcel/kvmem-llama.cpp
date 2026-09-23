"""Two compact Python coding tasks embedded in synthetic repository context."""

TASKS = [
    '''Implement merge_intervals(intervals). Input is a list of (start, end) integer
tuples with start <= end. Return a new list of tuples sorted by start, merging
overlapping or touching CLOSED intervals. Empty input returns []. Do not mutate
the input. Handle duplicates, nested intervals, negative coordinates and point
intervals. Target O(n log n) time. Example: [(5,7),(1,3),(3,4)] -> [(1,4),(5,7)].''',
    '''Implement topological_sort(nodes, edges). Nodes is a list of unique strings;
edges is a list of (source, destination) tuples whose endpoints belong to nodes.
Return a topological ordering of ALL nodes, including isolated nodes. At each
step choose the lexicographically smallest currently available zero-indegree
node. Duplicate edges count once. Raise ValueError for any cycle, including
self-loops. Empty input returns []. Do not mutate either input. Example:
nodes=['c','b','a'], edges=[('a','c')] -> ['a','b','c'].''',
]


def make_prompt(label, repeat):
    # The leading marker differs between tasks/lengths to prevent prefix reuse.
    # Both cases and all filler are identical across speculative modes.
    count = 80 if label == '4k' else 350
    source = '\n\n'.join(
        f'def normalize_field_{i:04d}(value: int) -> int:\n'
        f'    """Normalize registry field {i:04d} to its storage range."""\n'
        f'    return (value + {i % 97}) % 97'
        for i in range(count))
    return (f'Coding benchmark {label} task {repeat}. Repository context follows.\n'
            'These existing helpers are unrelated to the requested new utility; do not reproduce them.\n'
            f'```python\n{source}\n```\n'
            'Add one production utility in a new module.\n' + TASKS[repeat % len(TASKS)] +
            '\nReturn ONLY one complete Python function, with a short docstring and useful type hints. '
            'No imports, markdown fences, examples, tests, or explanation. Use Python builtins only. '
            'Keep the full solution within 450 tokens; prefer clear concise code.')
