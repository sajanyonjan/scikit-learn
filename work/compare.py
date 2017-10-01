import re
from pprint import pprint
from collections import Counter

pytest_content = open('pytest.log').read()
pytest_pattern = r'^(sklearn[\w/.]+)::([\w.]+)'

pytest_matches = re.findall(pytest_pattern, pytest_content, flags=re.MULTILINE)


def normalise_pytest(item):
    filename, test_func_name = item
    if '/test_' in filename:
        module = filename.replace('/', '.').replace('.py', '')
        return f'{module}.{test_func_name}'
    else:
        return test_func_name


pytest_matches = [normalise_pytest(item) for item in pytest_matches]

print(f'Got {len(pytest_matches)} pytest matches')

nose_content = open('nosetests.log').read()

nose_patterns = [
    # test_common starts has the full path for some reason
    r'^/home/lesteve/dev/(?:alt-)?scikit-learn/(sklearn[\w./]+test_[\w./]+)',
    # doctest
    r'^(?:Doctest: )?(sklearn[\w.]+)',
    # class tests
    r'^test_\w+ \(([\w.]+)']

nose_matches = sum((re.findall(pattern, nose_content, flags=re.MULTILINE)
                    for pattern in nose_patterns), [])

# Those are due to warnings where by luck /home/lesteve happens to be the start
# of the line
nose_matches = [each for each in nose_matches if not each.endswith('.py')]


def normalise_nose(item):
    # This is needed for some test_common (maybe because of the
    # function that gives them a better name?)
    if item.startswith('sklearn/'):
        item = item.replace('/', '.').replace('.py', '')
    if 'TestMetrics' in item:
        # Because TestMetrics does not derive from unittest.TestCase
        # the nose format is different ...
        item = item.rpartition('.')[0]
    return item


nose_matches = [normalise_nose(item) for item in nose_matches]

print(f'Got {len(nose_matches)} nose matches')
print('-'*80)
print('pytest_matches - nose_matches')
print('set difference:')
pprint(set(pytest_matches) - set(nose_matches))
print('counter differences:')
pprint(Counter(pytest_matches) - Counter(nose_matches))
print('-'*80)
print('nose_matches - pytest_matches')
print('set difference:')
pprint(set(nose_matches) - set(pytest_matches))
print('counter differences:')
counter_diff = Counter(nose_matches) - Counter(pytest_matches)
pprint(counter_diff)
counter_diff
nose_counter = Counter(nose_matches)
print('nose counter values with number differences')
pprint({k: nose_counter[k] for k in counter_diff})
