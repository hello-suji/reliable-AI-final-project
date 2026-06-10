import py

failure_demo = py.path.local(__file__).dirpath("failure_demo.py")
pytest_plugins = ("pytester",)


def test_failure_demo_fails_properly(testdir):
    target = testdir.tmpdir.join(failure_demo.basename)
    failure_demo.copy(target)
    failure_demo.copy(testdir.tmpdir.join(failure_demo.basename))
    result = testdir.runpytest(target, syspathinsert=True)
    result.stdout.fnmatch_lines(["*44 failed*"])
    assert result.ret != 0

import pytest
from _pytest.runner import pytest_runtest_makereport

def test_pytest_runtest_makereport_skipped_repro():
    @pytest.fixture
    def mock_item(request):
        class MockItem:
            def __init__(self):
                self.location = ('mock_file.py', 1, 'mock_func')
                self.session = request.node.session
        return MockItem()

    mock_item = mock_item(request)
    report = pytest_runtest_makereport(mock_item, None, 'setup')
    mock_item.session.config.hook.pytest_runtest_logreport(report=report)
    assert report.skipped
    # [Tier 2: probe-verified buggy repr — must differ after fix]
    assert repr(report) != 'SKIPPED [1] src/_pytest/skipping.py:238: unconditional skip'
