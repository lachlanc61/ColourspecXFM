import pytest

@pytest.fixture
def foo():
    return "foo"

@pytest.fixture
def bar(foo):
    return foo, "bar"

def test_foo_bar(bar):
    expected = ("foo", "bar")
    assert bar == expected