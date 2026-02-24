import pytest
from automet.base import BaseAnalyzer


def test_base_analyzer_is_abstract():
    """BaseAnalyzer cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseAnalyzer()


def test_concrete_subclass_must_implement_all_methods():
    """A subclass missing any abstract method cannot be instantiated."""

    class IncompleteAnalyzer(BaseAnalyzer):
        def load_data(self):
            return self

        def run(self):
            return self
        # save_output intentionally omitted

    with pytest.raises(TypeError):
        IncompleteAnalyzer()


def test_concrete_subclass_instantiates():
    """A fully implemented subclass instantiates without error."""

    class MinimalAnalyzer(BaseAnalyzer):
        def load_data(self):
            return self

        def run(self):
            return self

        def save_output(self, filename, dpi=150):
            return self

    obj = MinimalAnalyzer()
    assert isinstance(obj, BaseAnalyzer)


def test_method_chaining():
    """Abstract method contract requires return self for chaining."""

    class ChainableAnalyzer(BaseAnalyzer):
        def load_data(self):
            return self

        def run(self):
            return self

        def save_output(self, filename, dpi=150):
            return self

    obj = ChainableAnalyzer()
    assert obj.load_data() is obj
    assert obj.run() is obj
    assert obj.save_output("out.png") is obj
