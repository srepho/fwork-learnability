"""Tests for code extraction."""

import pytest

from src.harness.extractor import CodeExtractor


@pytest.fixture
def extractor():
    return CodeExtractor()


class TestCodeExtractor:
    def test_extract_single_python_block(self, extractor):
        response = """Here's the solution:

```python
def hello():
    return "world"
```

That should work!
"""
        results = extractor.extract(response)
        assert len(results) == 1
        assert results[0].language == "python"
        assert "def hello():" in results[0].code
        assert results[0].is_valid_syntax

    def test_extract_multiple_blocks(self, extractor):
        response = """First, install:

```bash
pip install pydantic-ai
```

Then the code:

```python
from pydantic_ai import Agent

agent = Agent()
```
"""
        results = extractor.extract(response)
        assert len(results) == 2
        assert results[0].language == "bash"
        assert results[1].language == "python"

    def test_extract_python_primary(self, extractor):
        response = """```bash
pip install foo
```

```python
def short():
    pass
```

```python
def longer_function():
    x = 1
    y = 2
    return x + y
```
"""
        result = extractor.extract_python(response)
        assert result is not None
        # Should return the largest Python block
        assert "longer_function" in result.code

    def test_extract_invalid_syntax(self, extractor):
        response = """```python
def broken(
    missing parenthesis
```
"""
        results = extractor.extract(response)
        assert len(results) == 1
        assert not results[0].is_valid_syntax
        assert results[0].syntax_error is not None

    def test_extract_no_code(self, extractor):
        response = "Just some text without any code blocks."
        results = extractor.extract(response)
        assert len(results) == 0

    def test_extract_and_merge(self, extractor):
        response = """First part:

```python
import os
```

Second part:

```python
def main():
    print(os.getcwd())
```
"""
        result = extractor.extract_and_merge(response)
        assert result is not None
        assert "import os" in result.code
        assert "def main():" in result.code
        assert result.is_valid_syntax

    def test_extract_block_without_language(self, extractor):
        response = """```
some_function()
```
"""
        results = extractor.extract(response)
        assert len(results) == 1
        # Default to python
        assert results[0].language == "python"
