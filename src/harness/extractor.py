"""Extract Python code from LLM responses."""

import ast
import re
from dataclasses import dataclass


@dataclass
class ExtractedCode:
    """Code extracted from an LLM response."""

    code: str
    language: str
    is_valid_syntax: bool
    syntax_error: str | None = None


class CodeExtractor:
    """Extract Python code blocks from LLM markdown responses."""

    # Pattern to match fenced code blocks with optional language
    CODE_BLOCK_PATTERN = re.compile(
        r"```(?P<lang>\w+)?\s*\n(?P<code>.*?)```",
        re.DOTALL,
    )

    def extract(self, response: str) -> list[ExtractedCode]:
        """Extract all code blocks from an LLM response.

        Args:
            response: The full LLM response text.

        Returns:
            List of ExtractedCode objects for each code block found.
        """
        results = []
        for match in self.CODE_BLOCK_PATTERN.finditer(response):
            lang = match.group("lang") or "python"
            code = match.group("code").strip()

            # Validate Python syntax
            is_valid, error = self._validate_python(code) if lang == "python" else (True, None)

            results.append(
                ExtractedCode(
                    code=code,
                    language=lang,
                    is_valid_syntax=is_valid,
                    syntax_error=error,
                )
            )

        return results

    def extract_python(self, response: str) -> ExtractedCode | None:
        """Extract the primary Python code block from a response.

        If multiple Python blocks exist, returns the largest one.
        Falls back to the first code block if no Python blocks found.

        Args:
            response: The full LLM response text.

        Returns:
            The primary ExtractedCode or None if no code found.
        """
        all_blocks = self.extract(response)
        if not all_blocks:
            return None

        # Filter to Python blocks
        python_blocks = [b for b in all_blocks if b.language == "python"]

        if python_blocks:
            # Return the largest Python block
            return max(python_blocks, key=lambda b: len(b.code))

        # Fall back to first block if no Python
        return all_blocks[0]

    def extract_and_merge(self, response: str) -> ExtractedCode | None:
        """Extract and merge all Python code blocks into a single module.

        Useful when the LLM splits code across multiple blocks.

        Args:
            response: The full LLM response text.

        Returns:
            Merged ExtractedCode or None if no code found.
        """
        all_blocks = self.extract(response)
        python_blocks = [b for b in all_blocks if b.language == "python"]

        if not python_blocks:
            return None

        # Merge all Python code
        merged_code = "\n\n".join(b.code for b in python_blocks)

        # Validate merged syntax
        is_valid, error = self._validate_python(merged_code)

        return ExtractedCode(
            code=merged_code,
            language="python",
            is_valid_syntax=is_valid,
            syntax_error=error,
        )

    def _validate_python(self, code: str) -> tuple[bool, str | None]:
        """Validate Python syntax.

        Args:
            code: Python code to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Line {e.lineno}: {e.msg}"
