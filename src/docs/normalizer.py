"""Normalize documentation to clean Markdown format."""

import re
from dataclasses import dataclass


@dataclass
class NormalizedDocument:
    """A normalized documentation document."""

    title: str
    content: str
    original_source: str
    token_count: int = 0


class DocumentNormalizer:
    """Normalize documentation from various sources to clean Markdown.

    Ensures fair comparison by:
    - Standardizing header levels
    - Normalizing code blocks
    - Removing navigation/sidebar artifacts
    - Preserving code examples exactly
    """

    def normalize(self, content: str, source: str = "") -> NormalizedDocument:
        """Normalize a document to clean Markdown.

        Args:
            content: Raw document content (HTML or Markdown).
            source: Original source URL or path.

        Returns:
            NormalizedDocument with cleaned content.
        """
        # Detect if HTML and convert
        if self._is_html(content):
            content = self._html_to_markdown(content)

        # Apply normalization steps
        content = self._normalize_headers(content)
        content = self._normalize_code_blocks(content)
        content = self._remove_navigation(content)
        content = self._normalize_lists(content)
        content = self._normalize_whitespace(content)

        # Extract title
        title = self._extract_title(content)

        return NormalizedDocument(
            title=title,
            content=content,
            original_source=source,
        )

    def _is_html(self, content: str) -> bool:
        """Check if content appears to be HTML."""
        html_patterns = [
            r"<!DOCTYPE\s+html",
            r"<html",
            r"<head>",
            r"<body>",
            r"<div\s",
        ]
        for pattern in html_patterns:
            if re.search(pattern, content, re.I):
                return True
        return False

    def _html_to_markdown(self, html: str) -> str:
        """Convert HTML to Markdown."""
        try:
            from bs4 import BeautifulSoup
            from markdownify import markdownify

            # Parse HTML
            soup = BeautifulSoup(html, "html.parser")

            # Remove script, style, nav elements
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()

            # Find main content area (common patterns)
            main_content = (
                soup.find("main")
                or soup.find("article")
                or soup.find(class_=re.compile(r"content|main|doc"))
                or soup.find("body")
            )

            if main_content:
                html = str(main_content)
            else:
                html = str(soup)

            # Convert to markdown
            md = markdownify(html, heading_style="ATX", code_language="python")
            return md

        except ImportError:
            # Fallback: basic HTML stripping
            return self._basic_html_strip(html)

    def _basic_html_strip(self, html: str) -> str:
        """Basic HTML to text conversion without external libs."""
        # Remove tags but keep content
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.I)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.I)
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"&nbsp;", " ", text)
        text = re.sub(r"&lt;", "<", text)
        text = re.sub(r"&gt;", ">", text)
        text = re.sub(r"&amp;", "&", text)
        return text

    def _normalize_headers(self, content: str) -> str:
        """Normalize header formatting."""
        # Ensure space after # in headers
        content = re.sub(r"^(#{1,6})([^\s#])", r"\1 \2", content, flags=re.MULTILINE)

        # Remove excessive header levels (keep max depth 3)
        def reduce_header(match):
            hashes = match.group(1)
            text = match.group(2)
            level = min(len(hashes), 3)
            return "#" * level + " " + text

        content = re.sub(r"^(#{4,})\s*(.+)$", reduce_header, content, flags=re.MULTILINE)

        return content

    def _normalize_code_blocks(self, content: str) -> str:
        """Normalize code block formatting."""
        # Ensure code blocks have language specifier
        def add_python_lang(match):
            if match.group(1):  # Already has language
                return match.group(0)
            return "```python\n" + match.group(2) + "```"

        content = re.sub(
            r"```(\w+)?\n(.*?)```",
            add_python_lang,
            content,
            flags=re.DOTALL,
        )

        return content

    def _remove_navigation(self, content: str) -> str:
        """Remove navigation elements and artifacts."""
        # Remove common navigation patterns
        nav_patterns = [
            r"^\s*\[.*?\]\(#.*?\)\s*$",  # Anchor links
            r"^\s*←.*?→\s*$",  # Navigation arrows
            r"^\s*Previous\s*\|\s*Next\s*$",  # Prev/Next
            r"^\s*Table of Contents\s*$",
            r"^\s*On this page:?\s*$",
        ]

        for pattern in nav_patterns:
            content = re.sub(pattern, "", content, flags=re.MULTILINE | re.I)

        return content

    def _normalize_lists(self, content: str) -> str:
        """Normalize list formatting."""
        # Ensure consistent bullet style
        content = re.sub(r"^\s*[*+-]\s+", "- ", content, flags=re.MULTILINE)

        # Ensure numbered lists use 1. format
        content = re.sub(r"^\s*(\d+)[.)]\s+", r"\1. ", content, flags=re.MULTILINE)

        return content

    def _normalize_whitespace(self, content: str) -> str:
        """Normalize whitespace."""
        # Remove trailing whitespace
        content = re.sub(r"[ \t]+$", "", content, flags=re.MULTILINE)

        # Collapse multiple blank lines to max 2
        content = re.sub(r"\n{4,}", "\n\n\n", content)

        # Ensure single newline at end
        content = content.strip() + "\n"

        return content

    def _extract_title(self, content: str) -> str:
        """Extract document title from first header."""
        match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return "Untitled"
