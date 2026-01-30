"""Documentation corpus storage and versioning."""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

from .normalizer import DocumentNormalizer, NormalizedDocument


class DocumentLevel(Enum):
    """Documentation levels for experiments."""

    NONE = "none"  # Framework name only
    MINIMAL = "minimal"  # One paragraph + install
    MODERATE = "moderate"  # Quickstart + core concepts
    FULL = "full"  # Above + tool use + API reference


@dataclass
class DocumentSnapshot:
    """A versioned snapshot of framework documentation."""

    framework: str
    version: str
    level: DocumentLevel
    content: str
    content_hash: str
    token_count: int
    snapshot_date: str
    source_urls: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "framework": self.framework,
            "version": self.version,
            "level": self.level.value,
            "content_hash": self.content_hash,
            "token_count": self.token_count,
            "snapshot_date": self.snapshot_date,
            "source_urls": self.source_urls,
        }


class DocumentCorpus:
    """Manage documentation corpus for experiments.

    Handles:
    - Storage of normalized documentation
    - Versioning via content hashes
    - Retrieval at different documentation levels
    """

    def __init__(self, corpus_dir: Path):
        """Initialize corpus.

        Args:
            corpus_dir: Directory to store documentation files.
        """
        self.corpus_dir = Path(corpus_dir)
        self.corpus_dir.mkdir(parents=True, exist_ok=True)
        self.normalizer = DocumentNormalizer()
        self._index: dict[str, dict] = {}
        self._load_index()

    def _load_index(self):
        """Load corpus index from disk."""
        index_file = self.corpus_dir / "index.json"
        if index_file.exists():
            self._index = json.loads(index_file.read_text())

    def _save_index(self):
        """Save corpus index to disk."""
        index_file = self.corpus_dir / "index.json"
        index_file.write_text(json.dumps(self._index, indent=2))

    def add_document(
        self,
        framework: str,
        version: str,
        level: DocumentLevel,
        content: str,
        source_urls: list[str] | None = None,
    ) -> DocumentSnapshot:
        """Add a document to the corpus.

        Args:
            framework: Framework name.
            version: Framework version.
            level: Documentation level.
            content: Raw document content.
            source_urls: Source URLs for the content.

        Returns:
            DocumentSnapshot with metadata.
        """
        # Normalize content
        normalized = self.normalizer.normalize(content, source_urls[0] if source_urls else "")

        # Compute hash
        content_hash = hashlib.sha256(normalized.content.encode()).hexdigest()[:16]

        # Count tokens (approximate)
        token_count = len(normalized.content.split()) * 1.3  # Rough estimate

        # Create snapshot
        snapshot = DocumentSnapshot(
            framework=framework,
            version=version,
            level=level,
            content=normalized.content,
            content_hash=content_hash,
            token_count=int(token_count),
            snapshot_date=datetime.utcnow().isoformat(),
            source_urls=source_urls or [],
        )

        # Store content
        doc_file = self._get_doc_path(framework, version, level)
        doc_file.parent.mkdir(parents=True, exist_ok=True)
        doc_file.write_text(normalized.content)

        # Update index
        key = f"{framework}/{version}/{level.value}"
        self._index[key] = snapshot.to_dict()
        self._save_index()

        return snapshot

    def get_document(
        self,
        framework: str,
        version: str,
        level: DocumentLevel,
    ) -> str | None:
        """Get documentation at specified level.

        Args:
            framework: Framework name.
            version: Framework version.
            level: Documentation level.

        Returns:
            Document content or None if not found.
        """
        doc_file = self._get_doc_path(framework, version, level)
        if doc_file.exists():
            return doc_file.read_text()
        return None

    def get_documentation(
        self,
        framework: str,
        version: str,
        level: DocumentLevel,
    ) -> str:
        """Get documentation content for a trial.

        Builds appropriate documentation based on level:
        - NONE: Just framework name
        - MINIMAL: Name + one paragraph + install
        - MODERATE: Quickstart content
        - FULL: All available documentation

        Args:
            framework: Framework name.
            version: Framework version.
            level: Documentation level.

        Returns:
            Documentation string for the LLM prompt.
        """
        if level == DocumentLevel.NONE:
            return f"Framework: {framework}"

        content = self.get_document(framework, version, level)
        if content:
            return content

        # Fall back to lower levels
        for fallback_level in [DocumentLevel.MODERATE, DocumentLevel.MINIMAL]:
            if level.value > fallback_level.value:
                content = self.get_document(framework, version, fallback_level)
                if content:
                    return content

        return f"Framework: {framework} (documentation not available)"

    def get_snapshot_info(
        self,
        framework: str,
        version: str,
        level: DocumentLevel,
    ) -> dict | None:
        """Get metadata about a document snapshot."""
        key = f"{framework}/{version}/{level.value}"
        return self._index.get(key)

    def list_frameworks(self) -> list[str]:
        """List all frameworks in the corpus."""
        frameworks = set()
        for key in self._index:
            framework = key.split("/")[0]
            frameworks.add(framework)
        return sorted(frameworks)

    def list_versions(self, framework: str) -> list[str]:
        """List all versions for a framework."""
        versions = set()
        for key in self._index:
            parts = key.split("/")
            if parts[0] == framework:
                versions.add(parts[1])
        return sorted(versions)

    def _get_doc_path(
        self,
        framework: str,
        version: str,
        level: DocumentLevel,
    ) -> Path:
        """Get the file path for a document."""
        return self.corpus_dir / framework / version / f"{level.value}.md"


def generate_minimal_doc(framework: str, description: str, install_cmd: str) -> str:
    """Generate minimal documentation level content.

    Args:
        framework: Framework name.
        description: One paragraph description.
        install_cmd: Installation command.

    Returns:
        Formatted minimal documentation.
    """
    return f"""# {framework}

{description}

## Installation

```bash
{install_cmd}
```
"""


def generate_level_from_parts(
    framework: str,
    quickstart: str,
    core_concepts: str = "",
    tool_use: str = "",
    api_reference: str = "",
    level: DocumentLevel = DocumentLevel.FULL,
) -> str:
    """Generate documentation at specified level from parts.

    Args:
        framework: Framework name.
        quickstart: Quickstart guide content.
        core_concepts: Core concepts content.
        tool_use: Tool use guide content.
        api_reference: API reference content.
        level: Target documentation level.

    Returns:
        Combined documentation for the level.
    """
    parts = [f"# {framework} Documentation\n"]

    if level in [DocumentLevel.MODERATE, DocumentLevel.FULL]:
        parts.append("## Quickstart\n")
        parts.append(quickstart)
        parts.append("\n")

        if core_concepts:
            parts.append("## Core Concepts\n")
            parts.append(core_concepts)
            parts.append("\n")

    if level == DocumentLevel.FULL:
        if tool_use:
            parts.append("## Tool Use\n")
            parts.append(tool_use)
            parts.append("\n")

        if api_reference:
            parts.append("## API Reference\n")
            parts.append(api_reference)
            parts.append("\n")

    return "\n".join(parts)
