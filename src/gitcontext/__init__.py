"""GitContext: your codebase, answered.

Retrieval-augmented Q&A over source code, git history, pull requests, discussions, docs and
incident postmortems, with inline citations and evidence-based refusal.
"""

from .engine import GitContext
from .models import Answer, Chunk, Citation, Document, SourceType

__all__ = ["Answer", "Chunk", "Citation", "Document", "GitContext", "SourceType"]
__version__ = "0.1.0"
