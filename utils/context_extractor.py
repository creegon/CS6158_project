"""Utility helpers for extracting source-context for test cases from cloned projects.

The FlakyLens dataset only stores the raw test method code.  When we want to
understand how the test behaves inside the original repository we have to
inspect the surrounding file in the upstream project.  This module provides a
`ProjectContextFetcher` that assumes the repository has been cloned locally and
offers convenience methods for locating the file, extracting the method block
with context lines, and discovering call-sites inside the repository.

The fetcher expects repositories to live under ``external_projects/`` (the
folder chosen in this workspace).  We map dataset project slugs such as
``apache_hadoop`` to GitHub URLs (``https://github.com/apache/hadoop``) and to a
local working tree.  The helper is intentionally lightweight and only relies on
the Python standard library so that it works in restricted environments.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional


PROJECT_BASE_DIR = Path("external_projects")


@dataclass
class InvocationMatch:
    """Container representing a single method invocation inside the repo."""

    file_path: Path
    line_number: int
    line_preview: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "file_path": str(self.file_path).replace(os.sep, "/"),
            "line_number": self.line_number,
            "line_preview": self.line_preview.strip(),
        }


class ProjectContextFetcher:
    """Extract context for dataset entries from locally cloned Git projects.

    Usage::

        fetcher = ProjectContextFetcher()
        context = fetcher.get_test_context(
            project="apache_hadoop",
            test_name="TestDelegationTokenRenewer.testAddRemoveRenewAction",
        )

    The resulting dictionary contains the resolved file path, the method block,
    a larger context window around the method, and the first few invocation
    sites discovered in the repository.
    """

    def __init__(self, repos_base: Path | str = PROJECT_BASE_DIR):
        self.repos_base = Path(repos_base)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ensure_repo(self, project: str, remote_url: Optional[str] = None) -> Path:
        """Ensure that the repository for ``project`` exists locally.

        If the working tree is missing this method attempts to ``git clone`` it
        (shallow depth) from GitHub.  The caller can override the remote URL via
        ``remote_url`` if a different hosting provider is required.
        """

        repo_path = self._resolve_repo_path(project)
        if repo_path.exists():
            return repo_path

        owner, repo = self._split_project_slug(project)
        remote = remote_url or f"https://github.com/{owner}/{repo}"
        repo_path.parent.mkdir(parents=True, exist_ok=True)

        clone_args = [
            "git",
            "clone",
            "--depth",
            "1",
            remote,
            str(repo_path),
        ]
        subprocess.run(clone_args, check=True, cwd=str(repo_path.parent))
        return repo_path

    def get_test_context(
        self,
        project: str,
        test_name: str,
        *,
        context_lines: int = 20,
        invocation_limit: int = 10,
    ) -> Dict[str, object]:
        """Return structured context for ``test_name`` inside ``project``.

        The result contains:

        - ``file_path``: repository-relative path of the test file
        - ``class_name`` / ``method_name``: parsed identifiers
        - ``method_block``: full method body including braces
        - ``surrounding_window``: a larger snippet around the method declaration
        - ``invocations``: a list of up to ``invocation_limit`` call-sites
        """

        repo_path = self.ensure_repo(project)
        class_name, method_name = self._split_test_name(test_name)

        test_file = self._locate_test_file(repo_path, class_name)
        if test_file is None:
            raise FileNotFoundError(
                f"Could not find Java file for class '{class_name}' in {repo_path}"
            )

        rel_file = test_file.relative_to(repo_path)
        lines = test_file.read_text(encoding="utf-8", errors="ignore").splitlines()

        method_idx = self._find_method_signature(lines, method_name)
        if method_idx is None:
            raise ValueError(
                f"Method '{method_name}' not found in {test_file}"
            )

        annotation_block = self._gather_leading_annotations(lines, method_idx)
        method_block = self._extract_method_block(lines, method_idx)
        window_snippet = self._extract_window(lines, method_idx, context_lines)
        invocations = self._search_invocations(
            repo_path=repo_path,
            method_name=method_name,
            exclude_file=test_file,
            limit=invocation_limit,
        )

        return {
            "project": project,
            "repo_path": str(repo_path),
            "file_path": str(rel_file).replace(os.sep, "/"),
            "class_name": class_name,
            "method_name": method_name,
            "annotations": annotation_block,
            "method_block": method_block,
            "surrounding_window": window_snippet,
            "invocations": [match.to_dict() for match in invocations],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _split_project_slug(project: str) -> tuple[str, str]:
        if "_" not in project:
            raise ValueError(
                f"Unexpected project slug '{project}'. Expected format 'owner_repo'."
            )
        owner, repo = project.split("_", 1)
        return owner, repo

    def _resolve_repo_path(self, project: str) -> Path:
        """Return the preferred local checkout directory for ``project``."""

        owner, repo = self._split_project_slug(project)

        candidates: Iterable[Path] = (
            self.repos_base / project,  # e.g. external_projects/apache_hadoop
            self.repos_base / repo,  # e.g. external_projects/hadoop
            self.repos_base / f"{owner}-{repo}",
        )

        for candidate in candidates:
            if candidate.exists():
                return candidate

        # prefer storing future clones in external_projects/{repo}
        return self.repos_base / repo

    @staticmethod
    def _split_test_name(test_name: str) -> tuple[str, str]:
        if "." not in test_name:
            raise ValueError(
                f"Test name '{test_name}' does not contain a class separator '.'"
            )
        # 处理多层级的类名（如 com.example.TestClass.testMethod）
        # 只取最后的类名和方法名
        class_name, method_name = test_name.rsplit(".", 1)
        return class_name, method_name

    def _locate_test_file(self, repo_path: Path, class_name: str) -> Optional[Path]:
        # Strategy 1: filename matches class name exactly.
        # 处理内部类（用$分隔）
        simple_class_name = class_name.split('$')[0] if '$' in class_name else class_name
        
        # 如果类名包含包路径，提取最后的类名
        if '.' in simple_class_name:
            simple_class_name = simple_class_name.split('.')[-1]
        
        exact_matches = list(repo_path.rglob(f"{simple_class_name}.java"))
        if exact_matches:
            return exact_matches[0]

        # Strategy 2: fall back to text search for class declaration.
        # 搜索简化的类名
        class_pattern = re.compile(rf"\bclass\s+{re.escape(simple_class_name)}\b")
        for java_file in repo_path.rglob("*.java"):
            try:
                content = java_file.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if class_pattern.search(content):
                return java_file
        
        # Strategy 3: 尝试原始类名（如果与简化类名不同）
        if simple_class_name != class_name:
            original_matches = list(repo_path.rglob(f"{class_name}.java"))
            if original_matches:
                return original_matches[0]
        
        return None

    @staticmethod
    def _find_method_signature(lines: List[str], method_name: str) -> Optional[int]:
        # 尝试多种模式匹配
        patterns = [
            # 标准方法: methodName(
            re.compile(rf"\b{re.escape(method_name)}\s*\("),
            # 带泛型的方法: methodName<T>(
            re.compile(rf"\b{re.escape(method_name)}\s*<[^>]+>\s*\("),
            # 方法前可能有修饰符: public void methodName(
            re.compile(rf"\s{re.escape(method_name)}\s*\("),
        ]
        
        for idx, line in enumerate(lines):
            # 跳过注释行
            stripped = line.strip()
            if stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
                continue
            
            # 尝试所有模式
            for pattern in patterns:
                if pattern.search(line):
                    return idx
        
        return None

    @staticmethod
    def _gather_leading_annotations(lines: List[str], start_idx: int) -> str:
        annotations: List[str] = []
        idx = start_idx - 1
        while idx >= 0:
            stripped = lines[idx].strip()
            if not stripped.startswith("@"):
                break
            annotations.insert(0, stripped)
            idx -= 1
        return "\n".join(annotations)

    @staticmethod
    def _extract_method_block(lines: List[str], start_idx: int) -> str:
        brace_depth = 0
        snippet: List[str] = []
        started = False

        for idx in range(start_idx, len(lines)):
            line = lines[idx]
            snippet.append(line)

            if "{" in line:
                brace_depth += line.count("{")
                started = True
            if "}" in line:
                brace_depth -= line.count("}")

            if started and brace_depth <= 0:
                break

        return "\n".join(snippet)

    @staticmethod
    def _extract_window(
        lines: List[str],
        center_idx: int,
        context_lines: int,
    ) -> str:
        start = max(center_idx - context_lines, 0)
        end = min(center_idx + context_lines, len(lines))
        return "\n".join(lines[start:end])

    def _search_invocations(
        self,
        repo_path: Path,
        method_name: str,
        *,
        exclude_file: Path,
        limit: int,
    ) -> List[InvocationMatch]:
        pattern = re.compile(rf"\b{re.escape(method_name)}\s*\(")
        matches: List[InvocationMatch] = []

        for java_file in repo_path.rglob("*.java"):
            if java_file.resolve() == exclude_file.resolve():
                continue
            try:
                lines = java_file.read_text(encoding="utf-8", errors="ignore").splitlines()
            except OSError:
                continue

            for idx, line in enumerate(lines):
                if pattern.search(line):
                    matches.append(
                        InvocationMatch(
                            file_path=java_file.relative_to(repo_path),
                            line_number=idx + 1,
                            line_preview=line.strip(),
                        )
                    )
                    if len(matches) >= limit:
                        return matches
        return matches
