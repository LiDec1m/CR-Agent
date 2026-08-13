"""Git diff parser: raw diff text → structured HunkInfo list."""

from __future__ import annotations

import re

from src.models import ChangeType, DiffLine, HunkInfo

_HUNK_HEADER_RE = re.compile(
    r"^@@ -(\d+),?(\d*) \+(\d+),?(\d*) @@(.*)$"
)
_FILE_HEADER_RE = re.compile(r"^diff --git a/(.+) b/(.+)$")


class GitDiffParser:
    """Parse unified git diff output into HunkInfo objects."""

    def parse(self, raw_diff: str) -> list[HunkInfo]:
        if not raw_diff.strip():
            return []

        hunks: list[HunkInfo] = []
        current_file: str | None = None
        current_hunk: HunkInfo | None = None
        old_line = 0
        new_line = 0

        for line in raw_diff.splitlines():
            file_match = _FILE_HEADER_RE.match(line)
            if file_match:
                # A new file begins even when the preceding hunk reaches the
                # next file header without an explicit trailing delimiter.
                # Flush it here so metadata lines such as ---/+++ are never
                # misclassified as removed or added source lines.
                if current_hunk is not None:
                    hunks.append(current_hunk)
                    current_hunk = None
                current_file = file_match.group(2)
                continue

            hunk_match = _HUNK_HEADER_RE.match(line)
            if hunk_match:
                if current_hunk is not None:
                    hunks.append(current_hunk)
                old_start = int(hunk_match.group(1))
                old_count = int(hunk_match.group(2)) if hunk_match.group(2) else 1
                new_start = int(hunk_match.group(3))
                new_count = int(hunk_match.group(4)) if hunk_match.group(4) else 1
                section = hunk_match.group(5).strip()
                current_hunk = HunkInfo(
                    file_path=current_file or "unknown",
                    old_start=old_start,
                    old_count=old_count,
                    new_start=new_start,
                    new_count=new_count,
                    section_header=section,
                )
                old_line = old_start
                new_line = new_start
                continue

            if current_hunk is None:
                continue

            if line.startswith("+"):
                current_hunk.lines.append(
                    DiffLine(
                        content=line[1:],
                        change_type=ChangeType.ADDED,
                        new_line_no=new_line,
                    )
                )
                new_line += 1
            elif line.startswith("-"):
                current_hunk.lines.append(
                    DiffLine(
                        content=line[1:],
                        change_type=ChangeType.REMOVED,
                        old_line_no=old_line,
                    )
                )
                old_line += 1
            else:
                content = line[1:] if line.startswith(" ") else line
                current_hunk.lines.append(
                    DiffLine(
                        content=content,
                        change_type=ChangeType.CONTEXT,
                        old_line_no=old_line,
                        new_line_no=new_line,
                    )
                )
                old_line += 1
                new_line += 1

        if current_hunk is not None:
            hunks.append(current_hunk)

        return hunks
