import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parsers.diff_parser import GitDiffParser
from src.models import ChangeType


SAMPLE_DIFF = """diff --git a/foo.py b/foo.py
index 1234567..abcdefg 100644
--- a/foo.py
+++ b/foo.py
@@ -10,5 +10,8 @@ def old_func():
     pass
 
 def old_func():
-    return None
+    return "hello"
+    password = "sk-1234567890"
+    eval(user_input)
+    os.system("rm -rf /")
 def end():
     pass
"""


def test_parse_basic():
    parser = GitDiffParser()
    hunks = parser.parse(SAMPLE_DIFF)
    assert len(hunks) == 1
    hunk = hunks[0]
    assert hunk.file_path == "foo.py"
    assert hunk.old_start == 10
    assert hunk.new_start == 10


def test_parse_added_lines():
    parser = GitDiffParser()
    hunks = parser.parse(SAMPLE_DIFF)
    hunk = hunks[0]
    added = hunk.added_lines
    assert len(added) == 4
    assert 'return "hello"' in added[0].content
    assert added[0].new_line_no is not None


def test_parse_removed_lines():
    parser = GitDiffParser()
    hunks = parser.parse(SAMPLE_DIFF)
    hunk = hunks[0]
    removed = hunk.removed_lines
    assert len(removed) == 1
    assert "return None" in removed[0].content


def test_parse_multiple_hunks():
    diff = """diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1,2 +1,3 @@
 a
+b
 c
@@ -10,2 +11,3 @@
 d
+e
 f
"""
    parser = GitDiffParser()
    hunks = parser.parse(diff)
    assert len(hunks) == 2
    assert hunks[0].new_start == 1
    assert hunks[1].new_start == 11


def test_parse_new_file():
    diff = """diff --git a/new.py b/new.py
new file mode 100644
--- /dev/null
+++ b/new.py
@@ -0,0 +1,2 @@
+import os
+print("hello")
"""
    parser = GitDiffParser()
    hunks = parser.parse(diff)
    assert len(hunks) == 1
    assert hunks[0].file_path == "new.py"
    assert len(hunks[0].added_lines) == 2


def test_parse_empty_diff():
    parser = GitDiffParser()
    hunks = parser.parse("")
    assert hunks == []


def test_parse_consecutive_files_does_not_include_file_headers_in_hunks():
    """A following diff header must flush the current hunk cleanly."""
    diff = """diff --git a/new.py b/new.py
new file mode 100644
--- /dev/null
+++ b/new.py
@@ -0,0 +1,1 @@
+print(\"new\")
diff --git a/deleted.py b/deleted.py
deleted file mode 100644
--- a/deleted.py
+++ /dev/null
@@ -1,1 +0,0 @@
-print(\"old\")
"""
    hunks = GitDiffParser().parse(diff)
    assert len(hunks) == 2
    assert [line.content for line in hunks[0].added_lines] == ['print("new")']
    assert [line.content for line in hunks[1].removed_lines] == ['print("old")']
