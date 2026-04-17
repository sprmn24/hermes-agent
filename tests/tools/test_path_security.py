from pathlib import Path

import pytest

from tools.path_security import has_traversal_component, validate_within_dir


class TestValidateWithinDir:
    def test_path_inside_root_returns_none(self, tmp_path):
        child = tmp_path / "subdir" / "file.txt"
        child.parent.mkdir()
        child.touch()
        assert validate_within_dir(child, tmp_path) is None

    def test_traversal_escaping_root_returns_error(self, tmp_path):
        escaped = tmp_path / ".." / "outside.txt"
        result = validate_within_dir(escaped, tmp_path)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_root_itself_returns_none(self, tmp_path):
        assert validate_within_dir(tmp_path, tmp_path) is None

    def test_absolute_path_outside_root_returns_error(self, tmp_path):
        other = tmp_path.parent / "other_dir"
        result = validate_within_dir(other, tmp_path)
        assert isinstance(result, str)
        assert len(result) > 0


class TestHasTraversalComponent:
    def test_leading_traversal(self):
        assert has_traversal_component("../secret") is True

    def test_middle_traversal(self):
        assert has_traversal_component("foo/../bar") is True

    def test_normal_path(self):
        assert has_traversal_component("normal/path") is False

    def test_absolute_path_without_dotdot(self):
        assert has_traversal_component("/absolute/path/file.txt") is False

    def test_only_dotdot(self):
        assert has_traversal_component("..") is True
