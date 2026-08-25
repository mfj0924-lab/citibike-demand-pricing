"""Tests for I/O utilities."""

import os, sys, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.io_utils import save_pickle, load_pickle


class TestIOUtils:

    def test_save_and_load_pickle_dict(self):
        data = {"a": 1, "b": [2, 3], "c": "hello"}
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            save_pickle(data, path)
            loaded = load_pickle(path)
            assert loaded == data
        finally:
            os.remove(path)

    def test_save_and_load_pickle_list(self):
        data = list(range(100))
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            save_pickle(data, path)
            loaded = load_pickle(path)
            assert loaded == data
        finally:
            os.remove(path)

    def test_save_pickle_creates_directory(self):
        import shutil

        tmpdir = tempfile.mkdtemp()
        nested = os.path.join(tmpdir, "a", "b", "test.pkl")
        try:
            save_pickle({"x": 1}, nested)
            assert os.path.exists(nested)
        finally:
            shutil.rmtree(tmpdir)
