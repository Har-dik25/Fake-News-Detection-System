"""
tests/test_data_loader.py — Unit tests for data loading and preprocessing.
"""
import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import NewsDataset, prepare_sequences


class TestNewsDataset:
    """Tests for the NewsDataset class."""

    def test_clean_text_lowercase(self):
        result = NewsDataset.clean_text("HELLO WORLD")
        assert result == "hello world"

    def test_clean_text_removes_html(self):
        result = NewsDataset.clean_text("Hello <b>World</b>")
        assert "<b>" not in result
        assert "world" in result

    def test_clean_text_handles_none(self):
        result = NewsDataset.clean_text(None)
        assert result == ""

    def test_clean_text_handles_int(self):
        result = NewsDataset.clean_text(123)
        assert result == ""

    def test_clean_text_strips_whitespace(self):
        result = NewsDataset.clean_text("  lots   of   spaces  ")
        assert result == "lots of spaces"


class TestPrepareSequences:
    """Tests for the prepare_sequences function."""

    def setup_method(self):
        self.vocab = {"<PAD>": 0, "<UNK>": 1, "hello": 2, "world": 3, "test": 4}

    def test_basic_padding(self):
        tokens_list = [["hello", "world"]]
        result = prepare_sequences(tokens_list, self.vocab, max_len=5)
        assert result.shape == (1, 5)
        assert result[0][0] == 2  # hello
        assert result[0][1] == 3  # world
        assert result[0][2] == 0  # PAD
        assert result[0][3] == 0  # PAD
        assert result[0][4] == 0  # PAD

    def test_truncation(self):
        tokens_list = [["hello", "world", "test", "hello", "world"]]
        result = prepare_sequences(tokens_list, self.vocab, max_len=3)
        assert result.shape == (1, 3)

    def test_unknown_tokens(self):
        tokens_list = [["hello", "unknown_word"]]
        result = prepare_sequences(tokens_list, self.vocab, max_len=3)
        assert result[0][1] == 1  # UNK

    def test_empty_tokens(self):
        tokens_list = [[]]
        result = prepare_sequences(tokens_list, self.vocab, max_len=3)
        assert result.shape == (1, 3)
        assert all(result[0] == 0)  # All PAD

    def test_max_len_512(self):
        """Verify the system supports the updated 512 token limit."""
        tokens_list = [["hello"] * 600]
        result = prepare_sequences(tokens_list, self.vocab, max_len=512)
        assert result.shape == (1, 512)

    def test_multiple_sequences(self):
        tokens_list = [["hello", "world"], ["test"]]
        result = prepare_sequences(tokens_list, self.vocab, max_len=4)
        assert result.shape == (2, 4)
