"""Tests for evaluation metrics."""

import pytest
from eval.exact_match import normalise_sql, exact_match, compute_exact_match_accuracy
from eval.bleu_score import tokenise_sql, compute_bleu


class TestNormaliseSql:
    def test_lowercase(self):
        assert normalise_sql("SELECT * FROM users") == "select * from users"

    def test_strip_semicolon(self):
        assert normalise_sql("select * from users;") == "select * from users"

    def test_collapse_whitespace(self):
        assert normalise_sql("select  *   from   users") == "select * from users"

    def test_empty_string(self):
        assert normalise_sql("") == ""

    def test_normalise_quotes(self):
        assert '"name"' in normalise_sql('SELECT "name" FROM users')


class TestExactMatch:
    def test_identical(self):
        assert exact_match("select * from users", "select * from users")

    def test_case_insensitive(self):
        assert exact_match("SELECT * FROM users", "select * from users")

    def test_semicolon_ignored(self):
        assert exact_match("select * from users;", "select * from users")

    def test_different_queries(self):
        assert not exact_match(
            "select count(*) from users",
            "select * from users"
        )

    def test_whitespace_normalised(self):
        assert exact_match("select  *  from  users", "select * from users")


class TestComputeExactMatchAccuracy:
    def test_all_correct(self):
        preds = ["select * from users", "select count(*) from orders"]
        golds = ["select * from users", "select count(*) from orders"]
        result = compute_exact_match_accuracy(preds, golds)
        assert result["exact_match_accuracy"] == 1.0
        assert result["num_correct"] == 2

    def test_none_correct(self):
        preds = ["select * from foo", "select * from bar"]
        golds = ["select * from users", "select * from orders"]
        result = compute_exact_match_accuracy(preds, golds)
        assert result["exact_match_accuracy"] == 0.0
        assert result["num_correct"] == 0

    def test_partial_correct(self):
        preds = ["select * from users", "select * from wrong"]
        golds = ["select * from users", "select count(*) from orders"]
        result = compute_exact_match_accuracy(preds, golds)
        assert result["exact_match_accuracy"] == 0.5

    def test_length_mismatch_raises(self):
        with pytest.raises(AssertionError):
            compute_exact_match_accuracy(["a"], ["a", "b"])


class TestBLEU:
    def test_identical_high_bleu(self):
        preds = ["select * from users where id = 1"]
        golds = ["select * from users where id = 1"]
        result = compute_bleu(preds, golds)
        if result.get("bleu4") is not None:
            assert result["bleu4"] > 0.9

    def test_completely_different_low_bleu(self):
        preds = ["drop table users"]
        golds = ["select count(*) from orders group by category"]
        result = compute_bleu(preds, golds)
        if result.get("bleu4") is not None:
            assert result["bleu4"] < 0.3

    def test_tokenise_sql_spaces_punctuation(self):
        tokenised = tokenise_sql("SELECT count(*) FROM users WHERE id=1")
        assert "(" in tokenised
        assert ")" in tokenised
        assert "=" in tokenised


def test_case_sensitive_literals_are_preserved():
    assert not exact_match("SELECT 'Alice'", "SELECT 'alice'")
