"""Tests for dataset preparation and prompt formatting."""

import pytest
from train.dataset_prep import format_prompt, extract_schema


class TestFormatPrompt:
    def test_inference_prompt_no_sql(self):
        prompt = format_prompt(
            question="How many users are there?",
            schema="CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);",
            sql=None,
        )
        assert "<|user|>" in prompt
        assert "<|assistant|>" in prompt
        assert "How many users are there?" in prompt
        # Should end with assistant tag, no completion
        assert prompt.strip().endswith("<|assistant|>")

    def test_training_prompt_with_sql(self):
        sql = "SELECT COUNT(*) FROM users;"
        prompt = format_prompt(
            question="How many users are there?",
            schema="CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);",
            sql=sql,
        )
        assert sql in prompt
        assert "<|end|>" in prompt

    def test_prompt_contains_schema(self):
        schema = "CREATE TABLE orders (id INTEGER, total REAL);"
        prompt = format_prompt("Count orders", schema=schema, sql=None)
        assert schema in prompt

    def test_system_prompt_present(self):
        prompt = format_prompt("test question", "test schema", sql=None)
        assert "<|system|>" in prompt
        assert "SQL" in prompt


class TestExtractSchema:
    def test_basic_schema_extraction(self):
        # Minimal Spider-style table metadata
        tables = {
            "table_names_original": ["users"],
            "column_names_original": [[-1, "*"], [0, "id"], [0, "name"]],
            "column_types": ["text", "number", "text"],
            "primary_keys": [1],
            "foreign_keys": [],
        }
        schema = extract_schema(tables)
        assert "CREATE TABLE users" in schema
        assert "id" in schema
        assert "name" in schema

    def test_schema_with_foreign_keys(self):
        tables = {
            "table_names_original": ["orders", "customers"],
            "column_names_original": [
                [-1, "*"],
                [0, "id"], [0, "customer_id"],
                [1, "id"],
            ],
            "column_types": ["text", "number", "number", "number"],
            "primary_keys": [1, 3],
            "foreign_keys": [[2, 3]],
        }
        schema = extract_schema(tables)
        assert "FK:" in schema
        assert "orders" in schema
        assert "customers" in schema


def test_spider_requires_schema_join():
    from train.dataset_prep import format_example
    with pytest.raises(ValueError, match="Missing schema"):
        format_example({"db_id": "demo", "question": "Count rows", "query": "SELECT 1"})


def test_schema_join_retains_database_id():
    from train.dataset_prep import format_example
    metadata = {"table_names_original": ["users"], "column_names_original": [[0, "id"]],
                "column_types": ["number"], "primary_keys": [0], "foreign_keys": []}
    row = format_example({"db_id": "demo", "question": "Count users", "query": "SELECT COUNT(*) FROM users"}, {"demo": metadata})
    assert row["db_id"] == "demo"
    assert "CREATE TABLE users" in row["schema"]
