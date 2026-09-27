"""
Unit tests for query interpretation module.
Tests:
- 4 supported intents:
  1. VEGETATION_LOSS
  2. VEGETATION_GAIN
  3. CHANGE_STATISTICS
  4. GENERAL_COMPARISON
- CDVQA dataset question templates
- Rejection of unrecognized / unsupported queries
"""

import pytest
from backend.query_interpretation import (
    interpret_query,
    interpret_query_rule_based,
    QueryIntent,
)


def test_vegetation_loss_queries():
    queries = [
        "Where has vegetation been lost?",
        "Where has vegetation decreased?",
        "Where did vegetation disappear?",
        "Which areas lost trees?",
        "Show deforestation and forest cleared",
        "Did the regions of trees decrease?",  # CDVQA format
    ]
    for q in queries:
        res = interpret_query(q)
        assert res.is_supported, f"Failed on: {q}"
        assert res.intent == QueryIntent.VEGETATION_LOSS.value, f"Mismatch for '{q}', got {res.intent}"
        assert res.confidence >= 0.70


def test_vegetation_gain_queries():
    queries = [
        "Where has vegetation increased?",
        "Which areas gained vegetation?",
        "Where has vegetation expanded?",
        "Show reforestation and forest regrowth",
        "Did the regions of low vegetation increase?",  # CDVQA format
    ]
    for q in queries:
        res = interpret_query(q)
        assert res.is_supported, f"Failed on: {q}"
        assert res.intent == QueryIntent.VEGETATION_GAIN.value, f"Mismatch for '{q}', got {res.intent}"


def test_statistics_queries():
    queries = [
        "How much vegetation changed?",
        "What percentage of vegetation was lost?",
        "Calculate quantitative statistics",
        "What is the percentage of changed areas?",  # CDVQA format
    ]
    for q in queries:
        res = interpret_query(q)
        assert res.is_supported, f"Failed on: {q}"
        assert res.intent == QueryIntent.CHANGE_STATISTICS.value, f"Mismatch for '{q}', got {res.intent}"


def test_general_comparison_queries():
    queries = [
        "Compare vegetation between the two dates.",
        "What changed in vegetation?",
        "Did the areas of non-vegetated ground surface change?",  # CDVQA format
        "Show the change overview between before and after",
    ]
    for q in queries:
        res = interpret_query(q)
        assert res.is_supported, f"Failed on: {q}"
        assert res.intent == QueryIntent.GENERAL_COMPARISON.value, f"Mismatch for '{q}', got {res.intent}"


def test_unsupported_queries_rejection():
    invalid_queries = [
        "What is the capital of France?",
        "Can you write a poem about flowers?",
        "Order pizza for lunch",
        "",
    ]
    for q in invalid_queries:
        res = interpret_query(q)
        assert not res.is_supported
        assert res.intent == ""
