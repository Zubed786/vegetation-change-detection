"""
Query Interpretation Module.
Supports 4 structured intents:
1. VEGETATION_LOSS
2. VEGETATION_GAIN
3. CHANGE_STATISTICS
4. GENERAL_COMPARISON
Employs a hybrid approach: rule-based semantic parsing with regex pattern weighting
and optional LLM structuring, plus strict backend intent validation.
"""

from dataclasses import dataclass
from enum import Enum
import re
from typing import Dict, List, Optional, Tuple, Any


class QueryIntent(str, Enum):
    VEGETATION_LOSS = "VEGETATION_LOSS"
    VEGETATION_GAIN = "VEGETATION_GAIN"
    CHANGE_STATISTICS = "CHANGE_STATISTICS"
    GENERAL_COMPARISON = "GENERAL_COMPARISON"


SUPPORTED_INTENTS = {intent.value for intent in QueryIntent}


@dataclass
class InterpretationResult:
    intent: str
    confidence: float
    query: str
    method: str  # "rule_based" or "llm"
    matched_patterns: List[str]
    is_supported: bool
    rejection_reason: Optional[str] = None


# Pattern lists with weights
LOSS_PATTERNS = [
    (r"\b(lost|loss|decreased|decrease|disappear|disappeared|reduction|reduced|destroyed|degraded|cleared|removed|deforestation)\b", 0.95),
    (r"\bwhere\s+(has\s+)?(vegetation|forest|greenery)\s+(been\s+)?(lost|decreased|reduced|cleared)\b", 0.99),
    (r"\bwhere\s+did\s+(vegetation|forest)\s+disappear\b", 0.98),
    (r"\bwhich\s+areas?\s+(lost|lost\s+any)\s+(vegetation|greenery|trees)\b", 0.96),
    (r"\b(vegetation|forest|tree)\s+(loss|decrease|reduction)\b", 0.94),
    (r"\bhow\s+much\s+(vegetation|forest)\s+was\s+lost\b", 0.92),  # if specific to loss
]

GAIN_PATTERNS = [
    (r"\b(gain|gained|gaining|increased|increase|growth|grown|expanded|expansion|reforested|reforestation|afforestation|re-growth|regrowth)\b", 0.95),
    (r"\bwhere\s+(has\s+)?(vegetation|forest|greenery)\s+(been\s+)?(increased|grown|expanded)\b", 0.99),
    (r"\bwhich\s+areas?\s+gained\s+(vegetation|greenery|trees)\b", 0.98),
    (r"\b(vegetation|forest|tree)\s+(gain|increase|expansion)\b", 0.94),
    (r"\bhow\s+much\s+(vegetation|forest)\s+was\s+gained\b", 0.92),
]

STATS_PATTERNS = [
    (r"\b(how\s+much|how\s+many|percentage|percent|hectares|ha|km2|quantify|quantitative|statistics|stats|amount|total\s+change|numbers)\b", 0.95),
    (r"\bhow\s+much\s+(vegetation|area|land)?\s*(changed|altered)\b", 0.98),
    (r"\bwhat\s+percentage\s+(of\s+vegetation\s+)?(was\s+lost|changed|gained)\b", 0.97),
    (r"\b(calculate|report|give)\s+(the\s+)?(statistics|metrics|numbers|areas)\b", 0.94),
]

COMPARISON_PATTERNS = [
    (r"\b(compare|comparison|difference|differ|what\s+changed|overview|overall|between\s+(the\s+)?two|before\s+and\s+after|inspect|show\s+change)\b", 0.92),
    (r"\bcompare\s+vegetation\s+between\s+(the\s+)?two\s+dates\b", 0.98),
    (r"\bwhat\s+changed\s+(in\s+vegetation|between|overall)\b", 0.97),
    (r"\bwhere\s+is\s+the\s+change\b", 0.90),
    (r"\b(did|have)\s+the\s+areas?\s+of\s+.*\s+change\b", 0.91),  # CDVQA question template
]


def interpret_query_rule_based(query: str) -> InterpretationResult:
    """
    Interprets natural language queries deterministically using weighted regex scoring.
    Guarantees strict categorization into the 4 supported intents or rejection.
    """
    q_norm = query.strip().lower()

    if not q_norm:
        return InterpretationResult(
            intent="",
            confidence=0.0,
            query=query,
            method="rule_based",
            matched_patterns=[],
            is_supported=False,
            rejection_reason="Query is empty.",
        )

    scores: Dict[str, float] = {
        QueryIntent.VEGETATION_LOSS.value: 0.0,
        QueryIntent.VEGETATION_GAIN.value: 0.0,
        QueryIntent.CHANGE_STATISTICS.value: 0.0,
        QueryIntent.GENERAL_COMPARISON.value: 0.0,
    }
    matches: Dict[str, List[str]] = {k: [] for k in scores}

    # Evaluate Loss
    for pattern, weight in LOSS_PATTERNS:
        m = re.findall(pattern, q_norm)
        if m:
            scores[QueryIntent.VEGETATION_LOSS.value] += weight
            matches[QueryIntent.VEGETATION_LOSS.value].append(pattern)

    # Evaluate Gain
    for pattern, weight in GAIN_PATTERNS:
        m = re.findall(pattern, q_norm)
        if m:
            scores[QueryIntent.VEGETATION_GAIN.value] += weight
            matches[QueryIntent.VEGETATION_GAIN.value].append(pattern)

    # Evaluate Stats
    for pattern, weight in STATS_PATTERNS:
        m = re.findall(pattern, q_norm)
        if m:
            scores[QueryIntent.CHANGE_STATISTICS.value] += weight
            matches[QueryIntent.CHANGE_STATISTICS.value].append(pattern)

    # Evaluate General Comparison
    for pattern, weight in COMPARISON_PATTERNS:
        m = re.findall(pattern, q_norm)
        if m:
            scores[QueryIntent.GENERAL_COMPARISON.value] += weight
            matches[QueryIntent.GENERAL_COMPARISON.value].append(pattern)

    # Specific disambiguation:
    # If question is "how much vegetation changed?", stats should rank above general comparison
    if "how much" in q_norm or "percentage" in q_norm or "quant" in q_norm:
        scores[QueryIntent.CHANGE_STATISTICS.value] += 0.5

    best_intent = max(scores, key=lambda k: scores[k])
    best_score = scores[best_intent]

    if best_score == 0.0:
        # Check if generic remote sensing question without keyword
        # E.g. "what is happening here?", "is there any change?"
        if "change" in q_norm or "differ" in q_norm:
            best_intent = QueryIntent.GENERAL_COMPARISON.value
            best_score = 0.70
            matches[best_intent].append("generic_change_fallback")
        else:
            return InterpretationResult(
                intent="",
                confidence=0.0,
                query=query,
                method="rule_based",
                matched_patterns=[],
                is_supported=False,
                rejection_reason=(
                    f"Could not match query '{query}' to any supported intent. "
                    "Supported intents: VEGETATION_LOSS, VEGETATION_GAIN, "
                    "CHANGE_STATISTICS, GENERAL_COMPARISON."
                ),
            )

    # Normalize confidence to [0.70, 0.99]
    confidence = min(0.99, max(0.70, 0.75 + (best_score * 0.12)))

    return InterpretationResult(
        intent=best_intent,
        confidence=round(confidence, 2),
        query=query,
        method="rule_based",
        matched_patterns=matches[best_intent],
        is_supported=True,
    )


def interpret_query(query: str, llm_callable: Optional[Any] = None) -> InterpretationResult:
    """
    Hybrid query interpreter.
    Attempts LLM interpretation if provided and structured;
    falls back cleanly to deterministic rule-based analysis.
    Validates that the returned intent is strictly one of the 4 supported intents.
    """
    if llm_callable is not None:
        try:
            llm_result = llm_callable(query)
            if isinstance(llm_result, dict):
                intent = llm_result.get("intent", "").upper()
                confidence = float(llm_result.get("confidence", 0.90))
                if intent in SUPPORTED_INTENTS:
                    return InterpretationResult(
                        intent=intent,
                        confidence=round(min(1.0, max(0.0, confidence)), 2),
                        query=query,
                        method="llm",
                        matched_patterns=["llm_structured"],
                        is_supported=True,
                    )
        except Exception:
            pass  # Fall back to rule-based parser on any LLM failure

    return interpret_query_rule_based(query)
