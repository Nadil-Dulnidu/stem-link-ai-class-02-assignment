"""
Assignment 2: AI Food Safety Inspector
Zero-Shot Prompting with Structured Outputs

Your mission: Analyze restaurant reviews and complaints to detect health violations
using only clear instructions — no training examples needed!
"""

import os
import json
import re
from dotenv import load_dotenv
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


class ViolationCategory(Enum):
    TEMPERATURE_CONTROL = "Food Temperature Control"
    PERSONAL_HYGIENE = "Personal Hygiene"
    PEST_CONTROL = "Pest Control"
    CROSS_CONTAMINATION = "Cross Contamination"
    FACILITY_MAINTENANCE = "Facility Maintenance"
    UNKNOWN = "Unknown"


class SeverityLevel(Enum):
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class InspectionPriority(Enum):
    URGENT = "URGENT"
    HIGH = "HIGH"
    ROUTINE = "ROUTINE"
    LOW = "LOW"


@dataclass
class Violation:
    """Structured violation data"""
    category: str
    description: str
    severity: str
    evidence: str
    confidence: float


@dataclass
class InspectionReport:
    """Complete inspection analysis"""

    restaurant_name: str
    overall_risk_score: int
    violations: List[Violation]
    inspection_priority: str
    recommended_actions: List[str]
    follow_up_required: bool


class FoodSafetyInspector:
    """
    AI-powered food safety analyzer using zero-shot structured prompting.
    """

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.1):
        """Initialize with LLM for consistent violation detection."""
        # TODO: Initialize an LLM for consistent JSON-style outputs
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.analysis_chain = None
        self.risk_chain = None
        self._setup_chains()

    def _setup_chains(self):
        """
        TODO #1: Create zero-shot prompts for violation detection and risk assessment.

        Create TWO chains:
        1. analysis_chain: Detects violations and extracts evidence
        2. risk_chain: Calculates risk scores based on violations

        Requirements:
        - Must output valid JSON
        - Include all violation categories
        - Extract specific evidence quotes
        - Generate confidence scores
        """

        # TODO: Create violation detection prompt (as a raw template string)
        analysis_template_str = (
            "You are a careful food safety inspector AI. Analyze the following text for health code violations.\n\n"
            "Instructions:\n"
            " - Identify any and all possible violations from the categories: "
            "Food Temperature Control, Personal Hygiene, Pest Control, Cross Contamination, Facility Maintenance.\n"
            " - For each detected violation, output an object with keys: "
            "category, description, severity, evidence, confidence.\n"
            "   * category: one of the allowed categories exactly or 'Unknown'.\n"
            "   * description: a short human-readable description of the violation.\n"
            "   * severity: one of ['Critical','High','Medium','Low'].\n"
            "   * evidence: a short verbatim quote (or short paraphrase) from the text that supports the violation.\n"
            "   * confidence: a float between 0.0 and 1.0 representing confidence.\n"
            " - If no violations are found, return an empty JSON array: []\n"
            " - ALWAYS output valid JSON ONLY (no surrounding commentary).\n"
            "\n"
            "Handle ambiguous language by assigning lower confidence (e.g., 0.2-0.5). If the reviewer uses clear statements like "
            "'I saw a mouse' or 'food was left unrefrigerated', use higher confidence (0.7-1.0).\n\n"
            "Text to analyze: {review_text}\n\n"
            "Output JSON array of violation objects:"
        )

        # TODO: Create risk assessment prompt (as a raw template string)
        risk_template_str = (
            "You are a risk scoring assistant.\n\n"
            "Input: a JSON array of violations with fields: category, description, severity, evidence, confidence.\n\n"
            "Scoring rules (apply these to compute a single integer score 0-100):\n"
            " - Assign base points per severity: Critical=30, High=20, Medium=10, Low=5.\n"
            " - Multiply each violation's base points by its confidence (0.0-1.0).\n"
            " - Add an extra +10 points if category is 'Pest Control' (pests raise priority).\n"
            " - Cap final score at 100 and round to nearest integer.\n"
            " - If there are zero violations, return 0.\n\n"
            "Output: a single integer (0-100) and nothing else. Example: 47\n\n"
            "Violations: {violations}\n\n"
            "Risk Score (integer only):"
        )

        # TODO: Build PromptTemplate objects from the strings above
        analysis_template = PromptTemplate.from_template(analysis_template_str)
        risk_template = PromptTemplate.from_template(risk_template_str)

        # TODO: Set up the chains
        self.analysis_chain = analysis_template | self.llm | StrOutputParser()
        self.risk_chain = risk_template | self.llm | StrOutputParser()

    def detect_violations(self, text: str) -> List[Violation]:
        """
        TODO #2: Detect health violations from text input.

        Args:
            text: Review, complaint, or social media post

        Returns:
            List of Violation objects with evidence
        """

        # TODO: Use analysis_chain to detect violations
        try:
            raw_response = self.analysis_chain.invoke({"review_text": text})
            data = json.loads(raw_response)
            violations: List[Violation] = [Violation(**violation) for violation in data]
            return violations
        except Exception as e:
            print(f"Error detecting violations: {e}")
            return []

    def calculate_risk_score(self, violations: List[Violation]) -> Tuple[int, str]:
        """
        TODO #3: Calculate overall risk score and determine inspection priority.

        Args:
            violations: List of detected violations

        Returns:
            Tuple of (risk_score, inspection_priority)
        """

        # TODO: Implement risk scoring logic
        # Consider: severity levels, number of violations, categories affected
        raw_response = self.risk_chain.invoke({"violations": violations})
        risk_score = int(raw_response)
        priority = None
        if risk_score > 80:
            priority = InspectionPriority.URGENT.value
        elif risk_score > 50:
            priority = InspectionPriority.HIGH.value
        elif risk_score > 20:
            priority = InspectionPriority.ROUTINE.value
        else:
            priority = InspectionPriority.LOW.value
        return risk_score, priority

    def analyze_review(
        self, text: str, restaurant_name: str = "Unknown"
    ) -> InspectionReport:
        """
        TODO #4: Complete analysis pipeline for a single review.

        Args:
            text: Review text to analyze
            restaurant_name: Name of the restaurant

        Returns:
            Complete InspectionReport with all findings
        """

        # TODO: Implement full analysis pipeline
        # 1. Detect violations
        # 2. Calculate risk score
        # 3. Generate recommendations
        # 4. Create InspectionReport

        violations = self.detect_violations(text)
        risk_score, priority = self.calculate_risk_score(violations)
        recommendations = [violation.description for violation in violations]
        follow_up = any("Critical" in v.severity for v in violations) or risk_score >= 70
        return InspectionReport(
            restaurant_name=restaurant_name,
            overall_risk_score=risk_score,
            violations=violations,
            inspection_priority=priority,
            recommended_actions=recommendations,
            follow_up_required=follow_up,
        )

    def batch_analyze(self, reviews: List[Dict[str, str]]) -> InspectionReport:
        """
        TODO #5: Analyze multiple reviews for the same restaurant.

        Args:
            reviews: List of dicts with 'text' and 'source' keys

        Returns:
            Aggregated InspectionReport
        """

        # TODO: Implement aggregation logic
        # - Combine violations from multiple sources
        # - Weight by source reliability
        # - Remove duplicates
        # - Calculate aggregate risk score

        # Simple reliability mapping for sources
        source_weights = {
            "yelp": 1.0,
            "google": 1.0,
            "tripadvisor": 0.9,
            "zomato": 0.9,
            "twitter": 0.7,
            "facebook": 0.8,
            "unknown": 0.6,
        }

        aggregated = {}  # key -> dict with accumulated data
        for review in reviews:
            text = review.get("text", "")
            source = review.get("source", "unknown").lower()
            weight = source_weights.get(source, 0.6)
            vio_list = self.detect_violations(text)
            # filter false positives early
            vio_list = self.filter_false_positives(vio_list)
            for v in vio_list:
                # create dedup key
                desc_norm = re.sub(r"\s+", " ", v.description.strip().lower())
                key = f"{v.category}||{desc_norm}"
                if key not in aggregated:
                    aggregated[key] = {
                        "category": v.category,
                        "description": v.description,
                        "evidence": [v.evidence],
                        "severity": v.severity,
                        "confidence_accum": v.confidence * weight,
                        "weight_sum": weight,
                    }
                else:
                    aggregated[key]["evidence"].append(v.evidence)
                    aggregated[key]["confidence_accum"] += v.confidence * weight
                    aggregated[key]["weight_sum"] += weight
                    # escalate severity if any instance is more severe
                    existing_sev = aggregated[key]["severity"]
                    # tie-breaker: Critical > High > Medium > Low
                    order = ["low", "medium", "high", "critical"]
                    if order.index(v.severity.lower()) > order.index(existing_sev.lower()):
                        aggregated[key]["severity"] = v.severity

        # Build list of aggregated Violation objects
        final_violations: List[Violation] = []
        for k, info in aggregated.items():
            avg_confidence = (
                info["confidence_accum"] / info["weight_sum"]
                if info["weight_sum"] > 0
                else 0.0
            )
            # choose representative evidence (join and keep short)
            evidence = "; ".join(info["evidence"][:3])
            final_violations.append(
                Violation(
                    category=info["category"],
                    description=info["description"],
                    severity=info["severity"],
                    evidence=evidence,
                    confidence=max(0.0, min(1.0, avg_confidence)),
                )
            )

        # Compute aggregate risk using the same risk calculator
        risk_score, priority = self.calculate_risk_score(final_violations)
        # Recommended actions: same style as analyze_review
        recommendations = []
        for v in final_violations:
            if "Critical" in v.severity:
                rec = f"Immediate inspection: {v.description}"
            elif "High" in v.severity:
                rec = f"Correct promptly: {v.description}"
            elif "Medium" in v.severity:
                rec = f"Monitor & correct: {v.description}"
            else:
                rec = f"Address during routine check: {v.description}"
            recommendations.append(rec)

        follow_up = any("Critical" in v.severity for v in final_violations) or risk_score >= 70

        return InspectionReport(
            restaurant_name=reviews[0].get("restaurant", "Unknown")
            if reviews and "restaurant" in reviews[0]
            else "Unknown",
            overall_risk_score=risk_score,
            violations=final_violations,
            inspection_priority=priority,
            recommended_actions=recommendations,
            follow_up_required=follow_up,
        )

    def filter_false_positives(self, violations: List[Violation]) -> List[Violation]:
        """
        TODO #6 (Bonus): Filter out likely false positives.

        Consider:
        - Sarcasm indicators
        - Exaggeration patterns
        - Confidence thresholds
        """

        # TODO: Implement false positive filtering
        filtered: List[Violation] = []
        sarcasm_patterns = [
            r"\bjust kidding\b",
            r"\badded extra protein\b",
            r"\bnot really\b",
            r"\bjust joking\b",
            r"\b(?:lol|jk|lmao)\b",
            r"#sushitime",  # example hashtag that may signal lightheartedness
        ]
        sarcasm_re = re.compile("|".join(sarcasm_patterns), flags=re.IGNORECASE)
        for v in violations:
            if v.confidence < 0.35:
                # likely too uncertain
                continue
            combined_text = f"{v.description} {v.evidence}"
            if sarcasm_re.search(combined_text):
                # likely sarcastic or joking — drop or lower confidence
                # If confidence is very high, keep; else drop
                if v.confidence >= 0.85:
                    filtered.append(v)
                else:
                    continue
            else:
                filtered.append(v)
        return filtered


def test_inspector():
    """Test the food safety inspector with various scenarios."""

    inspector = FoodSafetyInspector()

    # Test cases with varying violation types
    test_reviews = [
        {
            "restaurant": "Bob's Burgers",
            "text": "Great food but saw a mouse run across the dining room! Also, the chef wasn't wearing gloves while handling raw chicken.",
        },
        {
            "restaurant": "Pizza Palace",
            "text": "Just left and the bathroom had no soap, and I'm pretty sure that meat sitting on the counter wasn't refrigerated 😷",
        },
        {
            "restaurant": "Sushi Express",
            "text": "Love this place! Though it's weird they keep the raw fish next to the vegetables #sushitime #questionable",
        },
        {
            "restaurant": "Taco Town",
            "text": "Best tacos in town! Super clean kitchen, staff always wears hairnets, everything looks fresh!",
        },
        {
            "restaurant": "Burger Barn",
            "text": "The cockroach in my salad added extra protein! Just kidding, but seriously the place needs cleaning.",
        },
    ]

    print("🍽️ FOOD SAFETY INSPECTION SYSTEM 🍽️\n")
    print("=" * 70)

    for review_data in test_reviews:
        print(f"\n🏪 Restaurant: {review_data['restaurant']}")
        print(f"📝 Review: \"{review_data['text'][:100]}...\"")

        # Analyze the review
        report = inspector.analyze_review(
            review_data["text"], review_data["restaurant"]
        )

        # Display results
        print(f"\n📊 Inspection Report:")
        print(f"  Risk Score: {report.overall_risk_score}/100")
        print(f"  Priority: {report.inspection_priority}")
        print(f"  Violations Found: {len(report.violations)}")

        if report.violations:
            print("\n  Detected Violations:")
            for v in report.violations:
                print(f"    • [{v.severity}] {v.category}: {v.description}")
                print(f'      Evidence: "{v.evidence[:50]}..."')
                print(f"      Confidence: {v.confidence:.0%}")

        if report.recommended_actions:
            print("\n  Recommended Actions:")
            for action in report.recommended_actions:
                print(f"    ✓ {action}")

        print(f"\n  Follow-up Required: {'Yes' if report.follow_up_required else 'No'}")
        print("-" * 70)

    # Test batch analysis
    print("\n🔬 BATCH ANALYSIS TEST:")
    print("=" * 70)

    # Multiple reviews for same restaurant
    batch_reviews = [
        {"text": "Saw bugs in the kitchen!", "source": "Yelp"},
        {"text": "Food was cold and undercooked", "source": "Google"},
        {"text": "Staff not wearing hairnets", "source": "Twitter"},
    ]

    # TODO: Uncomment when batch_analyze is implemented
    batch_report = inspector.batch_analyze(batch_reviews)
    print(f"Aggregate Risk Score: {batch_report.overall_risk_score}/100")
    print(f"Total Violations: {len(batch_report.violations)}")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Set OPENAI_API_KEY before running.")
    test_inspector()
