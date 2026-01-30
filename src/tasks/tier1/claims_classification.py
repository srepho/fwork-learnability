"""Insurance claims classification task (Tier 1).

This task tests basic framework setup and model invocation.
The LLM must use the framework to classify insurance claims as 'simple' or 'complex'.
"""

from ..base import Task, TaskTier, TestCase


class ClaimsClassificationTask(Task):
    """Classify insurance claims as simple or complex.

    Simple claims:
    - Single vehicle damage
    - Clear liability
    - No injuries
    - Damage under $5000

    Complex claims:
    - Multiple parties
    - Disputed liability
    - Personal injury
    - Commercial vehicles
    - Damage over $10000
    """

    def __init__(self):
        super().__init__(
            task_id="tier1_claims_classification",
            name="Insurance Claims Classification",
            tier=TaskTier.TIER_1,
            description=self._get_description(),
            expected_interface=self._get_interface(),
            development_tests=self._get_dev_tests(),
            hidden_tests=self._get_hidden_tests(),
        )

    def _get_description(self) -> str:
        return """Create a claims classification system that categorizes insurance claims as either 'simple' or 'complex'.

**Classification Criteria:**

SIMPLE claims have ALL of these characteristics:
- Single vehicle involved OR property damage only
- Clear, undisputed liability
- No personal injuries
- Estimated damage/loss under $5,000

COMPLEX claims have ANY of these characteristics:
- Multiple vehicles or parties involved
- Disputed or unclear liability
- Personal injury claims
- Commercial vehicles involved
- Estimated damage/loss over $10,000
- Potential fraud indicators

**Input:** A claim dictionary with fields:
- claim_id: Unique identifier
- description: Text description of the incident
- damage_estimate: Estimated cost in dollars
- parties_involved: Number of parties
- has_injury: Boolean indicating if injuries occurred
- liability_clear: Boolean indicating if liability is undisputed
- vehicle_type: 'personal' or 'commercial'

**Output:** A result with:
- claim_id: The input claim ID
- classification: Either 'simple' or 'complex'
- confidence: A float between 0 and 1
- reasoning: Brief explanation of the classification
"""

    def _get_interface(self) -> str:
        return """Your solution must provide a `classify_claim` function:

```python
def classify_claim(claim: dict) -> dict:
    '''
    Classify an insurance claim as simple or complex.

    Args:
        claim: Dictionary with keys: claim_id, description, damage_estimate,
               parties_involved, has_injury, liability_clear, vehicle_type

    Returns:
        Dictionary with keys: claim_id, classification, confidence, reasoning
    '''
    pass
```

The function should use the framework's LLM capabilities to analyze the claim
and return a structured classification result.
"""

    def _get_dev_tests(self) -> list[TestCase]:
        """Development test set (5 cases) - shown during error feedback."""
        return [
            TestCase(
                case_id="dev_001",
                input_data={
                    "claim_id": "CLM-001",
                    "description": "Minor fender bender in parking lot. My car was hit while parked. Other driver admitted fault.",
                    "damage_estimate": 1200,
                    "parties_involved": 2,
                    "has_injury": False,
                    "liability_clear": True,
                    "vehicle_type": "personal",
                },
                expected_output={"classification": "simple"},
                description="Clear liability, low damage, no injury - should be simple",
            ),
            TestCase(
                case_id="dev_002",
                input_data={
                    "claim_id": "CLM-002",
                    "description": "Multi-vehicle pileup on highway. Three cars involved, one driver taken to hospital with neck pain.",
                    "damage_estimate": 25000,
                    "parties_involved": 3,
                    "has_injury": True,
                    "liability_clear": False,
                    "vehicle_type": "personal",
                },
                expected_output={"classification": "complex"},
                description="Multiple parties, injury, unclear liability - should be complex",
            ),
            TestCase(
                case_id="dev_003",
                input_data={
                    "claim_id": "CLM-003",
                    "description": "Delivery truck backed into storefront causing significant structural damage.",
                    "damage_estimate": 45000,
                    "parties_involved": 2,
                    "has_injury": False,
                    "liability_clear": True,
                    "vehicle_type": "commercial",
                },
                expected_output={"classification": "complex"},
                description="Commercial vehicle, high damage - should be complex",
            ),
            TestCase(
                case_id="dev_004",
                input_data={
                    "claim_id": "CLM-004",
                    "description": "Shopping cart rolled into my car door in supermarket parking lot.",
                    "damage_estimate": 450,
                    "parties_involved": 1,
                    "has_injury": False,
                    "liability_clear": True,
                    "vehicle_type": "personal",
                },
                expected_output={"classification": "simple"},
                description="Single party, very low damage - should be simple",
            ),
            TestCase(
                case_id="dev_005",
                input_data={
                    "claim_id": "CLM-005",
                    "description": "Two cars collided at intersection. Both drivers claim the other ran the red light. Dashcam footage is unclear.",
                    "damage_estimate": 8000,
                    "parties_involved": 2,
                    "has_injury": False,
                    "liability_clear": False,
                    "vehicle_type": "personal",
                },
                expected_output={"classification": "complex"},
                description="Disputed liability - should be complex",
            ),
        ]

    def _get_hidden_tests(self) -> list[TestCase]:
        """Hidden test set (10+ cases) - used for final scoring only."""
        return [
            TestCase(
                case_id="hidden_001",
                input_data={
                    "claim_id": "CLM-H01",
                    "description": "Windshield cracked by flying debris on highway.",
                    "damage_estimate": 800,
                    "parties_involved": 1,
                    "has_injury": False,
                    "liability_clear": True,
                    "vehicle_type": "personal",
                },
                expected_output={"classification": "simple"},
            ),
            TestCase(
                case_id="hidden_002",
                input_data={
                    "claim_id": "CLM-H02",
                    "description": "Cyclist hit by car turning right. Cyclist has broken arm and bicycle is destroyed.",
                    "damage_estimate": 15000,
                    "parties_involved": 2,
                    "has_injury": True,
                    "liability_clear": True,
                    "vehicle_type": "personal",
                },
                expected_output={"classification": "complex"},
            ),
            TestCase(
                case_id="hidden_003",
                input_data={
                    "claim_id": "CLM-H03",
                    "description": "Tree branch fell on parked car during storm.",
                    "damage_estimate": 3500,
                    "parties_involved": 1,
                    "has_injury": False,
                    "liability_clear": True,
                    "vehicle_type": "personal",
                },
                expected_output={"classification": "simple"},
            ),
            TestCase(
                case_id="hidden_004",
                input_data={
                    "claim_id": "CLM-H04",
                    "description": "Bus collided with pedestrian at crosswalk. Pedestrian hospitalized.",
                    "damage_estimate": 75000,
                    "parties_involved": 2,
                    "has_injury": True,
                    "liability_clear": False,
                    "vehicle_type": "commercial",
                },
                expected_output={"classification": "complex"},
            ),
            TestCase(
                case_id="hidden_005",
                input_data={
                    "claim_id": "CLM-H05",
                    "description": "Rear-ended at stoplight. Clear fault by following driver.",
                    "damage_estimate": 2800,
                    "parties_involved": 2,
                    "has_injury": False,
                    "liability_clear": True,
                    "vehicle_type": "personal",
                },
                expected_output={"classification": "simple"},
            ),
            TestCase(
                case_id="hidden_006",
                input_data={
                    "claim_id": "CLM-H06",
                    "description": "Four-car chain reaction collision on bridge. Multiple injuries reported.",
                    "damage_estimate": 85000,
                    "parties_involved": 4,
                    "has_injury": True,
                    "liability_clear": False,
                    "vehicle_type": "personal",
                },
                expected_output={"classification": "complex"},
            ),
            TestCase(
                case_id="hidden_007",
                input_data={
                    "claim_id": "CLM-H07",
                    "description": "Sideswiped parked car while parallel parking. Left note with insurance info.",
                    "damage_estimate": 1500,
                    "parties_involved": 2,
                    "has_injury": False,
                    "liability_clear": True,
                    "vehicle_type": "personal",
                },
                expected_output={"classification": "simple"},
            ),
            TestCase(
                case_id="hidden_008",
                input_data={
                    "claim_id": "CLM-H08",
                    "description": "Semi-truck jackknifed on icy road, hitting guardrail and blocking traffic. No other vehicles involved.",
                    "damage_estimate": 35000,
                    "parties_involved": 1,
                    "has_injury": False,
                    "liability_clear": True,
                    "vehicle_type": "commercial",
                },
                expected_output={"classification": "complex"},
            ),
            TestCase(
                case_id="hidden_009",
                input_data={
                    "claim_id": "CLM-H09",
                    "description": "Door ding in parking lot. No witnesses, unknown other party.",
                    "damage_estimate": 400,
                    "parties_involved": 1,
                    "has_injury": False,
                    "liability_clear": True,
                    "vehicle_type": "personal",
                },
                expected_output={"classification": "simple"},
            ),
            TestCase(
                case_id="hidden_010",
                input_data={
                    "claim_id": "CLM-H10",
                    "description": "Rideshare vehicle hit by drunk driver. Passenger claims whiplash. Rideshare company disputing coverage.",
                    "damage_estimate": 22000,
                    "parties_involved": 3,
                    "has_injury": True,
                    "liability_clear": False,
                    "vehicle_type": "commercial",
                },
                expected_output={"classification": "complex"},
            ),
            TestCase(
                case_id="hidden_011",
                input_data={
                    "claim_id": "CLM-H11",
                    "description": "Hail damage to vehicle roof and hood.",
                    "damage_estimate": 4200,
                    "parties_involved": 1,
                    "has_injury": False,
                    "liability_clear": True,
                    "vehicle_type": "personal",
                },
                expected_output={"classification": "simple"},
            ),
            TestCase(
                case_id="hidden_012",
                input_data={
                    "claim_id": "CLM-H12",
                    "description": "Hit and run in residential area. Security camera footage being reviewed by police.",
                    "damage_estimate": 6500,
                    "parties_involved": 2,
                    "has_injury": False,
                    "liability_clear": False,
                    "vehicle_type": "personal",
                },
                expected_output={"classification": "complex"},
            ),
        ]

    def _generate_single_test(self, test: TestCase) -> str:
        """Generate assertion code for a single test case."""
        input_repr = repr(test.input_data)
        expected = test.expected_output["classification"]
        return f"""result = classify_claim({input_repr})
assert 'classification' in result, "Result must have 'classification' key"
assert 'confidence' in result, "Result must have 'confidence' key"
assert 'reasoning' in result, "Result must have 'reasoning' key"
assert result['classification'] in ['simple', 'complex'], f"Invalid classification: {{result['classification']}}"
assert result['classification'] == '{expected}', f"Expected '{expected}', got '{{result['classification']}}' for {test.case_id}"
assert 0 <= result['confidence'] <= 1, f"Confidence must be between 0 and 1, got {{result['confidence']}}"
"""
