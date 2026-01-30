"""Insurance claims tool use task (Tier 2).

This task tests tool definition, schema handling, and result parsing.
The LLM must use the framework to build an agent that uses tools to classify claims.
"""

from ..base import Task, TaskTier, TestCase


class ClaimsToolUseTask(Task):
    """Classify insurance claims using tools.

    The agent must use provided tools to:
    1. Look up policy details
    2. Check claim history
    3. Assess damage estimates
    4. Make classification decision

    This tests framework's tool use capabilities.
    """

    def __init__(self):
        super().__init__(
            task_id="tier2_claims_tools",
            name="Insurance Claims Tool Use",
            tier=TaskTier.TIER_2,
            description=self._get_description(),
            expected_interface=self._get_interface(),
            development_tests=self._get_dev_tests(),
            hidden_tests=self._get_hidden_tests(),
        )

    def _get_description(self) -> str:
        return """Create a claims classification agent that uses tools to gather information before making decisions.

**Available Tools:**

1. `get_policy_details(policy_id: str) -> dict`
   - Returns: coverage_type, deductible, max_payout, special_conditions

2. `check_claim_history(customer_id: str) -> dict`
   - Returns: total_claims, recent_claims_count, fraud_flags

3. `get_damage_assessment(claim_id: str) -> dict`
   - Returns: estimated_cost, repair_complexity, parts_availability

**Classification Logic:**

Use the tools to gather information, then classify as 'simple' or 'complex':

SIMPLE if ALL:
- Damage estimate under deductible × 10
- No fraud flags in history
- Standard coverage type
- Repair complexity is 'low' or 'medium'

COMPLEX if ANY:
- Damage estimate exceeds deductible × 10
- Fraud flags present
- Special coverage conditions apply
- Repair complexity is 'high'
- More than 3 recent claims

**Input:** A claim dictionary with:
- claim_id, policy_id, customer_id, description

**Output:** A result with:
- claim_id, classification, tools_used, reasoning
"""

    def _get_interface(self) -> str:
        return """Your solution must provide:

1. Tool definitions that match the signatures above (can be stubs for testing)
2. An agent that uses these tools
3. A `classify_claim_with_tools` function:

```python
def classify_claim_with_tools(claim: dict) -> dict:
    '''
    Classify a claim using tool-gathered information.

    Args:
        claim: Dict with claim_id, policy_id, customer_id, description

    Returns:
        Dict with claim_id, classification, tools_used (list), reasoning
    '''
    pass
```

The function should orchestrate tool calls and return a structured result.
"""

    def _get_dev_tests(self) -> list[TestCase]:
        return [
            TestCase(
                case_id="dev_t2_001",
                input_data={
                    "claim_id": "CLM-T2-001",
                    "policy_id": "POL-001",
                    "customer_id": "CUST-001",
                    "description": "Minor scratch on door",
                },
                expected_output={
                    "classification": "simple",
                    "min_tools": 2,
                },
                description="Simple case - low damage, clean history",
            ),
            TestCase(
                case_id="dev_t2_002",
                input_data={
                    "claim_id": "CLM-T2-002",
                    "policy_id": "POL-002",
                    "customer_id": "CUST-002",
                    "description": "Major accident with multiple issues",
                },
                expected_output={
                    "classification": "complex",
                    "min_tools": 2,
                },
                description="Complex case - high damage, fraud flags",
            ),
            TestCase(
                case_id="dev_t2_003",
                input_data={
                    "claim_id": "CLM-T2-003",
                    "policy_id": "POL-003",
                    "customer_id": "CUST-003",
                    "description": "Windshield replacement needed",
                },
                expected_output={
                    "classification": "simple",
                    "min_tools": 2,
                },
                description="Simple case - standard repair",
            ),
            TestCase(
                case_id="dev_t2_004",
                input_data={
                    "claim_id": "CLM-T2-004",
                    "policy_id": "POL-004",
                    "customer_id": "CUST-004",
                    "description": "Total loss claim",
                },
                expected_output={
                    "classification": "complex",
                    "min_tools": 2,
                },
                description="Complex case - exceeds policy limits",
            ),
            TestCase(
                case_id="dev_t2_005",
                input_data={
                    "claim_id": "CLM-T2-005",
                    "policy_id": "POL-005",
                    "customer_id": "CUST-005",
                    "description": "Theft of personal items",
                },
                expected_output={
                    "classification": "complex",
                    "min_tools": 2,
                },
                description="Complex case - special coverage conditions",
            ),
        ]

    def _get_hidden_tests(self) -> list[TestCase]:
        return [
            TestCase(
                case_id="hidden_t2_001",
                input_data={
                    "claim_id": "CLM-T2-H01",
                    "policy_id": "POL-H01",
                    "customer_id": "CUST-H01",
                    "description": "Parking lot ding",
                },
                expected_output={"classification": "simple", "min_tools": 2},
            ),
            TestCase(
                case_id="hidden_t2_002",
                input_data={
                    "claim_id": "CLM-T2-H02",
                    "policy_id": "POL-H02",
                    "customer_id": "CUST-H02",
                    "description": "Repeated claim - same issue as last month",
                },
                expected_output={"classification": "complex", "min_tools": 2},
            ),
            # Add more hidden tests...
        ] * 5  # Repeat to get 10 tests

    def _generate_single_test(self, test: TestCase) -> str:
        input_repr = repr(test.input_data)
        expected = test.expected_output["classification"]
        min_tools = test.expected_output.get("min_tools", 1)
        return f"""result = classify_claim_with_tools({input_repr})
assert 'classification' in result, "Result must have 'classification' key"
assert 'tools_used' in result, "Result must have 'tools_used' key"
assert 'reasoning' in result, "Result must have 'reasoning' key"
assert result['classification'] in ['simple', 'complex'], f"Invalid classification: {{result['classification']}}"
assert result['classification'] == '{expected}', f"Expected '{expected}', got '{{result['classification']}}' for {test.case_id}"
assert len(result['tools_used']) >= {min_tools}, f"Expected at least {min_tools} tools used, got {{len(result['tools_used'])}}"
"""
