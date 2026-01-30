"""Insurance claims agent task (Tier 3).

This task tests agent-native capabilities: routing, state management, and multi-step reasoning.
The LLM must build a full agent system that handles complex claim workflows.
"""

from ..base import Task, TaskTier, TestCase


class ClaimsAgentTask(Task):
    """Build a complete claims processing agent.

    The agent must:
    1. Route claims to appropriate handlers based on type
    2. Maintain state across multiple processing steps
    3. Handle multi-step workflows with conditional logic
    4. Aggregate results from multiple sub-processes

    This tests where agent frameworks should excel.
    """

    def __init__(self):
        super().__init__(
            task_id="tier3_claims_agent",
            name="Insurance Claims Agent",
            tier=TaskTier.TIER_3,
            description=self._get_description(),
            expected_interface=self._get_interface(),
            development_tests=self._get_dev_tests(),
            hidden_tests=self._get_hidden_tests(),
        )

    def _get_description(self) -> str:
        return """Build a complete claims processing agent with routing, state, and multi-step workflows.

**Agent Architecture:**

1. **Intake Router** - Routes claim to appropriate handler:
   - auto_damage -> Auto Damage Handler
   - property -> Property Handler
   - injury -> Injury Handler (requires additional steps)

2. **Handlers** - Each handler has specific logic:
   - Auto: Check damage, verify coverage, calculate payout
   - Property: Assess damage, check exclusions, estimate repair
   - Injury: Medical review, liability assessment, negotiate settlement

3. **State Management** - Track across steps:
   - current_step: which step we're on
   - gathered_info: information collected
   - decisions_made: list of decisions
   - requires_escalation: boolean flag

4. **Multi-step Workflow:**
   - Step 1: Initial classification and routing
   - Step 2: Gather required information (tool calls)
   - Step 3: Apply business rules
   - Step 4: Calculate outcome
   - Step 5: Generate recommendation

**Input:** A claim with:
- claim_id, claim_type, description, amount, supporting_docs

**Output:** An agent result with:
- claim_id
- final_classification: 'approved', 'denied', 'escalate'
- workflow_steps: list of steps executed
- state_history: list of state snapshots
- total_processing_steps: int
- reasoning: explanation
"""

    def _get_interface(self) -> str:
        return """Your solution must provide:

1. A stateful agent class or function
2. Routing logic for different claim types
3. A `process_claim` function:

```python
def process_claim(claim: dict) -> dict:
    '''
    Process a claim through the full agent workflow.

    Args:
        claim: Dict with claim_id, claim_type, description, amount, supporting_docs

    Returns:
        Dict with:
        - claim_id
        - final_classification ('approved', 'denied', 'escalate')
        - workflow_steps (list of step names executed)
        - state_history (list of state dicts at each step)
        - total_processing_steps (int)
        - reasoning (str)
    '''
    pass
```

The agent should demonstrate:
- Conditional routing based on claim_type
- State persistence across steps
- Multi-step decision making
"""

    def _get_dev_tests(self) -> list[TestCase]:
        return [
            TestCase(
                case_id="dev_t3_001",
                input_data={
                    "claim_id": "CLM-T3-001",
                    "claim_type": "auto_damage",
                    "description": "Fender bender, minor damage",
                    "amount": 1500,
                    "supporting_docs": ["photos", "police_report"],
                },
                expected_output={
                    "final_classification": "approved",
                    "min_steps": 3,
                },
                description="Simple auto claim - should be approved",
            ),
            TestCase(
                case_id="dev_t3_002",
                input_data={
                    "claim_id": "CLM-T3-002",
                    "claim_type": "injury",
                    "description": "Whiplash from rear-end collision",
                    "amount": 25000,
                    "supporting_docs": ["medical_records", "police_report", "witness_statements"],
                },
                expected_output={
                    "final_classification": "escalate",
                    "min_steps": 4,
                },
                description="Injury claim - should escalate for review",
            ),
            TestCase(
                case_id="dev_t3_003",
                input_data={
                    "claim_id": "CLM-T3-003",
                    "claim_type": "property",
                    "description": "Water damage from burst pipe",
                    "amount": 8000,
                    "supporting_docs": ["photos", "plumber_report"],
                },
                expected_output={
                    "final_classification": "approved",
                    "min_steps": 3,
                },
                description="Property claim - standard approval flow",
            ),
            TestCase(
                case_id="dev_t3_004",
                input_data={
                    "claim_id": "CLM-T3-004",
                    "claim_type": "auto_damage",
                    "description": "Suspicious total loss claim",
                    "amount": 50000,
                    "supporting_docs": ["photos"],
                },
                expected_output={
                    "final_classification": "escalate",
                    "min_steps": 4,
                },
                description="Suspicious claim - should escalate",
            ),
            TestCase(
                case_id="dev_t3_005",
                input_data={
                    "claim_id": "CLM-T3-005",
                    "claim_type": "property",
                    "description": "Flood damage - excluded peril",
                    "amount": 15000,
                    "supporting_docs": ["photos", "adjuster_report"],
                },
                expected_output={
                    "final_classification": "denied",
                    "min_steps": 3,
                },
                description="Excluded peril - should be denied",
            ),
        ]

    def _get_hidden_tests(self) -> list[TestCase]:
        return [
            TestCase(
                case_id="hidden_t3_001",
                input_data={
                    "claim_id": "CLM-T3-H01",
                    "claim_type": "auto_damage",
                    "description": "Windshield crack",
                    "amount": 500,
                    "supporting_docs": ["photos"],
                },
                expected_output={"final_classification": "approved", "min_steps": 3},
            ),
            TestCase(
                case_id="hidden_t3_002",
                input_data={
                    "claim_id": "CLM-T3-H02",
                    "claim_type": "injury",
                    "description": "Broken arm from slip and fall",
                    "amount": 75000,
                    "supporting_docs": ["medical_records", "incident_report"],
                },
                expected_output={"final_classification": "escalate", "min_steps": 4},
            ),
            # Add more varied test cases...
        ] * 5

    def _generate_single_test(self, test: TestCase) -> str:
        input_repr = repr(test.input_data)
        expected_class = test.expected_output["final_classification"]
        min_steps = test.expected_output.get("min_steps", 1)
        return f"""result = process_claim({input_repr})
assert 'final_classification' in result, "Result must have 'final_classification' key"
assert 'workflow_steps' in result, "Result must have 'workflow_steps' key"
assert 'state_history' in result, "Result must have 'state_history' key"
assert 'total_processing_steps' in result, "Result must have 'total_processing_steps' key"
assert 'reasoning' in result, "Result must have 'reasoning' key"
assert result['final_classification'] in ['approved', 'denied', 'escalate'], f"Invalid classification: {{result['final_classification']}}"
assert result['final_classification'] == '{expected_class}', f"Expected '{expected_class}', got '{{result['final_classification']}}' for {test.case_id}"
assert result['total_processing_steps'] >= {min_steps}, f"Expected at least {min_steps} steps, got {{result['total_processing_steps']}}"
assert len(result['workflow_steps']) >= {min_steps}, f"Expected at least {min_steps} workflow steps"
assert len(result['state_history']) >= 1, "Expected state history to be recorded"
"""
