# LLM Framework Learnability Benchmark

Measuring how easily an LLM can produce working code with various agent frameworks, given access to documentation.

## Overview

This benchmark measures **LLM-assisted framework usability** using a "Turns to Working Code" methodology. An LLM attempts to implement tasks using different frameworks, with error feedback guiding corrections up to 10 turns.

### Key Features

- **Temporal Firewall**: Uses DeepSeek V3 (training cutoff mid-2024) to measure true learnability vs memorization
- **Three-Tier Tasks**: Classification (Tier 1), Tool Use (Tier 2), Agent-Native (Tier 3)
- **Four Documentation Levels**: None, Minimal, Moderate, Full
- **Hidden Test Sets**: Development tests for feedback, hidden tests for final scoring
- **Survival Analysis**: Kaplan-Meier curves for turns-to-success

## Experiment Results

We ran **48 trials** across 4 frameworks testing Tier 1 (Claims Classification):

| Framework | Success Rate | Avg Turns | First Attempt | Best Doc Level |
|-----------|-------------|-----------|---------------|----------------|
| **PydanticAI** | 92% (11/12) | 3.4 | 3 | moderate |
| **Direct API** | 83% (10/12) | 2.4 | 1 | moderate |
| **LangGraph** | 83% (10/12) | 4.7 | 0 | minimal |
| **Haystack** | 75% (9/12) | 3.0 | 3 | minimal |

### Key Findings

1. **High Contamination**: All frameworks succeeded 2-3/3 with NO documentation - DeepSeek V3 likely has these frameworks in training data

2. **Minimal Docs Often Best**: Haystack and LangGraph performed better with minimal docs than full docs!

3. **PydanticAI Most Reliable**: Highest success rate (92%) overall

### Error Analysis (8/48 failures = 17%)

| Error Type | Count | Description |
|------------|-------|-------------|
| SyntaxError Loop | 40 | Model generates unterminated strings, fails to recover |
| API Hallucination | 26 | PydanticAI `result_type` (15x), wrong method signatures |
| Logic Error | 15 | Wrong classification despite correct code structure |

## Quick Start

```bash
# Create environment
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Copy and configure API keys
cp .env.example .env
# Edit .env with your DEEPSEEK_API_KEY

# Setup framework venvs
python scripts/setup_venvs.py --frameworks pydantic-ai direct-api

# Fetch documentation
python scripts/fetch_docs.py --frameworks pydantic-ai

# Run experiment (dry run first)
python scripts/run_experiment.py --frameworks pydantic-ai --tiers 1 --runs 3 --dry-run

# Run actual experiment
python scripts/run_experiment.py --frameworks pydantic-ai --tiers 1 --runs 3
```

## Example: Single Trial Output

```json
{
  "trial_id": "6b1fa682",
  "framework": "pydantic-ai",
  "doc_level": "none",
  "outcome": "success",
  "final_turn": 4,
  "turns": [
    {"turn": 1, "error_type": "UserError", "error_message": "Unknown keyword arguments: `result_type`"},
    {"turn": 2, "error_type": "TypeError", "error_message": "..."},
    {"turn": 3, "error_type": "AssertionError", "error_message": "Expected 'simple', got 'complex'"},
    {"turn": 4, "outcome": "success"}
  ]
}
```

## Project Structure

```
fwork_learnability/
├── src/
│   ├── harness/           # Core test harness
│   │   ├── runner.py      # Experiment runner
│   │   ├── conversation.py # "Fresh Start with Error" loop
│   │   ├── sandbox.py     # Code execution sandbox
│   │   └── extractor.py   # Code extraction from LLM
│   ├── docs/              # Documentation management
│   ├── compliance/        # Framework compliance checking
│   ├── analysis/          # Metrics & error analysis
│   ├── llm/               # DeepSeek V3 client
│   └── tasks/             # Task definitions (tier1, tier2, tier3)
├── tests/                 # Unit tests
├── results/               # Experiment results (JSON logs)
├── visualizations/        # Generated charts
├── docs_corpus/           # Snapshotted documentation
├── config/                # Framework configuration
└── scripts/               # CLI utilities
```

## Known Issues & Future Work

### Documentation Fetching Issues

The current doc fetcher has issues that need to be addressed:

1. **Landing Pages vs Tutorials**: Fetching captures marketing content instead of actual code examples
   - Haystack "full" docs are 26 lines of marketing copy with no code
   - This explains why minimal docs (9 lines) outperformed full docs

2. **Missing Code Examples**: Full documentation should include:
   - Quickstart with working code
   - API reference with signatures
   - Tool use examples for Tier 2 tasks

3. **404 Errors**: Some LangGraph doc URLs return 404 (site structure changed)

### Recommended Fixes

```python
# TODO: Improve doc fetching to get actual content
# 1. Use framework-specific selectors for content extraction
# 2. Fetch quickstart tutorials, not landing pages
# 3. Include minimal working code examples
# 4. Add API signature reference for key classes
```

### Other Future Work

- [ ] Improve syntax error recovery in feedback loop
- [ ] Track code similarity to detect "stuck" loops
- [ ] Add Tier 2 (tool use) and Tier 3 (agent) tasks
- [ ] Test with older models for cleaner temporal firewall
- [ ] Add more frameworks (Anthropic Agents SDK, etc.)

## Methodology

See [llm-learnability-methodology.md](llm-learnability-methodology.md) for the full research methodology.

### The "Turns to Working Code" Loop

```
1. Initialize with task description + framework docs
2. Loop (max 10 turns):
   a. LLM generates implementation
   b. Run against test suite
   c. If pass → record success
   d. If fail → feed error back (Fresh Start with Error)
3. Record metrics
```

### Context Management: "Fresh Start with Error"

Each turn sees only:
- Original task description
- Framework documentation
- Code from turn N-1
- Error from turn N-1

This isolates each correction attempt.

## Metrics

| Metric | Description |
|--------|-------------|
| Turns to Success | Survival analysis (Kaplan-Meier) |
| Success Rate | % that eventually succeed |
| First-Attempt Success | % that work on turn 1 |
| Meaningful Error Rate | % of early errors with actionable messages |
| Hallucination Rate | Invented vs Version Conflict APIs |

## Cost

With DeepSeek V3's competitive pricing, experiments cost approximately **$2-4** for 720 trials.

## License

MIT
