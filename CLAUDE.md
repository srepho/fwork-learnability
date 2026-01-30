# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project for measuring **LLM-assisted framework usability** - how easily an LLM can produce working code with various agent frameworks given access to documentation. The methodology benchmarks frameworks including PydanticAI, Haystack, LangGraph, OpenAI Agents SDK, Anthropic Agents SDK, Microsoft Agent Framework, and Direct API baseline.

## Implementation Status

**Phase 1: Core Infrastructure ✅ COMPLETE**
- LLM client layer (DeepSeek V3)
- Code extraction from markdown responses
- Sandbox execution (subprocess-based)
- Conversation loop with "Fresh Start with Error"

**Phase 2: Documentation & Compliance ✅ COMPLETE**
- Documentation corpus management
- Markdown normalization
- Static compliance checking (imports, symbols)
- Dynamic compliance checking (entrypoint verification)

**Phase 3: Tasks & Testing ✅ COMPLETE**
- Tier 1: Claims Classification (5 dev, 12 hidden tests)
- Tier 2: Claims Tool Use (5 dev tests)
- Tier 3: Claims Agent (5 dev tests)

**Phase 4: Analysis & Metrics ✅ COMPLETE**
- Error categorization (7 categories)
- Hallucination detection (invented vs version conflict)
- Metrics computation
- Survival analysis (Kaplan-Meier)

**Phase 5: Integration ✅ COMPLETE**
- Trial logging (JSON)
- Experiment runner with progress tracking
- Visualization scripts

## Current Phase: Pilot Complete, Expanding

### Experiment Results (48 trials across 4 frameworks)
- **Model**: DeepSeek V3 (`deepseek-chat`)
- **Task**: Tier 1 Claims Classification

| Framework | Success Rate | Avg Turns | First Attempt | Best Doc Level |
|-----------|-------------|-----------|---------------|----------------|
| PydanticAI | 92% (11/12) | 3.4 | 3 | moderate |
| Direct API | 83% (10/12) | 2.4 | 1 | moderate |
| LangGraph | 83% (10/12) | 4.7 | 0 | minimal |
| Haystack | 75% (9/12) | 3.0 | 3 | minimal |

**Key Findings:**
1. **High Contamination**: All frameworks succeeded 2-3/3 at doc level "none" - DeepSeek V3 likely has all frameworks in training data
2. **Minimal docs often best**: Haystack and LangGraph performed best with minimal docs, not full docs
3. **PydanticAI most reliable**: Highest success rate overall
4. **Direct API fastest**: Lowest average turns (2.4), but no framework overhead benefits

### Error Analysis (8/48 failures = 17%)

**Failure Distribution:**
| Framework | Failures | Main Errors |
|-----------|----------|-------------|
| Haystack | 3/12 | SyntaxError loop, AttributeError |
| Direct API | 2/12 | SyntaxError loop |
| LangGraph | 2/12 | ModuleNotFound → SyntaxError |
| PydanticAI | 1/12 | API hallucination (result_type) |

**Root Causes:**
1. **Syntax Error Loop** (most common): Model generates code with unterminated strings, then fails to recover for 10 turns
2. **API Hallucinations**: PydanticAI `result_type` (15x), Haystack `resolve_value` (8x)
3. **Full Docs Hurt**: Fetched landing pages, not tutorials - marketing content confuses model

### Next Steps
1. Improve doc fetching to get actual code examples
2. Add better syntax error recovery feedback
3. Run Tier 2/3 tasks
4. Consider older model for cleaner temporal firewall

## Core Methodology

The "Turns to Working Code" loop:
1. Initialize with task description + framework documentation
2. LLM generates implementation attempt
3. Run against test suite
4. If tests fail, feed error back (max 10 turns)
5. Record metrics (turns to success, hallucination rate, error types)

**Context Management:** Use "Fresh Start with Error" - each turn sees only: original task, docs, previous code, and previous error. Do not accumulate failed attempts.

## Key Design Decisions

- **Temporal Firewall:** Use DeepSeek V3 (training cutoff Dec 2024) to test frameworks released after this date
- **Three Task Tiers:** Tier 1 (classification), Tier 2 (tool use), Tier 3 (agent-native)
- **Hidden Test Sets:** Development set (5 cases) for error feedback, hidden set (10+) for final scoring
- **Framework Compliance Gates:** Static/dynamic checks to ensure LLM doesn't bypass the framework

## Project Structure

```
fwork_learnability/
├── src/
│   ├── harness/           # Core test harness (runner, conversation, sandbox, extractor)
│   ├── docs/              # Documentation management (normalizer, corpus)
│   ├── compliance/        # Framework compliance (static, dynamic)
│   ├── analysis/          # Metrics & analysis (errors, hallucination, metrics, survival)
│   ├── llm/               # LLM clients (DeepSeek V3)
│   └── tasks/             # Task definitions (tier1, tier2, tier3)
├── tests/                 # Unit tests
├── docs_corpus/           # Snapshotted documentation
├── results/               # Experiment results
├── config/                # Configuration (frameworks.yaml)
├── scripts/               # CLI utilities (run_experiment, fetch_docs, setup_venvs, visualize)
└── venvs/                 # Framework virtual environments
```

## Metrics

| Metric | Purpose |
|--------|---------|
| Turns to Success | Survival analysis (Kaplan-Meier) treating failures as censored |
| Success@None | Contamination detection - did it work without docs? |
| Symbol Exactness | Did it guess or memorize API calls? |
| Hallucination (Invented vs Version Conflict) | Unintuitive API vs breaking changes |
| Meaningful Error Rate | Actionable errors in turns 1-2 |
| Self-Correction Ratio | Does LLM adapt or repeat mistakes? |

## Documentation Levels

| Level | Content |
|-------|---------|
| None | Framework name only |
| Minimal | One-paragraph + install command |
| Moderate | Official quickstart + core concepts |
| Full | Above + tool use section + API reference |

## Error Categories

Import, Dependency, Type, Runtime, Logic, Hallucination, Syntax - track framework-specific exceptions separately from standard Python errors.

## Environment Configuration

**Use conda for Python environments.** Create and activate a project-specific environment:
```bash
conda create -n fwork_learn python=3.11
conda activate fwork_learn
pip install -e .
```

Required environment variable:
- `DEEPSEEK_API_KEY` - Primary model for experiments (DeepSeek V3)

## Running Commands

```bash
# Setup framework venvs
python scripts/setup_venvs.py --frameworks pydantic-ai direct-api

# Fetch documentation
python scripts/fetch_docs.py --frameworks pydantic-ai

# Run experiment
python scripts/run_experiment.py --frameworks pydantic-ai --tiers 1 --runs 5

# Visualize results
python scripts/visualize.py <experiment_id>

# Run tests
pytest tests/ -v
```
