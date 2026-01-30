# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project for measuring **LLM-assisted framework usability** - how easily an LLM can produce working code with various agent frameworks given access to documentation.

**GitHub Repo**: https://github.com/srepho/fwork-learnability

## Current Status: Phase 1 Complete

### What Was Accomplished

1. **Core Infrastructure** ✅
   - LLM client (DeepSeek V3 with temperature=0)
   - Code extraction from markdown responses
   - Sandbox execution (subprocess-based isolation)
   - Conversation loop with "Fresh Start with Error" context management
   - Compliance checking (static + dynamic)
   - Error analysis and hallucination detection
   - Survival analysis (Kaplan-Meier)

2. **Task Definitions** ✅
   - Tier 1: Claims Classification (5 dev tests, 12 hidden tests)
   - Tier 2: Claims Tool Use (defined, not tested)
   - Tier 3: Claims Agent (defined, not tested)

3. **Experiments Run** ✅
   - 48 trials across 4 frameworks (PydanticAI, Haystack, LangGraph, Direct API)
   - 4 documentation levels (none, minimal, moderate, full)
   - 3 runs per condition

### Experiment Results Summary

| Framework | Success Rate | Avg Turns | Best Doc Level |
|-----------|-------------|-----------|----------------|
| PydanticAI | 92% (11/12) | 3.4 | moderate |
| Direct API | 83% (10/12) | 2.4 | moderate |
| LangGraph | 83% (10/12) | 4.7 | minimal |
| Haystack | 75% (9/12) | 3.0 | minimal |

### Key Findings

1. **High Contamination**: All frameworks succeeded at "none" doc level (2-3/3) - DeepSeek V3 has these frameworks in training data

2. **Documentation Paradox**: Minimal docs often outperformed full docs
   - Haystack: minimal (100%) > full (33%)
   - Root cause: Doc fetching captured landing pages, not tutorials

3. **Error Patterns** (8/48 failures):
   - SyntaxError loop (40 occurrences): Model gets stuck on unterminated strings
   - API hallucinations (26): PydanticAI `result_type` doesn't exist
   - Logic errors (15): Wrong classification despite working code

## Known Issues to Fix

### 1. Documentation Fetching (HIGH PRIORITY)
The current doc fetcher gets landing pages instead of actual tutorials:
- Haystack "full" docs = 26 lines of marketing, no code
- Need to fetch actual quickstart pages with code examples
- Consider manual curation of doc snippets

**Location**: `scripts/fetch_docs.py`, `src/docs/`

### 2. Syntax Error Recovery
Model gets stuck in SyntaxError loops (10 turns of same error):
- Need better error feedback showing the specific problematic line
- Consider detecting when model is "stuck" making same mistake

**Location**: `src/harness/conversation.py`

### 3. Contamination Control
DeepSeek V3 (Dec 2024 cutoff) knows these frameworks:
- Consider using an older model
- Or focus on frameworks released after Dec 2024

## Project Structure

```
fwork_learnability/
├── src/
│   ├── harness/           # runner.py, conversation.py, sandbox.py, extractor.py
│   ├── docs/              # corpus.py, normalizer.py
│   ├── compliance/        # static.py, dynamic.py
│   ├── analysis/          # errors.py, hallucination.py, metrics.py, survival.py
│   ├── llm/               # base.py, deepseek.py
│   └── tasks/             # tier1/, tier2/, tier3/
├── tests/                 # Unit tests
├── results/               # Experiment JSON logs
├── visualizations/        # Generated charts
├── docs_corpus/           # Snapshotted documentation
├── scripts/               # CLI utilities
└── config/                # frameworks.yaml
```

## Environment Setup

```bash
# Create venv and install
python -m venv .venv
source .venv/bin/activate
pip install -e .

# Configure API key
cp .env.example .env
# Edit .env with DEEPSEEK_API_KEY

# Setup framework venvs
python scripts/setup_venvs.py --frameworks pydantic-ai haystack langgraph direct-api

# Run experiment
python scripts/run_experiment.py --frameworks pydantic-ai --tiers 1 --runs 3
```

## Next Steps (Priority Order)

1. **Fix doc fetching** - Get actual code examples, not landing pages
2. **Run Tier 2/3 tasks** - Test tool use and agent-native capabilities
3. **Improve error recovery** - Better syntax error feedback
4. **Add more frameworks** - Anthropic Agents SDK, OpenAI Agents SDK
5. **Statistical analysis** - More runs for significance testing

## Key Files for Common Tasks

| Task | Files |
|------|-------|
| Add new framework | `config/frameworks.yaml`, `src/compliance/static.py` |
| Add new task | `src/tasks/tierN/` |
| Modify conversation loop | `src/harness/conversation.py` |
| Improve doc fetching | `scripts/fetch_docs.py`, `src/docs/corpus.py` |
| Analyze results | `scripts/visualize.py`, `src/analysis/` |

## API Keys Required

- `DEEPSEEK_API_KEY` - Primary model for experiments (required)
- Other keys in `.env` are optional for future framework testing
