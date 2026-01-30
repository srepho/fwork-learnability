# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research project for measuring **LLM-assisted framework usability** - how easily an LLM can produce working code with various agent frameworks given access to documentation.

**GitHub Repo**: https://github.com/srepho/fwork-learnability

## Current Status: Phase 2 In Progress

### What Was Accomplished in Phase 1

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

3. **Initial Experiments** ✅
   - 48 trials across 4 frameworks (PydanticAI, Haystack, LangGraph, Direct API)
   - High contamination detected (all frameworks succeeded at "none" doc level)

### Phase 2 Updates (Current Session)

1. **Fixed Documentation Fetching** ✅
   - Added curated documentation with real code examples
   - Haystack: 159-325 tokens of actual code (was 26 lines of marketing)
   - LangGraph: 191-322 tokens of actual code (was 18 tokens)
   - New `--prefer-curated` flag in `scripts/fetch_docs.py`

2. **Contamination Detection** ✅
   - New `src/analysis/contamination.py` module
   - Version alignment analysis (detects if model uses old vs new APIs)
   - Doc contradiction test (renamed APIs to detect training data usage)
   - Code fingerprinting across doc levels
   - CLI: `python scripts/analyze_contamination.py --results-dir results/exp_XXX`

3. **New Frameworks Added** ✅
   - `anthropic-agents` - Anthropic Claude API with tools
   - `openai-agents` - OpenAI API with function calling
   - Both have curated documentation with code examples

4. **Doc Contradiction Test** ✅
   - Provides modified docs with renamed APIs
   - If model uses REAL names despite modified docs → using training data
   - Run with: `--contradiction-test` flag

## Running Experiments

```bash
# Activate environment
source .venv/bin/activate

# Setup venvs for frameworks
python scripts/setup_venvs.py --frameworks haystack langgraph anthropic-agents openai-agents

# Fetch curated docs
python scripts/fetch_docs.py --frameworks haystack langgraph anthropic-agents openai-agents --prefer-curated

# Run experiment with contradiction test
python scripts/run_experiment.py \
  --frameworks haystack langgraph anthropic-agents openai-agents \
  --tiers 1 \
  --runs 3 \
  --contradiction-test

# Analyze contamination
python scripts/analyze_contamination.py --results-dir results/exp_XXXXXXXX_XXXXXX
```

## Contamination Analysis

From initial experiments, DeepSeek V3 shows high contamination:

| Framework | Success@None | API Version | Contamination |
|-----------|-------------|-------------|---------------|
| PydanticAI | 100% | - | HIGH |
| LangGraph | 100% | v0/v1 mixed | HIGH |
| Haystack | 67% | v2 (modern) | HIGH |
| Direct API | 67% | N/A | Expected |

Key finding: Haystack uses modern v2 APIs despite being "old" - DeepSeek was trained on recent Haystack content.

## Project Structure

```
fwork_learnability/
├── src/
│   ├── harness/           # runner.py, conversation.py, sandbox.py, extractor.py
│   ├── docs/              # corpus.py, normalizer.py
│   ├── compliance/        # static.py, dynamic.py
│   ├── analysis/          # errors.py, hallucination.py, metrics.py, survival.py, contamination.py
│   ├── llm/               # base.py, deepseek.py
│   └── tasks/             # tier1/, tier2/, tier3/
├── tests/                 # Unit tests
├── results/               # Experiment JSON logs
├── visualizations/        # Generated charts
├── docs_corpus/           # Snapshotted documentation
├── scripts/               # CLI utilities
│   ├── run_experiment.py  # Main experiment runner
│   ├── fetch_docs.py      # Doc fetching with curated support
│   ├── setup_venvs.py     # Framework venv setup
│   └── analyze_contamination.py  # Contamination analysis
└── config/                # frameworks.yaml
```

## Key Files for Common Tasks

| Task | Files |
|------|-------|
| Add new framework | `config/frameworks.yaml`, `scripts/setup_venvs.py`, `scripts/fetch_docs.py` |
| Add curated docs | `scripts/fetch_docs.py` (CURATED_DOCS dict) |
| Analyze contamination | `scripts/analyze_contamination.py` |
| Run contradiction test | `scripts/run_experiment.py --contradiction-test` |
| Modify conversation loop | `src/harness/conversation.py` |

## API Keys Required

- `DEEPSEEK_API_KEY` - Primary model for experiments (required)
- `ANTHROPIC_API_KEY` - For anthropic-agents framework testing
- `OPENAI_API_KEY` - For openai-agents framework testing

## Next Steps

1. **Analyze contradiction test results** - See if model follows modified docs or uses training data
2. **Run Tier 2/3 tasks** - Test tool use and agent-native capabilities
3. **Test with older models** - For cleaner contamination control
4. **Add more post-cutoff frameworks** - Frameworks released after mid-2024
