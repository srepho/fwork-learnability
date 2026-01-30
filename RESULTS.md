# Experiment Results

## Experiment Details

- **Date**: January 2026
- **Model**: DeepSeek V3 (`deepseek-chat`)
- **Task**: Tier 1 - Insurance Claims Classification
- **Total Trials**: 48 (4 frameworks × 4 doc levels × 3 runs)

## Results by Framework and Documentation Level

### PydanticAI (11/12 success = 92%)

| Doc Level | Success | Avg Turns | Notes |
|-----------|---------|-----------|-------|
| none | 3/3 (100%) | 5.0 | High contamination signal |
| minimal | 2/3 (67%) | 5.5 | One failure: `result_type` hallucination loop |
| moderate | 3/3 (100%) | 1.7 | Best performance |
| full | 3/3 (100%) | 2.0 | Good performance |

**Key observation**: PydanticAI succeeded 100% with NO documentation, suggesting DeepSeek V3 has this framework in training data.

### Haystack (9/12 success = 75%)

| Doc Level | Success | Avg Turns | Notes |
|-----------|---------|-----------|-------|
| none | 2/3 (67%) | 3.5 | One max_turns failure |
| minimal | 3/3 (100%) | 1.0 | **All first-attempt successes!** |
| moderate | 3/3 (100%) | 4.7 | Slower than minimal |
| full | 1/3 (33%) | 3.0 | Two SyntaxError loops |

**Key observation**: Minimal docs (9 lines) dramatically outperformed full docs. The "full" docs contained marketing content, not code examples.

### LangGraph (10/12 success = 83%)

| Doc Level | Success | Avg Turns | Notes |
|-----------|---------|-----------|-------|
| none | 3/3 (100%) | 6.7 | High contamination |
| minimal | 3/3 (100%) | 4.0 | Good performance |
| moderate | 2/3 (67%) | 4.0 | One failure |
| full | 2/3 (67%) | 3.5 | ModuleNotFoundError issues |

**Key observation**: 100% success at "none" indicates strong contamination from training data.

### Direct API Baseline (10/12 success = 83%)

| Doc Level | Success | Avg Turns | Notes |
|-----------|---------|-----------|-------|
| none | 2/3 (67%) | 2.5 | Raw HTTP, no framework |
| minimal | 2/3 (67%) | 2.0 | |
| moderate | 3/3 (100%) | 2.0 | |
| full | 3/3 (100%) | 3.0 | |

**Key observation**: Fastest average turns (2.4) but no framework abstraction benefits.

## Error Analysis

### Error Type Distribution (across all 48 trials)

| Error Type | Count | % of Errors |
|------------|-------|-------------|
| SyntaxError | 40 | 54% |
| AssertionError | 15 | 20% |
| TypeError | 7 | 9% |
| UserError | 4 | 5% |
| AttributeError | 3 | 4% |
| ModuleNotFoundError | 2 | 3% |
| Other | 3 | 4% |

### Failure Patterns

#### 1. Syntax Error Loop (Most Common)

When the model generates code with unterminated string literals (often in multi-line strings or embedded JSON), it gets stuck:

```
Turn 1: SyntaxError - Line 147: unterminated string literal
Turn 2: SyntaxError - Line 107: unterminated string literal
Turn 3: SyntaxError - Line 107: unterminated string literal
... (repeats for all 10 turns)
```

The "Fresh Start with Error" approach doesn't help because the model keeps making the same structural mistake.

#### 2. API Hallucination (PydanticAI)

The model repeatedly tried to use `result_type` parameter which doesn't exist:

```
Turn 1: UserError: Unknown keyword arguments: `result_type`
Turn 2: TypeError: AbstractAgent.run() got an unexpected keyword argument 'result_type'
Turn 3: UserError: Unknown keyword arguments: `result_type`
```

This happened 15 times across trials - the model is "remembering" an API that doesn't exist (or existed in a different version).

#### 3. Logic Errors

Classification errors where the code runs but produces wrong output:

```
AssertionError: Expected 'simple', got 'complex' for dev_001
```

These are harder to fix because the model doesn't know which part of its logic is wrong.

## Contamination Analysis

All frameworks showed high success rates at the "none" documentation level:

| Framework | Success @ None | Interpretation |
|-----------|---------------|----------------|
| PydanticAI | 3/3 (100%) | High contamination |
| LangGraph | 3/3 (100%) | High contamination |
| Haystack | 2/3 (67%) | Moderate contamination |
| Direct API | 2/3 (67%) | Expected (raw HTTP) |

This suggests DeepSeek V3's training data (cutoff Dec 2024) includes these frameworks, despite some being released later. This limits our ability to measure "true learnability" vs "recall from training".

## Why Full Docs Sometimes Hurt

### Haystack Case Study

- **Minimal docs** (9 lines): Clean description + install command → 3/3 first-attempt success
- **Full docs** (26 lines): Marketing copy, no code examples → 1/3 success

The "full" documentation was actually just the landing page content:
- "enterprise-grade support"
- Links to other pages
- No actual code examples

This explains the paradox: more documentation confused the model because it was low-quality documentation.

## Recommendations

1. **Improve doc fetching**: Get actual tutorials with code, not landing pages
2. **Better error recovery**: Show the specific line causing syntax errors
3. **Detect loops**: Track when model is making the same mistake repeatedly
4. **Use older model**: For cleaner contamination control
