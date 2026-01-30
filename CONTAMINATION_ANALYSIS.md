# Contamination Analysis Report

## Summary

DeepSeek V3 (training cutoff: mid-2024) shows **high contamination signals** for all tested frameworks. This limits our ability to measure "true learnability" vs "recall from training data."

## Results by Framework

| Framework | Success@None | API Version | Contamination Level | Notes |
|-----------|-------------|-------------|---------------------|-------|
| PydanticAI | 100% (3/3) | - | **HIGH** | 100% success without any docs |
| LangGraph | 100% (3/3) | v0/v1 mixed (50/50) | **HIGH** | Uses APIs from both versions |
| Haystack | 67% (2/3) | v2 (100%) | **HIGH** | Uses modern v2 APIs, not old v1 |
| Direct API | 67% (2/3) | N/A | Expected | Basic HTTP patterns are universal |

## Key Findings

### 1. Haystack Uses Modern APIs Despite Being "Old"

Even though Haystack has been around since 2020, the model uses **v2 APIs exclusively**:
- `PromptBuilder`, `Pipeline`, `Secret.from_token` (v2)
- No usage of deprecated v1 APIs like `DocumentStore`, `BaseReader`, etc.

**Interpretation**: DeepSeek V3's training data includes Haystack v2 (released 2024), not just older v1 content.

### 2. LangGraph Shows Version Mixing

The model uses APIs from both v0.x and v1.x:
- `StateGraph`, `END` (exist in both versions)
- 50/50 ratio between version patterns

**Interpretation**: Training data includes LangGraph content from before and after API changes.

### 3. PydanticAI Perfect Recall

100% success without documentation suggests:
- PydanticAI was in training data despite September 2025 release date
- The model may have seen pre-release/beta documentation
- Or: The API is intuitive enough to guess correctly

## Detection Approaches Implemented

### 1. Success@None Metric
Simple but effective: If the model succeeds without documentation, it's using training data.

```
Threshold: ≥66% success at "none" → High contamination
```

### 2. Version Alignment Analysis
Compare which API version the generated code matches:

```python
# v1 patterns (deprecated)
HAYSTACK_V1_APIS = {"DocumentStore", "BaseReader", "Pipeline.add_node", ...}

# v2 patterns (current)
HAYSTACK_V2_APIS = {"PromptBuilder", "OpenAIGenerator", "Pipeline", ...}
```

If model uses old APIs despite new documentation → Training data contamination

### 3. Doc Contradiction Test (Available but not run)
Provide modified documentation with renamed APIs:
- Original: `PromptBuilder`
- Modified: `TemplateBuilder`

If model uses `PromptBuilder` anyway → Ignoring docs, using training data

### 4. Code Fingerprinting
Compare code structure across documentation levels:
- High similarity between "none" and "full" → Not learning from docs

## Recommendations

### For This Research

1. **Use older models**: Models with earlier training cutoffs (e.g., pre-2024) would have fewer frameworks memorized

2. **Use newer frameworks**: Frameworks released after training cutoff provide cleaner signal
   - Anthropic Agents SDK (2025)
   - OpenAI Agents SDK (2025)

3. **Run Doc Contradiction Tests**: Modify documentation to detect if model actually reads it

4. **Focus on Tier 2/3 tasks**: More complex tasks may show differentiation even with contaminated frameworks

### For Framework Authors

The contamination analysis reveals what's in LLM training data:
- If success@none is low → Documentation is critical for adoption
- If success@none is high → Users may not need docs (but may get outdated patterns)

## Running the Analysis

```bash
# Analyze existing experiment results
python scripts/analyze_contamination.py --results-dir results/exp_20260129_234834

# Analyze specific framework
python scripts/analyze_contamination.py --results-dir results/exp_20260129_234834 --framework haystack
```

## Files

- `src/analysis/contamination.py` - Core contamination detection logic
- `scripts/analyze_contamination.py` - CLI tool for analysis
- Version-specific API patterns for Haystack and LangGraph included

## Future Work

- [ ] Run Doc Contradiction Test on new experiment
- [ ] Add more framework version patterns
- [ ] Symbol recency dating (when was each API introduced?)
- [ ] Test with older models (GPT-4 Turbo has earlier cutoff)
