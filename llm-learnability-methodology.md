# LLM Learnability: Measuring Framework Usability for Selection

## Purpose

This methodology measures how easily an LLM can produce working code with each framework, given access to documentation. This serves as a practical signal for framework selection—frameworks that are LLM-friendly are increasingly relevant since many developers learn and code with AI assistants.

**What we're measuring:** LLM-assisted developer usability—how well the framework + docs + error messages support an LLM pair programmer.

**What we're not claiming:** Direct equivalence to human learnability (though correlation is likely).

**Explicit scope statement:**

> This benchmark measures *LLM-assisted implementation usability under constrained documentation exposure*, not absolute framework quality. Success can come from good documentation, intuitive API naming, or task-framework alignment. We use contamination detection, hidden tests, and multi-tier tasks to isolate these factors where possible.

---

## Core Methodology

### The "Turns to Working Code" Loop

For each framework:

```
1. Initialize conversation with:
   - Task description
   - Framework documentation (at specified level)

2. Loop (max 10 turns):
   a. LLM generates implementation attempt
   b. Run code against test suite
   c. If tests pass → record turns, exit
   d. If tests fail → feed error message back to LLM

3. Record metrics (see below)
```

### Context Management Strategy

**Problem:** If the LLM fails 5 times, the context window fills with 5 wrong attempts and 5 error logs. The LLM may get stuck in a rut, confused by its own prior bad code. A framework might appear harder simply because its first error message was verbose enough to flood the context.

**Solution: "Fresh Start with Error"**

On turn N, provide only:
- Original task description
- Framework documentation
- Code from turn N-1
- Error message from turn N-1

Do **not** include turns 1 through N-2. This isolates each correction attempt and makes the self-correction metric more meaningful—we're measuring whether the LLM can fix *this specific error*, not whether it can navigate a maze of accumulated mistakes.

### Key Design Principles

- **Deterministic**: Temperature 0, pinned versions, reproducible
- **Bounded**: Max 10 turns prevents runaway costs
- **Informative**: Rich logging captures where and why failures occur
- **Fair**: Same task, same tests, equivalent documentation coverage

---

## Experimental Design

### Independent Variables

| Variable | Values |
|----------|--------|
| Framework | PydanticAI, Haystack, LangGraph, OpenAI Agents SDK, Direct API |
| Documentation Level | None, Minimal, Moderate, Full |
| Model | DeepSeek V3 (primary), Qwen 2.5 Coder (secondary) |
| Task Tier | Tier 1 (classification), Tier 2 (tool use), Tier 3 (agent-native) |

### Task Tier Rationale

| Tier | Task Type | What It Tests | Why It Matters |
|------|-----------|---------------|----------------|
| **Tier 1** | Simple classification | Basic framework setup, model invocation | Baseline overhead—if a framework is hard here, it's hard everywhere |
| **Tier 2** | Tool use | Tool definition, schema handling, result parsing | Core agent functionality—where frameworks should add value |
| **Tier 3** | Agent-native | Routing, memory/state, multi-step tool loops | Where agent frameworks justify their complexity |

**Why Tier 3 matters:** Tiers 1-2 may unfairly penalize agent frameworks for overhead that doesn't pay off on simple tasks. Tier 3 tests routing between multiple tools, maintaining state across turns, and multi-step reasoning—the problems agent frameworks are designed to solve. A framework that looks heavy in Tier 1 but excels in Tier 3 is doing its job.

### Model Selection Rationale: The Temporal Firewall

**The core problem:** If the model has seen a framework in its training data, we measure "Recall" (how well it remembers), not "Learnability" (how well it reads docs).

**The solution:** Use a capable model whose training data *definitively predates* the target frameworks.

| Model | Release/Cutoff | Contamination Risk |
|-------|----------------|-------------------|
| DeepSeek V3 | Mid-2024 | 🛡️ Cannot know frameworks released after mid-2024 |
| Qwen 2.5 Coder | Nov 2024 | 🛡️ Cannot know frameworks released in 2025 |
| Claude 4.5 Sonnet | Sept 2025 | ⚠️ High risk—likely trained on beta versions |

| Framework | Release | Status for DeepSeek V3 |
|-----------|---------|------------------------|
| PydanticAI v1.0 | Sept 2025 | 🛡️ Safe—post-dates training |
| Anthropic Agent SDK | Sept 2025 | 🛡️ Safe—post-dates training |
| OpenAI Agents SDK | 2025 | 🛡️ Safe—post-dates training |
| LangGraph v1.0 | Oct 2025 | ⚠️ Caution—older versions in training data |

**Why this is scientifically superior:**

A smart model frozen in time, learning tools from the future, is the gold standard for learnability testing:
- If it succeeds → documentation was good (valid signal)
- If it fails → documentation was unclear or API is unintuitive (valid signal)

Using a 2025 model like Claude 4.5 Sonnet creates the "Just-Missed Paradox": did it fail because it doesn't know the framework, or because it learned the beta API and is now hallucinating outdated patterns instead of reading your v1.0 docs?

**The LangGraph exception:**

LangGraph has existed since early 2024. DeepSeek V3 has likely seen v0.1/v0.2 in training. Mitigations:
1. The contamination matrix will catch this—high Success@None signals memorization
2. Add explicit version instruction: *"You are using LangGraph v1.0. Do not use methods from v0.2 or earlier. Rely ONLY on the provided documentation."*
3. Version Conflict hallucination tracking will flag old-API usage

**Why DeepSeek V3 over Llama 3 8B:**

DeepSeek V3 is substantially more capable. This is *better* for the experiment—failures are more clearly attributable to framework/doc issues rather than model limitations. With a weaker model, there's always the confound: "is this framework hard, or is the model just not smart enough?"

### Documentation Levels (Conceptual, Not Token-Based)

| Level | Content |
|-------|---------|
| None | Framework name only |
| Minimal | One-paragraph description + install command |
| Moderate | Official quickstart + core concepts pages |
| Full | Above + tool use section + relevant API reference |

**Key insight:** If Framework A needs 4000 tokens to explain what Framework B explains in 500, that verbosity is itself a usability signal. We record token counts as a secondary metric ("Documentation Verbosity").

**Documentation selection rules:**
- Always include the official quickstart page
- For Tier 2: include the official "tools / function calling" page
- Include API reference for minimum symbols needed by quickstart + tools page
- Snapshot docs at experiment start (don't scrape live)
- Publish exact doc extracts or hashes alongside results

**Documentation normalization:**

LLMs are sensitive to formatting. If Framework A's docs are clean markdown and Framework B's are scraped HTML with artifacts, we're measuring doc formatting quality, not framework learnability.

- Convert all documentation to standardized Markdown format
- Normalize headers, code blocks, and lists
- Strip navigation elements, sidebars, and non-content artifacts
- Preserve code examples and API signatures exactly

---

## Metrics

### Primary Metrics

| Metric | Description |
|--------|-------------|
| **Turns to Success** | Conversation turns until tests pass (see survival analysis below) |
| **Success Rate** | % of attempts that eventually succeed |
| **First-Attempt Success** | % that work on first try |
| **Total Tokens** | Cumulative tokens generated |
| **Meaningful Error Rate** | % of early failures (turns 1-2) with actionable error messages |

### Survival Analysis for Turns-to-Success

**Problem:** With max-10 turns, you have right-censored data. Treating failures as "turns=10" distorts averages and makes frameworks look similar when both failed frequently.

**Solution:** Analyze turns-to-success like survival data:

- **Success probability by turn:** Kaplan-Meier style curve showing P(success by turn N)
- **Median turns among successes:** Central tendency without censoring distortion
- **Success rate:** Separately reported, not conflated with turn count
- **Treat failures as censored at turn 10:** Not as "took 10 turns"

This framing makes comparisons cleaner. A framework with 60% success in median 3 turns is clearly better than one with 60% success in median 7 turns—but both would look similar if you averaged in failures as 10.

### Meaningful Error Rate

Two frameworks can both fail in turn 1, but one fails with "missing API key" and the other fails with a crisp type error that tells you the fix.

**Computation:**
- For failures in turns 1-2, classify error messages:
  - **Actionable:** Clear what to fix (type error with expected vs actual, missing parameter, invalid value)
  - **Generic/Internal:** Stack trace pointing into library internals, opaque exception
- Report: actionable / total early failures

This predicts real-world developer pain better than raw success rates.

### Contamination Detection (Two-Axis)

Rather than a single "Prior Knowledge Score," we split contamination detection:

| Metric | What It Measures | How to Compute |
|--------|------------------|----------------|
| **Success at None** | Did it work without docs? | Success rate with framework name only |
| **Symbol Exactness at None** | Did it guess or memorize? | Fraction of API calls that exactly match installed package's symbol table |

**Interpretation matrix:**

| Success at None | Symbol Exactness | Interpretation |
|-----------------|------------------|----------------|
| High | High | Memorized from training data—learnability metric less reliable |
| High | Low | API is highly guessable/intuitive—valid signal |
| Low | - | Clean slate—learnability metric fully valid |

### Derived Metrics (Where It Hurts)

| Metric | Description | Computation |
|--------|-------------|-------------|
| **Self-Correction Ratio** | Does LLM adapt or repeat mistakes? | Cluster attempts by AST similarity; count distinct clusters / total turns |
| **Hallucination Rate** | Invented non-existent APIs | Hallucinated calls / total API calls (see below) |
| **Boilerplate Coefficient** | Framework overhead vs task logic | Import + setup LOC / task logic LOC |
| **Error Debuggability** | How helpful are error messages? | Errors self-corrected / total errors |
| **Documentation Verbosity** | Tokens needed to explain concepts | Token count of full docs per framework |
| **Framework Exception Ratio** | Quality of error messages | Framework-specific exceptions / total exceptions |

### Hallucination Types (Split Metric)

Not all hallucinations are equal. We distinguish:

| Type | Description | What It Signals |
|------|-------------|-----------------|
| **Invented** | Method/class that never existed in any version | Unintuitive API naming—LLM guessed wrong |
| **Version Conflict** | Method existed in older version (in training data) but not current | Breaking changes, poor deprecation signals, version drift |

**Detection approach:**
- Extract API surface from *installed* version (current)
- Extract API surface from *previous major versions* (if available via git tags or PyPI history)
- If hallucinated symbol exists in old version but not current → Version Conflict
- If hallucinated symbol never existed → Invented

This distinction matters: a framework with high "Invented" hallucinations has unintuitive naming; one with high "Version Conflict" hallucinations has unstable APIs or the LLM is fighting its training data.

### Error Message Attribution

**Problem:** Many errors are generic Python exceptions (`KeyError`, `AttributeError`), not informative framework errors.

**Solution:** Classify exceptions by origin:

| Type | Description | Example |
|------|-------------|---------|
| **Framework Exception** | Custom exception class defined in the package | `AgentStateError: Missing required tool` |
| **Standard Exception (Framework Context)** | Built-in exception with framework in traceback | `KeyError: 'model'` from inside `langchain/agents.py` |
| **Standard Exception (User Code)** | Built-in exception from generated code | `KeyError: 'model'` from `solution.py` |

A framework scores higher on debuggability if it raises helpful custom exceptions rather than letting raw `KeyError` bubble up from library internals.

### Error Category Taxonomy

| Category | Description | Example |
|----------|-------------|---------|
| Import | Module not found, wrong import path | `ModuleNotFoundError` |
| Dependency | Missing extras, system deps, version conflicts | `pip install foo[all]` required but not documented |
| Type | Type mismatches, wrong argument types | `TypeError: expected str, got int` |
| Runtime | Execution failures, exceptions | `RuntimeError: API key not set` |
| Logic | Code runs but wrong output | Test assertion fails |
| Hallucination | Non-existent API called | `AttributeError: 'Agent' has no method 'verify'` |
| Syntax | Python syntax errors | `SyntaxError: unexpected EOF` |

### Dependency Friction (Separate Tracking)

**Problem:** Some frameworks require `pip install foo[all]` or system dependencies not mentioned in Minimal/Moderate docs. Failures from missing dependencies are about install complexity, not API learnability.

**Solution:** Track dependency friction separately:

- **Extras required:** Does the framework need optional extras (`[all]`, `[tools]`, etc.)?
- **System deps:** Does it need non-Python dependencies (e.g., `libmagic`, `poppler`)?
- **Docs mention install complexity:** Do Minimal/Moderate docs explain this?
- **Failure attribution:** If error is clearly dependency-related, attribute to this category, not Import/Runtime

Report dependency friction alongside learnability scores—a framework with great API learnability but painful installation is still painful.

---

## Critical Controls

### 1. Task Difficulty Floor (Baseline)

Before testing frameworks, run Direct API (pure Python + requests) implementation:
- If LLM succeeds with full docs → task difficulty is appropriate
- If LLM fails → task too hard, simplify before proceeding

This ensures we measure framework overhead, not raw task complexity.

### 2. Framework Compliance Gates

**Problem:** LLM might bypass the framework entirely and just write vanilla Python.

**Solution:** Enforce framework usage:

- **Static check:** Submission must import target framework and reference at least one framework-specific class/function from an allowlist
- **Dynamic check:** Wrap expected framework entrypoint; assert it was called at runtime
- **Environment isolation:** Each framework runs in its own venv/container with only that framework + minimal dependencies

Track "LLM bypass rate" as its own metric, but don't let it pollute learnability scores.

### 3. Deterministic Stub Models for Tier 2

**Problem:** If agent code uses a live LLM for tool selection, we're measuring model reliability, not framework usability.

**Solution:** Use deterministic stub models inside code-under-test:

```python
# Scripted fake that returns predetermined responses
class StubLLM:
    def __init__(self, responses: dict):
        self.responses = responses
    
    def complete(self, prompt):
        # Return predetermined tool calls based on test claim ID
        return self.responses.get(extract_claim_id(prompt))
```

This isolates framework wiring + tool plumbing + state management from LLM stochasticity.

### 4. Reproducibility Requirements

- Pin exact versions: framework, Python, all dependencies
- Each framework in isolated container/venv with lockfile
- Store snapshot of docs used (don't scrape live at runtime)
- Log: model ID, server build, decoding params
- Temperature = 0

---

## Hallucination Detection: API Surface Extraction

To accurately measure hallucination rate and distinguish types:

```python
import inspect
import importlib

def extract_api_surface(package_name):
    """Extract all public symbols from a package."""
    symbols = set()
    module = importlib.import_module(package_name)
    
    for name, obj in inspect.getmembers(module):
        if not name.startswith('_'):
            symbols.add(f"{package_name}.{name}")
            # Recurse into classes
            if inspect.isclass(obj):
                for method_name, _ in inspect.getmembers(obj):
                    if not method_name.startswith('_'):
                        symbols.add(f"{package_name}.{name}.{method_name}")
    
    return symbols

def classify_hallucination(symbol, current_surface, old_surfaces):
    """
    Classify a hallucinated symbol.
    
    Args:
        symbol: The API call that failed
        current_surface: Symbols from installed version
        old_surfaces: Dict of {version: symbols} from previous versions
    
    Returns:
        'invented' | 'version_conflict' | 'valid'
    """
    if symbol in current_surface:
        return 'valid'
    
    for version, surface in old_surfaces.items():
        if symbol in surface:
            return 'version_conflict'
    
    return 'invented'
```

**Building old version surfaces:**
- Check out previous major version tags from git
- Or install previous versions from PyPI in isolated venvs
- Extract surfaces from each, store as reference

**Classify hallucinations as:**
- **Invented:** Symbol not in any known version—API naming is unintuitive
- **Version Conflict:** Symbol existed in old version—framework has breaking changes or LLM is fighting training data
- **Ambiguous:** Dynamic dispatch, plugin system (flag but don't count)

---

## Test Suite Design

Tests must be:
- **Framework-agnostic:** Test inputs/outputs, not implementation details
- **Deterministic:** Same input always expects same output
- **Informative:** Error messages help identify failure type

```python
def test_tier1_classification():
    """Test that implementation correctly classifies claims."""
    classifier = load_implementation(framework_name)
    
    for claim in TEST_CLAIMS:
        result = classifier.classify(claim)
        
        assert result.label in ['simple', 'complex'], \
            f"Invalid label: {result.label}"
        assert result.label == claim['expected_label'], \
            f"Wrong classification for {claim['claim_id']}"

def test_tier2_tool_use():
    """Test that implementation correctly uses tools."""
    classifier = load_implementation(framework_name)
    
    for claim in TEST_CLAIMS:
        result = classifier.classify_with_tools(claim)
        
        assert result.tools_called is not None
        assert len(result.tools_called) >= 1
```

### Hidden Test Set (Critical for Credibility)

**Problem:** If the model can infer labels from visible test cases, or if tests are in the environment, it can "cheat" by special-casing. Even without malice, LLMs sometimes converge on brittle solutions that pass a narrow visible suite.

**Solution:** Use a hidden test set:

- **Development set (visible):** 5 test cases used during the turns-to-success loop for error feedback
- **Hidden set (held out):** 10+ additional test cases from the same distribution, never shown to the model
- **Final scoring:** Report success rate on *hidden* set, not development set

**Implementation:**
- Sandbox cannot read hidden test files (or test suite runs outside the container where solution lives)
- Development and hidden sets drawn from same distribution (same task structure, similar difficulty)
- Consider property-style tests where possible (invariants that hold regardless of specific inputs)

This single addition dramatically upgrades credibility against "the model just memorized the test cases" critiques.

---

## Implementation Plan

### Phase 1: Setup (Day 1)
1. Prepare documentation corpus (extract, normalize to markdown, snapshot)
2. **Doc ingestion pre-flight check** (see below)
3. Build test harness (conversation loop, code extraction, sandbox, logging)
4. Create test suites:
   - Development set: 5 claims for error feedback during turns loop
   - Hidden set: 10+ claims for final scoring (same distribution, never shown to model)
   - All three tiers: classification, tool use, agent-native
5. Build API surface extractor for hallucination detection (current + old versions)
6. Set up isolated environments per framework

**Doc Ingestion Pre-Flight Check:**

Before running any trials, validate that documentation snapshots are LLM-readable:

```
For each framework:
  1. Feed "Full" doc snapshot to the LLM
  2. Prompt: "Summarize the key classes and their primary methods"
  3. If LLM cannot parse/summarize coherently → doc snapshot is malformed
  4. Fix extraction/normalization before proceeding
```

This catches data entry errors early. If the LLM can't understand your doc snapshot, the entire experiment produces garbage.

### Phase 2: Baseline & Pilot (Day 2)
1. Run Direct API baseline to validate task difficulty (all tiers)
2. Run PydanticAI end-to-end (all doc levels, all tiers)
3. Validate compliance gates work
4. Validate hidden test set catches overfitting to development set
5. Check error categorization captures meaningful distinctions
6. Adjust prompts/tests if needed

### Phase 3: Full Experiment (Days 3-5)
```
5 frameworks × 4 doc levels × 3 tiers × 2 models × 5 runs = 600 trials
```

At ~2 minutes per trial, ~20 hours runtime. Parallelizable to ~4 hours.

### Phase 4: Analysis (Day 6)
1. Calculate metrics per framework (using hidden test set for final scores)
2. Survival analysis for turns-to-success
3. Contamination analysis (success-at-none × symbol-exactness matrix)
4. Generate visualizations
5. Synthesize findings for framework selection

---

## Implementation Notes

### Trial Logging Requirements

Every trial must log sufficient data for reproducibility and debugging. Recommended fields:

```json
{
  "trial_id": "uuid",
  "timestamp": "ISO8601",
  "framework": "pydantic-ai",
  "framework_version": "1.0.2",
  "doc_level": "full",
  "doc_hash": "sha256:abc123...",
  "doc_token_count": 3420,
  "task_tier": 2,
  "model": "deepseek-v3",
  "model_version": "2024-07-01",
  
  "turns": [
    {
      "turn": 1,
      "code_hash": "sha256:...",
      "code_loc": 45,
      "error_type": "hallucination",
      "error_signature": "AttributeError:Agent.verify",
      "error_is_framework_exception": false,
      "error_is_actionable": true,
      "tokens_generated": 892,
      "hallucinated_symbols": ["Agent.verify"],
      "hallucination_types": {"invented": 1, "version_conflict": 0}
    }
  ],
  
  "outcome": "success",
  "final_turn": 3,
  "dev_set_pass": true,
  "hidden_set_pass": true,
  "hidden_set_score": 0.9,
  "compliance_check_pass": true
}
```

### Error Signature Normalization

The same underlying error can show up with different stack trace noise across environments. Normalize before clustering:

```python
def normalize_error_signature(exception, traceback):
    """
    Extract stable error signature for clustering.
    Returns: "ExceptionType:key_message_fragment"
    """
    exc_type = type(exception).__name__
    # Extract first line of message, strip paths/line numbers
    msg = str(exception).split('\n')[0]
    msg = re.sub(r'line \d+', 'line N', msg)
    msg = re.sub(r'/[\w/]+\.py', 'module.py', msg)
    return f"{exc_type}:{msg[:100]}"
```

### Version Pinning Checklist

Log these in every trial to catch doc/API drift:

- [ ] Framework version (exact, from `pkg.__version__`)
- [ ] Python version
- [ ] Doc snapshot hash
- [ ] Doc snapshot date
- [ ] All dependency versions (pip freeze)
- [ ] Model identifier and API version

If doc hash + framework version don't match, results are suspect.

---

## Expected Outputs

### Primary Results Table

| Framework | Success@None | Symbol Exact | Learnability Δ | Median Turns | Success Rate | Meaningful Error % | Halluc (Inv) | Halluc (Ver) |
|-----------|--------------|--------------|----------------|--------------|--------------|-------------------|--------------|--------------|
| Direct API | ?% | — | (baseline) | ? | ?% | ?% | — | — |
| PydanticAI | ?% | ?% | ? | ? | ?% | ?% | ?% | ?% |
| Haystack | ?% | ?% | ? | ? | ?% | ?% | ?% | ?% |
| LangGraph | ?% | ?% | ? | ? | ?% | ?% | ?% | ?% |
| OpenAI Agents | ?% | ?% | ? | ? | ?% | ?% | ?% | ?% |

*Note: Results reported separately for each tier. Median turns computed among successes only; failures treated as censored.*

### Key Visualizations

1. **Survival curves (Kaplan-Meier):** P(success by turn N) per framework—shows how quickly each framework yields working code

2. **Learning curve plot:** Success rate vs documentation level (per framework)

3. **Friction heatmap:** Framework × Error Category → frequency (shows where each framework hurts)

4. **Contamination scatter:** Success@None vs Symbol Exactness (distinguishes memorized from guessable)

5. **Learnability Delta quadrant plot:**
   - X-Axis: Prior knowledge (Success @ None)
   - Y-Axis: Doc responsiveness (Success @ Full − Success @ None)
   - Quadrants:
     - **Top-Left: Gold Standard** — Low prior knowledge, high doc response (new to model, easy to learn)
     - **Top-Right: Trivial/Standard** — High prior knowledge, high doc response (model knows it, docs still help)
     - **Bottom-Left: Hard/Obscure** — Low prior knowledge, low doc response (unknown and docs don't help)
     - **Bottom-Right: Legacy Trap** — High prior knowledge, low doc response (model knows old patterns, docs don't improve)

6. **Self-correction trajectories:** How error types shift across turns (do errors progress from Import → Logic as LLM learns?)

7. **Hallucination breakdown:** Stacked bar of Invented vs Version Conflict hallucinations per framework

8. **Tier comparison:** Success rate by tier (1/2/3) per framework—shows where abstraction overhead pays off

### Questions to Answer

**Learnability:**
- Which framework requires the least documentation to use successfully?
- Which frameworks benefit most from documentation (high learnability delta)?
- Which frameworks fall into the "Gold Standard" quadrant (low prior knowledge, high doc response)?

**Error Patterns:**
- Do different frameworks fail in different ways?
- Which have the most debuggable error messages (high self-correction)?
- Which have the highest meaningful error rate (actionable early failures)?
- Which have intuitive APIs (low hallucination rate)?
- Which frameworks raise helpful custom exceptions vs raw Python errors?

**API Stability:**
- Which frameworks have high Version Conflict hallucinations (LLM fighting its training data)?
- Does this correlate with recent breaking changes in the framework?

**Complexity Scaling:**
- Does Tier 2 (tool use) amplify framework differences?
- Does Tier 3 (agent-native) show where framework abstraction pays off?
- Which frameworks scale gracefully across all tiers?
- Which frameworks have overhead that only pays off at Tier 3?

---

## Limitations & Mitigations

| Limitation | Mitigation |
|------------|------------|
| LLM ≠ Human | Frame as "LLM-assisted usability"; note correlation is plausible but unproven |
| Doc quality varies | Measure verbosity as signal, not confounder |
| Doc formatting varies | Normalize all docs to standardized markdown |
| Test suite overfitting | Hidden test set for final scoring; development set only for error feedback |
| Test suite bias | Direct API baseline validates task is fair |
| Training contamination | Temporal firewall (model predates frameworks) + two-axis detection for edge cases |
| LLM bypasses framework | Compliance gates + bypass rate as separate metric |
| Context pollution spiral | "Fresh Start with Error" context management—only show previous turn's code + error |
| Version drift hallucinations | Split hallucination metric into Invented vs Version Conflict |
| LangGraph pre-exists model | Explicit version instructions + contamination matrix will flag memorization |
| Censored turn data | Survival analysis (Kaplan-Meier) treats failures as censored, not as "10 turns" |
| Task-framework mismatch | Three tiers (simple → agent-native) show where abstraction overhead pays off |
| Dependency friction conflated with API friction | Separate error category for dependency issues; track install complexity |

---

## Cost Estimate

| Component | Cost |
|-----------|------|
| DeepSeek V3 API (~750K tokens for 600 trials) | ~$1.00–1.50 |
| Qwen 2.5 Coder (local) | $0 (compute only) |
| **Total** | **< $3** |

*Note: DeepSeek V3 API pricing is competitive. Qwen 2.5 Coder can run locally for zero API cost if you have adequate GPU.*

---

## Summary

This methodology provides:
- **Reproducible measurement** of framework usability via LLM proxy
- **Temporal firewall** using a model that predates target frameworks, eliminating training contamination
- **Hidden test sets** preventing overfitting to visible test cases
- **Survival analysis** treating turn-to-success as censored data for clean comparisons
- **Three-tier task design** showing where framework abstraction overhead pays off
- **Rich diagnostics** showing where and why frameworks cause friction
- **Contamination-aware** scoring that distinguishes memorization from intuitive design
- **Version-aware hallucination tracking** that separates "unintuitive API" from "unstable API"
- **Context-controlled iteration** that isolates self-correction ability from accumulated confusion
- **Meaningful error rate** tracking actionable vs opaque early failures
- **Practical relevance** for framework selection given prevalence of AI-assisted development

The key claim: "Given a fixed task, fixed doc budget, and fixed debugging loop, frameworks differ measurably in how quickly an LLM can reach passing code, what kinds of errors occur, and how effectively error feedback leads to correction."
