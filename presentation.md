# CS 6158 Midterm Report: Multi-Agent Courtroom for Flaky Test Classification

**Authors:** Han Li (hl2595) & Beichen Yu (by325)
**Date:** November 2025

---

## 1. The Problem: Flaky Test Classification

- **What is a Flaky Test?**
  - A test that can both pass and fail with the same code version.
  - Causes: Concurrency, Async waits, Time dependency, Unordered collections, etc.

- **The Challenge for LLMs:**
  - **"The Pattern Matching Trap"**: LLMs tend to rely on surface-level keywords (e.g., seeing `HashMap` -> assumes "Unordered Collection").
  - **Lack of Context**: Unit tests are isolated; understanding them requires knowing the external environment.

- **Goal:**
  - Accurately classify flaky tests into 6 categories (Async, Concurrency, Time, UC, OD, Non-flaky).
  - Move from "Guessing" to "Reasoning".

---

## 2. Prior Work & Limitations

- **Traditional ML / Fine-tuning (e.g., FlakyLens with BERT):**
  - **Pros:** Good at capturing "Annotator Intent" (learning what the human labeler thought).
  - **Cons:** "Black box" reasoning; requires massive labeled data; hard to generalize to new types.

- **Standard LLM Approaches (Zero-shot/Few-shot):**
  - **Pros:** General knowledge, easy to use.
  - **Cons:**
    - **Hallucination:** "Shooting the arrow then painting the target."
    - **Surface-level bias:** Over-focus on keywords like `Thread.sleep` or `timeout`.
    - **Imbalance:** Vast majority of tests are Non-flaky, but LLMs tend to be over-suspicious.

---

## 3. Our Idea: The "Courtroom" Metaphor

To solve the reasoning deficit, we propose a **Multi-Agent Courtroom System**.

- **Core Philosophy: "Presumption of Innocence"**
  - A test is considered **Non-flaky** unless proven otherwise.
  - We separate the **Analysis** (Reasoning) from the **Verdict** (Inferring).

- **The Roles:**
  1.  **Analyst (Defender):** Defends the test, explaining why risks are mitigated.
  2.  **Auditor (Challenger):** Aggressively questions the test (e.g., "Is that timeout really long enough?").
  3.  **Judge:** Weighs the arguments and makes the final classification.

- **Other Key Design Choices (Overview):**
  - Multi-dimensional evidence gathering (Context, Features, Few-shot).
  - Structured static analysis for explicit risk quantification.
  - Two-round debate to ensure thorough examination.

---

## 4. Methodology: Context Extraction (Repo-Mining)

**The Problem:** FlakyLens dataset only contains isolated test method snippets.

- **What We Extract:**
  - **Annotations:** `@Test`, `@Before`, `@After`, `@Mock` (test lifecycle info).
  - **Method Body:** Complete test logic with bracket-matching algorithm.
  - **Surrounding Window:** 20 lines before/after (class fields, helper methods).
  - **External Invocations:** Where else is this method called? (up to 10 locations).

- **Implementation:**
  1. Shallow clone from GitHub based on project name.
  2. Multi-strategy file search: filename match → class declaration search → recursive package path.
  3. Regex + bracket depth counting for method extraction.

- **Why It Didn't Help Much:**
  - Unit tests are designed to be **isolated** — they rarely have external invocations.
  - Class-level context often contains boilerplate (`setUp`, `tearDown`) that doesn't reveal flakiness causes.
  - **Lesson:** Context is necessary but not sufficient; the model still needs guidance on *what to look for*.

---

## 5. Methodology: Feature-based Hints

**The Problem:** LLMs over-rely on surface keywords (Pattern Matching Trap).

- **Our Solution:** Statistical Analysis of Training Set.
  - Calculate **Discrimination Score** = (Density in Flaky Category) / (Density in Non-flaky).
  - *Example:* `CountDownLatch` appears 20x more in Async → "Very Strong" hint.

- **Feature Levels:**
  | Level | Score | Example |
  |-------|-------|---------|
  | Unique | ∞ (only in one category) | `setDaemon` → Async |
  | Very Strong | ≥20x | `CountDownLatch` → Async |
  | Strong | 10-20x | `ExecutorService` → Conc |
  | Moderate | 5-10x | `HashMap` → UC |

- **Why It Has Limitations:**
  - **Over-triggering:** `Thread.sleep` is a moderate hint for Time, but it's also used in Async for waiting.
  - **False confidence:** Model sees "Very Strong hint for Time" and ignores other evidence.
  - **Lesson:** Hints are double-edged swords — they can reinforce bias instead of correcting it.

---

## 6. Methodology: Few-shot via API Signature Matching

**The Problem:** How to retrieve relevant examples for in-context learning?

- **Our Solution:** API Signature Similarity (Jaccard).
  - Extract API calls using **Regex + AST parsing**: method calls, annotations, assertions.
  - $\text{sim}(T_q, T_i) = \frac{|A_q \cap A_i|}{|A_q \cup A_i|}$
  - Retrieve Top-K (default K=3, min similarity 0.1).

- **Why It Had Limited Impact:**
  - **Dataset Imbalance:** 96.7% of tests are Non-flaky → retrieved examples are almost always Non-flaky.
  - **API ≠ Behavior:** Two tests using `ExecutorService` may have completely different flakiness causes.
  - **Lesson:** Few-shot helps recall Non-flaky cases but provides little guidance for identifying *actual* flakiness.

---

## 7. Methodology: Structured Analysis Agent (Static Analysis)

**The Problem:** LLMs struggle to reason about large amounts of unstructured code.

- **Our Solution:** Pre-compute structured JSON with static analysis.

- **What We Analyze:**
  1. **Nondeterministic Operations Detection:**
     - Pattern matching for: `System.currentTimeMillis()`, `Random`, `UUID`, `Thread.sleep()`, etc.
  2. **Contamination Propagation:**
     - Track how nondeterminism flows through variables.
     - *Example:* `long t = System.currentTimeMillis(); int x = (int)t;` → `x` is contaminated.
  3. **Assertion Exposure Calculation:**
     - What % of variables in `assertEquals(expected, actual)` are contaminated?
     - **Exposure > 50%** = High-risk assertion.

- **Output Format (JSON):**
  ```json
  {
    "detected_operations": [
      {"category": "Time", "line": 15, "code": "System.currentTimeMillis()"}
    ],
    "contaminated_variables": ["timestamp", "result"],
    "high_risk_assertions": [
      {"line": 25, "exposure": 0.75, "assertion": "assertEquals(expected, result)"}
    ],
    "preliminary_score": {"Time": 0.8, "Async": 0.1, ...}
  }
  ```

- **Why It Helps:**
  - Provides **explicit, line-level evidence** instead of vague suspicions.
  - Reduces hallucination by grounding the model in concrete analysis.

---

## 8. Methodology: The Courtroom Process (Deep Dive)

### The Evolution of Our Design

**Version 1: Single Agent (Zero-shot)**
- Problem: "Pattern Matching Trap" — sees `HashMap`, outputs "Unordered Collection".

**Version 2: Reasoning Agent + Inferring Agent**
- Separated "Thinking" from "Deciding" to prevent "shoot arrow, paint target".
- Problem: Near 100% Non-flaky accuracy, but Flaky recall was terrible.
- *Why?* Model identified risks but dismissed them as "negligible".

**Version 3: Courtroom (Analyst + Auditor + Judge)**
- Introduced **adversarial debate** to force thorough examination.

---

## 9. Methodology: The Courtroom Roles (Detailed)

### The Analyst (Defender)

- **Role:** Defend the test's stability.
- **Prompt Strategy:**
  - "Assume the test is well-written. Explain why each potential risk is mitigated."
  - Must provide **code references** for every claim.
- **Example Output:**
  > "Line 12 uses `CountDownLatch.await(10, TimeUnit.SECONDS)`. This ensures the async operation completes within a bounded time. The timeout is reasonable for network operations."

### The Auditor (Challenger)

- **Role:** Find loopholes in the Analyst's defense.
- **Key Checklist:**
  - "Is the timeout truly sufficient under load?"
  - "Are static variables properly cleaned up between tests?"
  - "Does the async callback guarantee completion?"
  - "Is the assertion dependent on execution order?"
- **Critical Design:** Must be a **"Rational Code Auditor"**, not a "nitpicker".
  - Cite specific lines.
  - Provide technical grounds, not speculation.
- **Example Output:**
  > "The Analyst claims the timeout is sufficient, but Line 15 shows `Thread.sleep(9000)` inside the async task. If network latency adds 2 seconds, the 10-second timeout will fail. This is a Time-related risk."

### The Judge (Final Verdict)

- **Role:** Weigh both arguments and decide.
- **Input:** Debate history + Original code + Feature hints + Few-shot examples.
- **Decision Logic:**
  - If Auditor's concerns are **weak or speculative** → Default to **Non-flaky**.
  - If Auditor provides **concrete, code-backed evidence** → Classify as Flaky with the most supported category.
- **Key Philosophy:** "Presumption of Innocence" — burden of proof is on showing flakiness.

---

## 10. Methodology: The Two-Round Debate

**Round 1:**
- Analyst presents initial defense.
- Auditor challenges with specific questions.

**Round 2:**
- Analyst responds to Auditor's challenges.
- Auditor provides final rebuttal.

**Why Two Rounds?**
- One round is too shallow — Auditor might raise valid points that Analyst could legitimately refute.
- More than two rounds showed diminishing returns and increased API costs.

**Example Flow:**
```
[Round 1 - Analyst]: "The test uses synchronized blocks, so there's no race condition."
[Round 1 - Auditor]: "But the synchronized block is on a local object (line 8), not the shared static field (line 3)."
[Round 2 - Analyst]: "The static field is only read, never written during the test."
[Round 2 - Auditor]: "However, @BeforeClass initializes it, and parallel test execution could cause order dependency issues."
[Judge]: "The Auditor's final point about @BeforeClass and parallel execution is valid. Classification: Order Dependency."
```

---

## 11. Evaluation Setup

- **Dataset:** FlakyLens (Real-world Java projects).
  - **Total:** 8,574 tests.
  - **Flaky:** 280 (Imbalanced!).
  - **Non-flaky:** 8,294.

- **Test Set:**
  - All **280** Flaky tests.
  - **1,000** Randomly sampled Non-flaky tests.

- **Baseline:**
  - **DeepSeek-V3 (Zero-shot)**: Raw prompt, no agents, no context.

- **Metric:**
  - **Macro F1 Score** (Average across 6 categories).

---

## 12. Results: Significant Improvement

| System | Async | Conc. | Time | UC | OD | Non-flaky | **Macro Avg** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline** | 11.84 | 13.51 | 24.24 | 9.76 | 16.13 | 75.28 | **25.13** |
| **Ours** | **27.63** | **24.32** | **57.58** | **24.39** | **31.18** | **94.82** | **43.32** |

- **Key Wins:**
  - **Non-flaky (94.82%):** The "Presumption of Innocence" works. We stopped false alarms.
  - **Time (57.58%):** Huge jump (+138%). The Auditor is great at catching missing time-bound logic.
  - **Overall:** +18% absolute improvement in Macro F1.

---

## 13. The Key Insight: Fundamental Limitation of Reasoning

### The Core Discovery

**"Flaky Test Classification is an Annotator Intent Inference Problem, not a Pure Logical Reasoning Task."**

---

### Why This Matters

- **The Ambiguity Problem:**
  - Many flaky tests exhibit **multiple** potential causes.
  - *Example:* A test uses `Thread.sleep(1000)` inside an async callback.
    - Is it **Time** flaky (depends on wall-clock timing)?
    - Is it **Async** flaky (callback might not complete)?
    - **Both are logically defensible.**

- **The "Correct" Answer:**
  - Depends on what the **human annotator** considered the **primary** cause.
  - Two experts might disagree — and neither would be "wrong" in a pure reasoning sense.

---

### The Gap Between Reasoning and Pattern Matching

| Approach | What It Does | Strength | Weakness |
|----------|--------------|----------|----------|
| **BERT (Fine-tuned)** | Learns statistical patterns from labels | Mimics annotator's decision process | "Black box", can't explain why |
| **LLM (Reasoning)** | Performs logical analysis | Transparent, generalizable | May be "logically correct" but "labeled wrong" |

- **Fine-tuning teaches:** "When the annotator saw X, they labeled it Y."
- **Reasoning assumes:** "The objectively correct answer is derivable from logic."

---

### A Concrete Example

```java
@Test
public void testAsyncOperation() {
    Future<Result> future = executor.submit(() -> {
        Thread.sleep(500);  // Simulate work
        return computeResult();
    });
    Result result = future.get(1, TimeUnit.SECONDS);
    assertEquals(expected, result);
}
```

**Analyst's Defense:**
> "The `future.get(1, SECONDS)` provides adequate timeout. The sleep is only 500ms, leaving 500ms buffer."

**Auditor's Challenge:**
> "Under CPU contention, `computeResult()` might take longer than 500ms. The total could exceed 1 second."

**Logical Analysis:** Could be **Time** (timeout too short) or **Async** (future completion not guaranteed).

**Actual Label:** Depends entirely on what the annotator *felt* was the primary issue.

---

### Why the Debate Mechanism Can't Fully Solve This

- The Analyst and Auditor can both construct **valid arguments**.
- The Judge has no access to the annotator's **mental model**.
- Without knowing *why* the annotator chose one label over another, the Judge must guess.

**Result:** Many "misclassified" cases are actually **defensible from multiple perspectives**.

---

### The Implication for LLM-based Approaches

1. **Pure reasoning has a ceiling** on this task.
2. **The ceiling is determined by annotator consistency**, not model capability.
3. **To exceed this ceiling**, we must **align with annotator intent**, not just reason better.

---

## 14. Secondary Insights

### Insight A: The Courtroom Balance Problem

- **Initial Failure:** The Judge always agreed with the Analyst.
- **Root Cause:** Auditor was prompted to be "argumentative" — sounded irrational.
- **Fix:** Redefined as **"Rational Code Auditor"** with code citations.
- **Lesson:** In multi-agent debate, **tone matters as much as logic**.

### Insight B: Systematic Time Bias

- **Problem:** Auditor flagged any test with `timeout` as "Time Flaky".
- **Reality:** Timeout is often a **safety mechanism**, not the cause.
- **Fix:** Balanced prompts with explicit criteria for each category.
- **Residual Issue:** Time-related keywords remain visually dominant to LLMs.

---

## 15. Future Work & Timeline

**Goal:** Bridge the gap between "Reasoning" and "Annotator Intent".

### Phase 1: Human-in-the-loop Data Generation (Next 2 Weeks)
- Use our Agents to generate debates.
- **Human Expert** curates the debate:
  - Remove incorrect reasoning paths.
  - Add annotations: "Focus on THIS, not that."
- Create a high-quality "Chain of Thought" dataset.

### Phase 2: Fine-tuning (End of Semester)
- Fine-tune a smaller model (e.g., Llama 3) on curated debate data.
- Teach the model the *specific flavor* of reasoning the annotators used.
- Transform our system from **inference tool** to **data generation framework**.

### Phase 3: Ablation Studies
- Quantify individual contributions:
  - Context Extraction alone?
  - Feature Hints alone?
  - Debate mechanism alone?
- Identify which components provide the most value.

---

## 16. Summary

| Aspect | Description |
|--------|-------------|
| **Problem** | LLMs fall into pattern traps; lack deep reasoning |
| **Solution** | Multi-Agent Courtroom (Analyst vs Auditor vs Judge) |
| **Key Components** | Context, Features, Few-shot, Structured Analysis, Debate |
| **Result** | **43.32%** Macro F1 (vs 25.13% Baseline) |
| **Key Insight** | Flaky test classification = **Annotator Intent Inference** |
| **Future** | Use debate data to fine-tune models aligned with human judgment |

---

## 17. Thank You & Questions

**Key Takeaway:**

> "We discovered that flaky test classification is not purely a reasoning problem — it's about inferring what the human annotator considered most important. This insight redirects our future work from 'better prompts' to 'better alignment with human judgment.'"

**Questions?**
