---
description: Review a Claude Code plan file from a staff engineer perspective
argument-hint: "[--updated] <path-to-plan-file>"
---

# Review Plan

Review a Claude Code plan file from a staff engineer perspective and provide structured feedback across 8 dimensions.

## Arguments

`$ARGUMENTS` may contain:
- **Plan file path** (required): Path to the plan file, e.g., `~/.claude/plans/dreamy-coalescing-brook.md`
- `--updated` (optional): Signal that the plan has been revised since a prior review. Forces a fresh full review and includes a delta assessment of what changed.

Parse `$ARGUMENTS` to extract:
- **--updated**: Split `$ARGUMENTS` on whitespace and check if any token is exactly `--updated`. Remove that token to get the remaining text.
- **Plan file path**: The remaining non-flag tokens after removing `--updated`, joined back together. The flag may appear before or after the path.
- If no path remains after stripping the flag, use AskUserQuestion to request it:
```
Which plan file would you like me to review?

Options:
1. Enter the path (e.g., ~/.claude/plans/plan-name.md)
```

## Constraints

- **Read-only**: Do NOT create, edit, or delete any files. This skill only reads and reports.
- **Advisory-only**: Provide feedback and recommendations. Do not implement fixes.
- **No code changes**: Do not modify any source code, test files, or documentation.
- Use the Read tool for files and the Glob/Grep tools for searching. Do not use Write, Edit, NotebookEdit, or any file-modifying Bash commands.

## Instructions

### Step 1: Read the Plan File

Read the plan file at the path provided in `$ARGUMENTS`.

If the file does not exist, report the error and stop:
```
Error: Plan file not found at <path>
```

### Step 1b: Handle Re-Review (if `--updated`)

If the `--updated` flag is present, this is a re-review of a revised plan.

**You MUST perform a complete fresh review** — do not skip or abbreviate any steps. Treat the plan file contents as the authoritative source, not your memory of a prior version.

After completing the standard 8-dimension review in Step 4, add a **Delta Assessment** section to the output (see Step 5 template for format). This section compares the revised plan against the prior review's feedback:
- Which previously-raised issues have been addressed?
- Which previously-raised issues remain unresolved?
- Are there any new issues introduced by the revisions?

If no prior review is available in conversation context (e.g., the user passed `--updated` on the first invocation, or the context was compressed), still include the Delta Assessment section but fill each subsection with: "Delta assessment unavailable — no prior review found in conversation context. Full fresh review performed."

### Step 2: Read CLAUDE.md for Project Context

Read the project's `CLAUDE.md` file to understand:
- Module structure and architecture
- Estimator inheritance map
- Development checklists (adding parameters, methodology-critical code, etc.)
- Test structure and conventions
- Design patterns (sklearn-like API, formula interface, results objects, etc.)

If the plan modifies estimator math, standard error formulas, inference logic, or edge-case handling, also read `docs/methodology/REGISTRY.md` to understand the academic foundations and reference implementations for the affected estimator(s).

### Step 3: Read Referenced Files

Identify all files the plan references (file paths, module names, class names). Then read them to validate the plan's assumptions:

**Priority order for reading files:**
1. **Files the plan proposes to modify** — read ALL of these first
2. **Files referenced for context** (imports, call sites, existing patterns) — read selectively to verify specific claims

**Scope restriction:**
- Only read files that are within the project repository (the working directory tree).
- The plan file itself (the `$ARGUMENTS` input) is exempt — it can be anywhere (e.g., `~/.claude/plans/`).
- If the plan references paths outside the repo (home directory configs, SSH keys, `/etc/` files, etc.), do NOT read them. Instead, note in the review output under Dimension 2 (Codebase Correctness) that those external paths were not verified.

**What to verify:**
- File paths exist
- Class names, function signatures, and method names match what the plan describes
- Line numbers (if referenced) are accurate
- The plan's description of existing code matches reality

If the plan references more than ~15 files, use judgment: read all files slated for modification, then spot-check context files as needed rather than reading every one.

### Step 4: Evaluate Across 8 Dimensions

#### Dimension 1: Completeness & Executability

Could a fresh Claude Code session — with no access to the conversation history that produced this plan — execute it without asking clarifying questions?

Check for:
- Are all file paths explicit? (No "the relevant file" or "the test file")
- Are code changes described concretely? (Function signatures, parameter names, not just "add a method")
- Are decision points resolved, not deferred? ("We'll figure out the API later" is a red flag)
- Are there implicit assumptions that require conversation context to understand?

#### Dimension 2: Codebase Correctness

Do file paths, class names, function signatures, and line-number references in the plan match the actual codebase?

Use your findings from Step 3. Flag:
- File paths that don't exist
- Function/class names that are misspelled or don't exist
- Line numbers that point to the wrong code
- Descriptions of existing behavior that don't match reality

#### Dimension 3: Scope

Is the scope right — not too much, not too little?

Check for **missing related changes**:
- Tests for new/changed functionality
- `__init__.py` export updates
- `get_params()` / `set_params()` updates for new parameters
- Documentation updates (README, RST, tutorials, CLAUDE.md)
- For bug fixes: did the plan grep for ALL occurrences of the pattern, or just the one reported?

Check for **unnecessary additions**:
- Docstrings/comments/type annotations for untouched code
- Premature abstractions or over-engineering
- Feature flags or backward-compatibility shims when the code can just be changed

#### Dimension 4: Edge Cases & Failure Modes

For methodology-critical code:
- NaN propagation through ALL inference fields (SE, t-stat, p-value, CI)
- Empty inputs / empty result sets
- Boundary conditions (single observation, single group, etc.)

For all code:
- Error handling paths — are they tested with behavioral assertions (not just "runs without exception")?
- What happens when the feature interacts with other parameters/modes?

#### Dimension 5: Architecture & Patterns

Check against CLAUDE.md conventions:
- Does it respect the estimator inheritance map? (Adding a param to `DifferenceInDifferences` auto-propagates to `TwoWayFixedEffects` and `MultiPeriodDiD`; standalone estimators need individual updates)
- Does it use `linalg.py` for OLS/variance instead of reimplementing?
- Does it follow the sklearn-like `fit()` / results-object pattern?
- Is there a simpler alternative that avoids new abstraction?
- Does it match existing code patterns in the codebase?

#### Dimension 6: Plan Execution Risks

Plan-specific failure modes that wouldn't show up in a code review:

- **Ordering issues**: Does the plan propose changes in an order that would break things mid-implementation? (e.g., changing an import before the module it imports from exists, deleting a function before updating its callers)
- **Ambiguous decision points**: Does the plan defer decisions that should be made now? Vague phrases like "choose an appropriate approach" or "handle edge cases" without specifying which ones
- **Missing rollback path**: For risky changes (public API modifications, data format changes), does the plan consider what happens if something goes wrong?
- **Implicit dependencies**: Does step N assume step M was completed, but this ordering isn't stated?

#### Dimension 7: Backward Compatibility & API Risk

- Does the plan add, remove, or rename public API surface (parameters, methods, classes)?
- If so, does it acknowledge the breaking change and state the versioning decision (deprecation period vs clean removal)?
- Downstream effects on:
  - Convenience functions
  - Re-exports in `__init__.py`
  - Existing tutorials and documentation
  - User code that may depend on the current API

#### Dimension 8: Testing Strategy

- Are tests included in the plan? Do they cover the happy path AND the edge cases from Dimension 4?
- Are test assertions behavioral (checking outcomes) rather than just "runs without exception"?
- For bug fixes: does the plan fix all pattern instances and test all of them?
- Are there missing test scenarios? (Parameter interactions, error paths, boundary conditions)

### Step 5: Present Structured Feedback

Present the review in the following format. Do NOT skip any section — if a section has no findings, write "None." for that section. The Delta Assessment section is only included when the `--updated` flag was provided (see Step 1b).

```
## Overall Assessment

[2-3 sentences: what the plan does, whether it's ready for implementation, and the biggest concern if any]

---

## Issues

### CRITICAL
[Issues that would cause implementation failure, incorrect results, or breaking changes if not addressed. Each issue should include: file path and/or line number if applicable, what's wrong, and a concrete suggestion for fixing it.]

### MEDIUM
[Issues that should be addressed but won't block implementation. Missing test cases, incomplete documentation updates, scope gaps.]

### LOW
[Minor suggestions. Style consistency, optional improvements, things to consider.]

---

## Checklist Gaps

Cross-reference against the relevant CLAUDE.md checklists. List which checklist items are not addressed by the plan.

[Identify which CLAUDE.md checklist applies (e.g., "Adding a New Parameter to Estimators", "Implementing Methodology-Critical Code", "Fixing Bugs Across Multiple Locations") and list any items from that checklist that the plan doesn't cover.]

---

## Questions for the Author

[Ambiguities or missing information that should be clarified before implementation begins. Phrase as specific questions.]

---

## Delta Assessment (only include if `--updated` flag was provided)

### Addressed
[List prior issues that have been resolved in the revised plan]

### Unresolved
[List prior issues that remain. Include the original issue text for reference.]

### New Issues
[List any new issues introduced by the revisions, or "None."]

---

## Summary

| Category | Issues |
|----------|--------|
| Critical | [count] |
| Medium | [count] |
| Low | [count] |
| Checklist gaps | [count] |
| Questions | [count] |

**Verdict**: [Ready / Ready with minor fixes / Needs revision]

- **Ready**: No critical issues, few or no medium issues
- **Ready with minor fixes**: No critical issues, some medium issues that are straightforward to address
- **Needs revision**: Has critical issues or many medium issues that require rethinking the approach
```

## Notes

- This skill is strictly read-only — it does not create, edit, or delete any files
- Plan files are typically located in `~/.claude/plans/`
- The review is displayed directly in the conversation, not saved to a file
- For best results, run this before implementing a plan to catch issues early
- The 8 dimensions are tuned for plan-specific failure modes, not generic code review
- Use `--updated` when re-reviewing a revised plan to get a delta assessment of what changed since the prior review
