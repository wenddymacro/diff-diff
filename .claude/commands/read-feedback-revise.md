---
description: Read PR review feedback from a GitHub comment and enter plan mode to address it
argument-hint: "<comment-URL>"
---

# Read Feedback & Revise

Read PR review feedback from a GitHub comment URL, analyze the feedback, and enter plan mode to produce an actionable implementation plan addressing the reviewer's concerns.

## Arguments

`$ARGUMENTS` should contain:
- **Comment URL** (required): A GitHub PR comment URL in one of these formats:
  - `https://github.com/{owner}/{repo}/pull/{number}#issuecomment-{id}` (PR-level comment)
  - `https://github.com/{owner}/{repo}/pull/{number}#discussion_r{id}` (inline review comment)
  - `https://github.com/{owner}/{repo}/pull/{number}#pullrequestreview-{id}` (full PR review)

## Constraints

Steps 1-7 are **read-only** -- only fetching data from GitHub, reading local files, and analyzing content. No file modifications occur until the user approves the plan generated in plan mode (Step 8). The one exception is Step 4, which may switch branches (with user confirmation) to ensure subsequent reads target the correct code.

## Instructions

### Step 1: Parse Arguments

Parse `$ARGUMENTS` to extract the comment URL. If `$ARGUMENTS` is empty or does not contain a URL, use AskUserQuestion to request one:
```
What is the URL of the review comment you want to address?

Supported formats:
1. PR-level comment: https://github.com/owner/repo/pull/42#issuecomment-123456
2. Inline review comment: https://github.com/owner/repo/pull/42#discussion_r789012
3. Full PR review: https://github.com/owner/repo/pull/42#pullrequestreview-345678
```

Strip any query parameters from the URL before parsing: remove the query string (the `?...` portion) while preserving the `#` fragment. For example, `https://github.com/o/r/pull/1?notification_referrer_id=abc#issuecomment-123` becomes `https://github.com/o/r/pull/1#issuecomment-123`. If the URL has `?` after `#`, no stripping is needed (the `?` is part of the fragment). However, if the fragment itself contains `?` (e.g., `#discussion_r123?foo=bar`), strip the `?` and everything after it from the fragment before pattern matching, since GitHub fragments never contain `?` as meaningful data. Then parse the URL path and fragment:

**URL path pattern**: `https://github.com/{owner}/{repo}/pull/{pr_number}[/...]#{fragment}`

The `{pr_number}` is always the path segment immediately after `/pull/`. Additional path segments (like `/files`, `/commits`) may appear between the PR number and the `#` fragment -- ignore them.

Only `github.com` URLs are supported. If the URL host is not `github.com`, report an error:
```
Error: Only github.com URLs are currently supported. GitHub Enterprise URLs are not handled.
```

**Fragment patterns:**

| URL Fragment | Comment Type | `gh api` Endpoint |
|---|---|---|
| `#issuecomment-{id}` | Issue comment (PR-level) | `repos/{owner}/{repo}/issues/comments/{id}` |
| `#discussion_r{id}` | Inline review comment | `repos/{owner}/{repo}/pulls/comments/{id}` |
| `#pullrequestreview-{id}` | PR review | `repos/{owner}/{repo}/pulls/{pr_number}/reviews/{id}` |

Extract and store: `owner`, `repo`, `pr_number` (from path), `comment_type`, `comment_id` (from fragment).

If the URL doesn't match any fragment pattern, use AskUserQuestion to request a valid URL with the three supported format examples shown above.

### Step 2: Fetch Comment Content

Use `gh api` based on comment type:

**Issue comment:**
```bash
gh api repos/{owner}/{repo}/issues/comments/{id} --jq '{body: .body, user: .user.login, created_at: .created_at}'
```

**Inline review comment:**
```bash
gh api repos/{owner}/{repo}/pulls/comments/{id} --jq '{body: .body, user: .user.login, path: .path, line: .line, diff_hunk: .diff_hunk}'
```

**PR review** (fetch both the review body AND its inline comments):
```bash
# Review body
gh api repos/{owner}/{repo}/pulls/{pr_number}/reviews/{id} --jq '{body: .body, user: .user.login, state: .state}'

# All inline comments belonging to this review
gh api repos/{owner}/{repo}/pulls/{pr_number}/reviews/{id}/comments --paginate --jq '.[] | {body: .body, path: .path, line: .line, diff_hunk: .diff_hunk}'
```

Note: The inline comments query uses `.[] | {...}` (newline-delimited JSON objects) rather than wrapping in an array, because `--paginate` concatenates multiple pages and `[...] + [...]` is not valid JSON. The `.[] |` format produces newline-delimited objects that concatenate cleanly.

For PR reviews, the actionable feedback is often in the inline comments, not just the review body. Always fetch both.

### Step 3: Fetch PR Context

```bash
# PR metadata
gh pr view {pr_number} --repo {owner}/{repo} --json title,body,baseRefName,headRefName,author

# Changed files
gh api repos/{owner}/{repo}/pulls/{pr_number}/files --paginate --jq '.[] | .filename'
```

### Step 4: Check Branch State

```bash
git branch --show-current
```

Compare the current branch to the PR's head branch (`headRefName`). If they don't match, use AskUserQuestion to ask:
```
You are currently on branch '<current-branch>' but the PR's head branch is '<headRefName>'.

Options:
1. Continue on current branch
2. Check out the PR branch first (git checkout <headRefName>)
```

This prevents generating a plan against the wrong code state.

### Step 5: Read Project Context and Source Files

Read files in this order:

1. **CLAUDE.md** -- project conventions and architecture (informs analysis in Step 6)
2. **Files referenced in the review feedback** -- all files from inline comment `path` fields, plus any file paths mentioned in the comment body
3. **Files changed in the PR** (from Step 3) -- read selectively, prioritizing files related to the feedback

Use the Read tool for each file. If a file doesn't exist locally, note it may need to be pulled from the remote branch.

### Step 6: Analyze Feedback and Present to User

Parse the review into discrete action items. For each item, identify:
- **What**: The specific issue or suggestion
- **Where**: File path(s) and location(s) affected
- **Priority**: Critical / Medium / Low (based on reviewer's language and issue severity)
- **Clarity**: Whether the feedback is clear and actionable, or ambiguous

When the local code already reflects changes that appear to address a feedback item (e.g., the user partially addressed the review before running this skill), note it as "potentially already addressed" rather than omitting it.

Display the parsed action items to the user in this format before proceeding:

```
## Review Feedback Analysis

PR #<number>: <title>
Reviewer: <user>
Comment: <url>

### Action Items

1. [CRITICAL] <description>
   File: <path>:<line>
   Quote: "<reviewer's words>"

2. [MEDIUM] <description>
   File: <path>
   Quote: "<reviewer's words>"
   Note: Potentially already addressed in local code

3. [LOW] <description>
   File: <path>
   Quote: "<reviewer's words>"
```

### Step 7: Ask Clarifying Questions

If any feedback items are ambiguous, use AskUserQuestion to resolve them BEFORE entering plan mode. Present:
- The exact quote from the reviewer
- The possible interpretations
- Ask the user to choose or clarify

Skip this step if all items are clear and actionable.

### Step 8: Enter Plan Mode

Call `EnterPlanMode` to transition into plan mode. All context from Steps 1-7 is already in the conversation. In plan mode:

1. Write a plan file addressing each feedback item
2. Structure the plan with:
   - **Context section**: PR number, reviewer, comment URL
   - **Action items organized by priority**: Each with feedback quote, analysis, files to modify, specific changes, and testing approach
   - **Implementation order**: With dependency reasoning
   - **Verification section**: How to test the changes
3. Call `ExitPlanMode` for user approval

## Error Handling

| Scenario | Response |
|---|---|
| Empty `$ARGUMENTS` | AskUserQuestion with URL format examples (see Step 1) |
| Invalid URL format | Error with supported formats and examples |
| Non-github.com URL | `Error: Only github.com URLs are currently supported. GitHub Enterprise URLs are not handled.` |
| 404 from `gh api` | `Comment not found. It may have been deleted or the URL may be incorrect.` |
| `gh` not authenticated | `GitHub CLI not authenticated. Run 'gh auth login' first.` |
| Empty comment body (PR review) | Try fetching inline comments; if also empty, report: `Review body and inline comments are both empty. Nothing to address.` |
| Files referenced in review don't exist locally | Note in analysis, suggest pulling the branch |

## Examples

```bash
# Address feedback from a PR-level comment
/read-feedback-revise https://github.com/owner/repo/pull/42#issuecomment-123456

# Address feedback from an inline review comment
/read-feedback-revise https://github.com/owner/repo/pull/42#discussion_r789012

# Address feedback from a full review (fetches body + all inline comments)
/read-feedback-revise https://github.com/owner/repo/pull/42#pullrequestreview-345678
```

## Notes

- Steps 1-7 are strictly read-only -- no files are created, edited, or deleted until plan mode
- The skill transitions to plan mode (Step 8) so the user reviews and approves the implementation plan before any code changes
- For PR reviews (`#pullrequestreview-`), both the review body and all inline comments are fetched to capture the full feedback
- If the reviewer's feedback references files not in the PR diff, those files are still read for context
- This skill pairs naturally with `/push-pr-update` after implementing the approved plan
