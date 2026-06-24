---
description: Trigger this skill ONLY when the user says "Commit these changes". It runs bash commands to safely append a changelog to handover_notes.md.
---

- **Global Mandate:** **ALL generated content, summaries, and explanations MUST be written in Traditional Chinese (繁體中文). Do NOT use English or Simplified Chinese.**

- **Step 0: Time Sync:** - Execute `date +%Y-%m-%d`. Memorize as `TODAY`.

- **Step 1: Header Logic:**
  - Execute `tail -n 20 ./handover_notes.md`.
  - **IF** output contains `## <TODAY>`, skip date header.
  - **ELSE**, write new `## <TODAY>` header first.

- **Step 2: Integrity Guard:**
  - **STRICT BAN:** NEVER use `edit()` tool on `handover_notes.md`.
  - **MANDATORY:** Use `bash` with `cat <<EOF >> ./handover_notes.md` to append content.

- **Step 3: Diff Verification:** - Before writing, run `git show <hash>` and `git diff <hash>^!` to ensure absolute accuracy of logic analysis.

- **Step 4: Format Mandate:**
  - **New Day Header (If Step 1 ELSE):**
    ```markdown
    ## YYYY-MM-DD
    ```
  - **Content Entry (Always append):**
    ```markdown
    ### <N>. <Title> (Commit `hash`)
    <One sentence high-level summary>

    * **[問題背景]**
        <Explain the pain point or bug.>
    * **[代碼指引]**
        <Modified file and function names.>
    * **[邏輯解析]**
        <Step-by-step logic explanation based on git diff.>
    ```
