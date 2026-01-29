---
name: experiment
description: Create or document a research experiment following the hypothesis, results, learnings structure
argument-hint: [number-name] [create|results|learnings]
disable-model-invocation: true
allowed-tools: Read, Write, Edit, Glob, Bash(mkdir *)
---

Manage experiments in `experiments/` following CLAUDE.md research documentation rules.

## Commands

### Create new experiment: `/experiment NNN-name create`

1. Find next available experiment number by checking existing folders in `experiments/`
2. Create `experiments/NNN-name/` folder
3. Create `hypothesis.md` from template with sections:
   - Date
   - Hypothesis (what we're testing)
   - Background (context and prior work)
   - Experimental Design (control, treatment, metrics)
   - Expected Results
   - Risks and mitigations
   - Success Criteria
4. Remind user: "Write hypothesis BEFORE running experiment"

### Document results: `/experiment NNN results`

1. Read `experiments/NNN-*/hypothesis.md` to understand what was tested
2. Create `results.md` with:
   - Actual outcomes
   - Key metrics (SSIM, depth, RGB, view consistency)
   - What worked vs what didn't
   - Include specific numbers and comparisons

### Document learnings: `/experiment NNN learnings`

1. Read hypothesis and results
2. Create `learnings.md` with:
   - Key insights discovered
   - Bugs found and how they were fixed
   - Optimal hyperparameters (if discovered)
   - Approaches that failed and why
3. Update `experiments/README.md` summary table with result

## Templates

Use templates from this skill's templates/ folder:

- [hypothesis.md](templates/hypothesis.md)
- [results.md](templates/results.md)
- [learnings.md](templates/learnings.md)

## Example Usage

```
/experiment 016-attention-pooling create
/experiment 016 results
/experiment 016 learnings
```

## Important Rules (from CLAUDE.md)

1. **Every experiment** MUST have a folder in `experiments/`
2. Write `hypothesis.md` BEFORE starting the experiment
3. **Every discovery** MUST be recorded in experiment docs
4. **Validate before expensive training** - generate small sample, inspect manually
5. Failed experiments should fail FAST and CHEAP
