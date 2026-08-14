# Selecting tests for one module

A covering set is the reviewed list of tests used for one target Python module.
Use it to keep a mutation run small without missing the module's behavior.
Searching for likely tests is only a starting point. The coverage report must
show that the selected tests execute the required percentage of the exact
target file.

## 1. Find candidate tests

Start with evidence that another developer can review:

1. Include `Tensile/Tests/unit/test_<Module>.py` when it exists.
2. Search unit tests and tests that record current behavior for imports or
   references to the target module.
3. Include existing test directories whose fixtures or public API calls reach
   the target indirectly, even when they do not import it.
4. Reuse a previously committed test selection when it still applies. Explain
   every addition or removal.

Text search can miss aliases, re-exports, multiline imports, fixtures, and
imports performed while the program runs. An `rg` or grep result is not a
complete test selection. Read the selected tests and the target module
together.

## 2. Run only the reviewed tests

Run pytest in the Linux or Windows Subsystem for Linux (WSL) mutation
environment. List every selected path explicitly. For example:

```bash
PROJ=/work/projects/hipblaslt/tensilelite
OUT=work/mutation/covering/utilities
mkdir -p "$OUT"

set +e
docker exec -e PYTHONPATH="$PROJ" -w "$PROJ" tl-mut pytest -p no:cacheprovider -m unit --cov=Tensile/Common --cov-report=term-missing --cov-fail-under=0 Tensile/Tests/unit/characterization/CommonUtilities >"$OUT/coverage.log" 2>&1
rc=$?
set -e
```

Change the `--cov` path and test list for the target. Use a package path, not a
dotted Python module, as the coverage source. Earlier runs observed rocisa
unload/re-import failures with a dotted module.

The command sets `--cov-fail-under=0` because the repository-wide percentage
does not apply to this focused run. This does not remove the coverage
requirement. The next step checks the exact target-file row independently.

## 3. Stop unless every check passes

Require all four conditions:

1. Pytest exits with status `0`.
2. The coverage report contains the exact target file, such as
   `Tensile/Common/Utilities.py`.
3. That file's executed-line percentage meets the reviewed threshold. Use 80%
   unless the slice records another value.
4. The selected list contains at least one explicit test path and is not an
   accidental full-suite fallback.

Do not use the package `TOTAL` row when the target file is missing. A missing
target row means **Deferred**, even when the total package percentage is high.

## 4. Save the decision

Write `covering-set.json`, or an equivalent record, in the slice output
directory:

```json
{
  "module": "Tensile/Common/Utilities.py",
  "source_sha": "<commit>",
  "selected": [
    "Tensile/Tests/unit/characterization/CommonUtilities"
  ],
  "command": "<exact command>",
  "exit_code": 0,
  "target_row": "<verbatim coverage row>",
  "coverage_percent": 100.0,
  "threshold": 80.0,
  "status": "ok",
  "reason": "exact target row meets threshold"
}
```

Use `status: "defer"` when pytest fails, the target row is missing, the
selection is empty, or coverage is below the threshold. Record the actual
reason. Do not replace missing data with zero or with the package total.

## 5. Measure again when inputs change

Repeat this validation when:

- the target module changes in a way that may affect behavior;
- a selected test or fixture moves or changes;
- the base branch changes relevant imports or validation;
- the Python, coverage, or pytest environment changes; or
- a mutation run unexpectedly reports `no tests`.

The committed selection records which tests were chosen. A new measurement
shows whether they still execute enough of the target file.
