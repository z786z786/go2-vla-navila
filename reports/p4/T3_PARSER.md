# P4-T3 deterministic language parser

This slice maps model text to a validated `NavCommand` using only a frozen
vocabulary, regular-expression normalization, and a finite-state branch. It
does not import an LLM client and performs no file, HTTP, socket, or network
operation. The default is `fallback_policy="layered"` at
`actions/language_parser.py:175`.

## Contract

`NavCommand(vx, vy, wz, hold_steps, stop)` enforces `vx∈[0,0.5]`, `vy=0`,
`|wz|≤pi/6` (`0.5235987755982988`), non-stop `hold_steps∈{25,50,75}`, and
stop `(0,0,0,0,True)`. The 50 Hz period is 0.02 s, so 0.5/1.0/1.5 s map
exactly to 25/50/75 steps. The ten mapping rows and machine-readable policy
contract are in [t3_parser_contract.json](t3_parser_contract.json).

## Policy behavior on a total miss

For an input with no vocabulary keyword (for example `unrelated words`), all
three results have `parse_failure=true`, `vy=0.0`, `wz=0.0`, `hold_steps=25`,
and `stop=false`; `layered` and `brake` return `vx=0.0`, while `native`
returns `vx=0.5` (native's forward-25 fallback).

## Required native quirk

`turn left 145 degree` is `partial_match` under every policy and returns
`NavCommand(vx=0.0, vy=0.0, wz=+0.5235987755982988, hold_steps=75, stop=False)`.
This reproduces native's 45-degree result because the implementation checks the
substring `"45" in text` before `"30"` and `"15"`; no silent tightening was
made. A brake policy changes only an actual no-number fallback, not a recognized
native substring.

## Native line-by-line comparison

The reference is `eval_utils.py:get_vel_command`, lines 57–85, SHA-256
`c42b832a6e37b786f8c42e42f9ba790ca88fbb4fcf1cad8822d9695abc1e5bb0`.

| Native line(s) | Native operation | This implementation | Difference / rationale |
|---|---|---|---|
| 57 | Define `get_vel_command(text)` | `LanguageParser.parse(text)` (196–233) | Returns `ParseResult` plus validated `NavCommand` and tier counters. |
| 58–59 | Lowercase and test `turn left` first | `_native_branch` (107–119) | Same branch priority; exact vocabulary normalization is an added front-end. |
| 60–66 | `45`, then `30`, then `15`, else left 15 at 0.5 s | `_native_branch_command` (127–136) | Same order, durations become 75/50/25 steps; substring quirk retained. |
| 67 | Test `turn right` second | `_native_branch` (113–114) | Same. |
| 68–74 | Right `45/30/15`, else right 15 | `_native_branch_command` (137–144) | Same order and signs; duration converted to steps. |
| 75–76 | Test `move forward` or `move` | `_native_branch` (115) | Same, including bare `move`. |
| 77–82 | Forward `75/50/25`, else forward 25 | `_native_branch_command` (145–152) | Same order and velocity; duration converted to steps. Brake can replace fallback. |
| 83 | `stop` → zero velocity, 0.0 s | `_native_branch_command` (153–154) | Same command; non-canonical stop text is classified partial. |
| 84–85 | No keyword → forward 25 | `_native_branch_command` (155–157) | `native` preserves it; `layered` (default) and `brake` intentionally emit limiter-brake zero velocity. |
| — | No classification/counters or validation | `ParseCounters` (54–88), `NavCommand` | Dispatch-required additions: exact/partial/total-miss accounting and domain checks. |

## Deviations from native

The verbatim machine-readable section is the `deviations_from_native` array in
`t3_parser_contract.json`: layered default total-miss braking, brake-policy
fallback braking, normalized exact-input acceptance, and the object/time-step
adapter. The native 45-substring quirk is explicitly **not** a deviation.

## Verification

Pytest is not installed in this CPU-only environment. The exact fallback command
used was:

```text
python3 -m unittest discover -s tests/actions -p 'test_*.py' -v
```

Its complete output was:

```text
test_01_all_ten_frozen_vocabulary_entries (test_language_parser.TestLanguageParserContract.test_01_all_ten_frozen_vocabulary_entries) ... ok
test_02_surface_form_variants_are_exact (test_language_parser.TestLanguageParserContract.test_02_surface_form_variants_are_exact) ... ok
test_03_native_145_substring_quirk (test_language_parser.TestLanguageParserContract.test_03_native_145_substring_quirk) ... ok
test_04_native_keyword_priority (test_language_parser.TestLanguageParserContract.test_04_native_keyword_priority) ... ok
test_05_move_without_forward (test_language_parser.TestLanguageParserContract.test_05_move_without_forward) ... ok
test_06_total_misses_and_tiered_counters (test_language_parser.TestLanguageParserContract.test_06_total_misses_and_tiered_counters) ... ok
test_07_out_of_range_numbers (test_language_parser.TestLanguageParserContract.test_07_out_of_range_numbers) ... ok
test_08_stop_variants (test_language_parser.TestLanguageParserContract.test_08_stop_variants) ... ok
test_09_duration_to_hold_step_boundaries (test_language_parser.TestLanguageParserContract.test_09_duration_to_hold_step_boundaries) ... ok
test_10_emitted_commands_stay_in_domain (test_language_parser.TestLanguageParserContract.test_10_emitted_commands_stay_in_domain) ... ok

----------------------------------------------------------------------
Ran 10 tests in 0.003s

OK
```
