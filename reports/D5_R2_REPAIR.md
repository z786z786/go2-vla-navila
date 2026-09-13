# D5-R2 — Zero-Frame Official Orientation Exit Repair

## Diagnosis

`short_vln_v1_1050` stopped before the first recorded expert action.  Its immutable M4 runner log contains the official planner guard event `roll=-0.3257`, `pitch=-0.6511`, followed by `M4 timeline STOP`; the strict pitch threshold is `0.6 rad`.  The attempt retains `collector_config.json`, an empty `steps.jsonl`, and no RGB frames.  The official planner remains unmodified at SHA-256 `20ea09b51287defead44d20a47730a38ba1ecf399e034c8e3acf13196a686432`.

The previous collector only finalized a timeline stop when frame records already existed.  Thus the official pre-action return produced no `summary.json`, and the outer D5 collector conservatively stopped as an infrastructure failure.

## Repair

- Capture the exact official large-orientation print event in the collection process and persist `terminal_event.json`.
- On the corresponding timeline stop, write a zero-frame terminal summary with reason `official_large_orientation_before_first_action`.
- Produce a dedicated zero-frame sanity record without fabricating action, plot, or video artifacts.
- Classify only verified zero-frame official exits as `expert_reset_reject`; leave every unrecognized stop or runtime failure blocking.
- Recover the existing `1050` attempt only from its unique runner log, config hash, empty evidence, conservative rounded threshold bound, and immutable planner hash.
- Use incrementing immutable manifest snapshots for repeated `--resume` calls.  Accepted episodes are never re-collected or overwritten.

## Expected Resume State

Before collection continues, D5-R2 will retain 28 accepted train episodes and mark `1050` as a tenth train rejection.  Only the missing right-turn training replacement is selected; then D5 proceeds to 30 accepted train episodes and 12 accepted seen-val episodes before conversion or training starts.

## Verification

- Unit coverage verifies strict threshold behavior, multi-frame and zero-frame evidence recovery, refusal of unverifiable evidence, repeated snapshot naming, classification, and report counts.
- The zero-frame OpenCV sanity tests run in the remote Isaac environment, where the dependency is part of the existing collection stack.
