# D5-R6 — Equal-exposure training acceptance repair

The D5 equal-exposure run completed all 43,000 requested steps, saved and reloaded `step_043000`, produced finite `1×50×3` actions, and retained a complete finite 43,000-row history. Its legacy M6 smoke result was nevertheless marked false solely because the acceptance expression required `steps == 2000`.

D5-R6 changes the live smoke acceptance to a minimum-step check plus complete finite history, checkpoint reload, and output-shape checks. For the already completed run, it preserves the original report byte-for-byte, writes a hash-bound independent acceptance audit, and migrates only the report PASS field with explicit provenance. No weight, processor, optimizer state, dataset, checkpoint, or training-history file is changed.

The post-training recovery entrypoint refuses missing audit evidence, substantial-normalizer-shift runs, partial output directories, and GPU free memory below 12,288 MiB. It resumes at offline evaluation and therefore does not repeat collection, conversion, or training.

The recovery entrypoint explicitly inherits the main D5 Hugging Face cache and offline-mode environment. This prevents checkpoint evaluation from consulting the default user cache or attempting network access.
