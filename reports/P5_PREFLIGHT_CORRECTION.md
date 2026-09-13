# P5 preflight correction

Status: direction confirmation required. No P5 training has run.

The recent continuation followed superseded v2 guidance. Controlling v3.1/v3.4
specifies VLM LoRA + lm_head text generation in phase 1, duration-aware
NavCommand, and expert overfit only in phase 2.

`p5_dev100.json` and `p5_dev100.jsonl` are NOT benchmark dev100. They contain
100 sampled training decision records from 61 training scenes, not 100 official
evaluation episodes selected from the 1077-episode benchmark. Retained for audit
only; do not use for evaluation or claim they are ten overfit episodes.

The initial p5_contract assumed the wrong numeric label order. The index uses
STOP=0 and right45=9. It now delegates to the existing exact text parser;
previous target counts are invalid.

P4 throughput measured forward-only synthetic text embeddings under no_grad,
without backward/optimizer or timed data loading. Its rates and maximum batch
are not validated training throughput or training batch limits.
