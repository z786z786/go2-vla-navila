"""Phase 1 prompt contract: one builder shared by training and the evaluator.

The evaluator must import this module rather than re-deriving the format. A
prompt that differs between training and inference by even one token (a space
before the answer, a stripped quote) changes what the model conditions on and
fails silently.

Format = SmolVLM2's own chat template wrapped around NaVILA's native question.

  <|im_start|>User: Imagine you are a robot programmed for navigation tasks.
  You have been given a video of historical observations IMG\\nIMG\\n...(7 frames),
  and current observation IMG\\n. Your assigned task is: "<instruction>" Analyze
  this series of images to decide your next action, which could be turning left
  or right by a specific degree, moving forward a certain distance, or stop if
  the task is completed.<end_of_utterance>\\nAssistant:

  IMG = <fake_token_around_image><global-img><image>x64<fake_token_around_image>

Why each piece:
- Question text is copied verbatim from NaVILA-Bench scripts/vlm_server.py:126-131,
  including its "<image>\\n." oddity, because v3.1 #9 aligns with NaVILA natively.
- The chat wrapper and per-image block are SmolVLM2's own (chat_template.json and
  transformers processing_smolvlm._prompt_single_image). The pretrained model only
  ever saw images inside that block; bare vision embeddings with no delimiters are
  a format it never trained on.
- Single-image blocks, not image splitting: the model's processor would upscale a
  512x512 frame to 2048 and cut 16 crops (1088 tokens/frame). Video mode uses one
  64-token block per frame, which is what the throughput probe measured.
- The target ends with <end_of_utterance> (the generation eos_token_id, 49279) so
  greedy decoding terminates on its own instead of running to max_new_tokens.
- Instruction whitespace is stripped at both ends: every NaVILA record carries a
  trailing space inside the quotes, and evaluation instructions do not. Stripping
  here, in the shared builder, keeps train and eval byte-identical.
- The instruction alone is capped at tokenizer_max_length=160 (v3.1 #11). The
  template and the target are never truncated. Measured: training max is 203
  tokens (5 of 10,815 distinct instructions exceed 160); evaluation max is 137.
"""
from __future__ import annotations

from dataclasses import dataclass

HISTORY_FRAMES = 8
TOKENS_PER_FRAME = 64
MAX_INSTRUCTION_TOKENS = 160

IMAGE_TOKEN = "<image>"
FAKE_AROUND_IMAGE = "<fake_token_around_image>"
GLOBAL_IMAGE = "<global-img>"
END_OF_UTTERANCE = "<end_of_utterance>"
IM_START = "<|im_start|>"

# Frozen token ids of SmolVLM2-500M-Video-Instruct (snapshot 7b375e1b), checked at
# builder construction so a different tokenizer cannot slip in unnoticed.
EXPECTED_IDS = {IMAGE_TOKEN: 49190, FAKE_AROUND_IMAGE: 49189, GLOBAL_IMAGE: 49152,
                END_OF_UTTERANCE: 49279}

QUESTION_HEAD = ("Imagine you are a robot programmed for navigation tasks. You have been "
                 "given a video of historical observations ")
QUESTION_MID_HISTORY_SEP = "\n"               # after each history image block
QUESTION_MID = ", and current observation "
QUESTION_AFTER_CURRENT = "\n. Your assigned task is: \""
QUESTION_TAIL = ("\" Analyze this series of images to decide your next action, which could "
                 "be turning left or right by a specific degree, moving forward a certain "
                 "distance, or stop if the task is completed.")


def frames_to_pixel_values(frames_uint8):
    """uint8 [..., 3, 512, 512] frames -> SigLIP input in [-1, 1]; no resampling.

    Equals SmolVLM's rescale (1/255) + normalize (mean 0.5, std 0.5). The HF
    image processor is deliberately NOT used: even with do_image_splitting=False
    it resizes 512 -> size.longest_edge (2048) -> max_image_size (512), two
    resamples that are not an identity (up to 0.49 per pixel in normalized
    units on real frames, measured by scripts/p5_verify_native_forward.py).
    NaVILA frames and the NaVILA-Bench eval camera are both 512x512, so direct
    normalization needs no resize on either side. Training and the evaluator
    must both call this.
    """
    return frames_uint8.float() / 127.5 - 1.0 if not frames_uint8.is_floating_point() else frames_uint8


def navila_history_layout(n_frames: int, count: int = HISTORY_FRAMES) -> list[int | None]:
    """Which frames the model sees, oldest first; None marks a black padding frame.

    NaVILA's rule, identical on both sides of its pipeline:
      training  llava/mm_utils.py vlnce_frame_sampling (AnjieCheng/NaVILA, main)
      eval      NaVILA-Bench scripts/navila_eval.py:157-173
    With fewer than `count` frames, black frames are PREPENDED up to `count`;
    then 7 indices floor(i*(m-1)/7), i=0..6, over the padded list, plus its last
    frame. Floor, not round, and padding, not repetition: on the full training
    index the round/repeat rule this replaces picked a different frame set for
    310,772 of 353,894 records (87.8%), and repeated frames instead of black
    padding for the 15.3% with n_frames < 8.
    Integer floor division is exact, unlike np.linspace(..., dtype=int), and
    agrees with it here because i*(m-1)/7 is never within float error of an
    integer unless it is one.
    """
    if n_frames <= 0 or count < 2:
        raise ValueError("positive n_frames and count >= 2 required")
    pad = max(0, count - n_frames)
    m = n_frames + pad
    idx = [i * (m - 1) // (count - 1) for i in range(count - 1)] + [m - 1]
    return [None if j < pad else j - pad for j in idx]


@dataclass(frozen=True)
class EncodedPrompt:
    ids: list[int]
    instruction_tokens: int          # before truncation
    instruction_truncated: bool


class Phase1PromptBuilder:
    """Builds token ids directly; the template pieces are tokenized once.

    Pieces are tokenized separately, so BPE merges never cross a piece boundary
    (e.g. the opening quote and the instruction's first word stay separate
    tokens). That differs from tokenizing the rendered string in one pass, and it
    is fine only because training and inference both come through here.
    """

    def __init__(self, tokenizer, history_frames: int = HISTORY_FRAMES,
                 max_instruction_tokens: int = MAX_INSTRUCTION_TOKENS) -> None:
        if history_frames != HISTORY_FRAMES:
            raise ValueError(f"phase 1 freezes history_frames={HISTORY_FRAMES}")
        self.tok = tokenizer
        self.max_instruction_tokens = max_instruction_tokens
        ids = {t: tokenizer.convert_tokens_to_ids(t) for t in EXPECTED_IDS}
        if ids != EXPECTED_IDS:
            raise ValueError(f"tokenizer special ids {ids} != frozen {EXPECTED_IDS}")
        self.image_token_id = ids[IMAGE_TOKEN]
        self.eou_id = ids[END_OF_UTTERANCE]

        enc = lambda s: tokenizer(s, add_special_tokens=False)["input_ids"]
        image_block = ([ids[FAKE_AROUND_IMAGE], ids[GLOBAL_IMAGE]]
                       + [self.image_token_id] * TOKENS_PER_FRAME
                       + [ids[FAKE_AROUND_IMAGE]])
        head = enc(IM_START) + enc("User: " + QUESTION_HEAD)
        history = []
        for _ in range(history_frames - 1):
            history += image_block + enc(QUESTION_MID_HISTORY_SEP)
        self._prefix = (head + history + enc(QUESTION_MID) + image_block
                        + enc(QUESTION_AFTER_CURRENT))
        self._suffix = enc(QUESTION_TAIL) + [self.eou_id] + enc("\nAssistant:")
        n_img = sum(1 for t in self._prefix if t == self.image_token_id)
        if n_img != history_frames * TOKENS_PER_FRAME:
            raise AssertionError(f"{n_img} image tokens, expected "
                                 f"{history_frames * TOKENS_PER_FRAME}")

    @property
    def template_tokens(self) -> int:
        return len(self._prefix) + len(self._suffix)

    def prompt(self, instruction: str) -> EncodedPrompt:
        instr = self.tok(instruction.strip(), add_special_tokens=False)["input_ids"]
        n = len(instr)
        return EncodedPrompt(ids=self._prefix + instr[:self.max_instruction_tokens] + self._suffix,
                             instruction_tokens=n,
                             instruction_truncated=n > self.max_instruction_tokens)

    def target(self, action_text: str) -> list[int]:
        """Assistant answer tokens, supervised in full, eos included.

        The leading space matches SmolVLM2's template, which renders an assistant
        turn as "Assistant: <text><end_of_utterance>".
        """
        return self.tok(" " + action_text.strip(), add_special_tokens=False)["input_ids"] + [self.eou_id]

    def decode_answer(self, ids: list[int]) -> str:
        """Generated ids -> text for the parser, cut at the first eos."""
        if self.eou_id in ids:
            ids = ids[:ids.index(self.eou_id)]
        return self.tok.decode(ids, skip_special_tokens=True).strip()

    def render(self, instruction: str, action_text: str | None = None,
               collapse_images: bool = True) -> str:
        """Human-readable view of exactly what the model sees (for review only)."""
        ids = self.prompt(instruction).ids + (self.target(action_text) if action_text else [])
        text = self.tok.decode(ids, skip_special_tokens=False)
        if collapse_images:
            text = text.replace(IMAGE_TOKEN * TOKENS_PER_FRAME, f"<image>x{TOKENS_PER_FRAME}")
        return text
