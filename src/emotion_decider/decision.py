"""The trainable decision model: a frozen Qwen2.5-Omni thinker, LoRA on its language model, a pointer head.

`logits(clip, questions)` is the scorer interface the evaluation shares with the zero-shot baseline
(emotion_decider.zeroshot): one packed pass per clip, one logits vector per question, in each question's option order.
"""
import json
from pathlib import Path

import torch
from torch import nn

from .model import PointerHead
from .omni import batch_vision_attention, packed_hidden_batch
from .questions import delimiter_ids, encode_question

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# Optional encoder LoRA (same rank/alpha/dropout): config key -> thinker attribute. Targets are module-name suffixes,
# e.g. vision [attn.q, attn.k, attn.v, attn.proj, gate_proj, up_proj, down_proj, mlp.0, mlp.2] (mlp.* = merger),
# audio [q_proj, k_proj, v_proj, out_proj, fc1, fc2].
ENCODER_LORA = {"vision_targets": "visual", "audio_targets": "audio_tower"}
PREFIX, SUFFIX = "<|im_start|>user\n", "<|im_end|>\n"   # the clip state's text around the media


class DecisionModel(nn.Module):
    prefix, suffix = PREFIX, SUFFIX

    def __init__(self, thinker, tokenizer, lora=None, pointer_dim=256, gradient_checkpointing=False):
        """thinker: Qwen2_5OmniThinkerForConditionalGeneration (frozen here). lora: {rank, alpha, dropout, targets}
        or None. The vocabulary head is dropped: decisions never read it (~0.6 GB in bf16 for 3B)."""
        super().__init__()
        thinker.requires_grad_(False)
        thinker.lm_head = None
        batch_vision_attention(thinker)
        if lora:
            from peft import LoraConfig, TaskType, get_peft_model
            thinker.model = get_peft_model(thinker.model, LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION, r=lora["rank"], lora_alpha=lora["alpha"],
                lora_dropout=lora["dropout"], target_modules=lora.get("targets", LORA_TARGETS)))
        self.encoder_lora = [attr for key, attr in ENCODER_LORA.items() if lora and lora.get(key)]
        for key, attr in ENCODER_LORA.items():
            if attr in self.encoder_lora:
                from peft import LoraConfig, inject_adapter_in_model
                encoder = getattr(thinker, attr)
                inject_adapter_in_model(LoraConfig(r=lora["rank"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                                                   target_modules=lora[key]), encoder)
                for n, p in encoder.named_parameters():   # fp32 adapters, as get_peft_model makes for the LM
                    if "lora_" in n:
                        p.data = p.data.float()
        if gradient_checkpointing:
            text = thinker.model.get_base_model() if lora else thinker.model
            text.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.thinker, self.tokenizer = thinker, tokenizer
        self.delimiters = delimiter_ids(tokenizer)
        self.head = PointerHead(thinker.config.text_config.hidden_size, dp=pointer_dim).float()
        self.lora_config = lora

    @property
    def device(self):
        return self.head.q.weight.device

    @classmethod
    def from_pretrained(cls, name, tokenizer, revision=None, dtype=torch.bfloat16, device="cpu", attn="sdpa", **kw):
        from transformers import Qwen2_5OmniThinkerForConditionalGeneration
        thinker = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            name, revision=revision, dtype=dtype, attn_implementation=attn)
        return cls(thinker, tokenizer, **kw).to(device)

    def logits_batch(self, clips, question_lists):
        """Several clips in one forward pass. question_lists[k]: API-format question dicts for clip k ->
        per clip, a list of fp32 logit vectors (option order)."""
        branch_lists = [[encode_question(self.tokenizer, q, self.delimiters) for q in qs] for qs in question_lists]
        hidden = packed_hidden_batch(self.thinker, [c.to(self.device) for c in clips], branch_lists,
                                     encoders_grad=bool(self.encoder_lora))
        return [[self.head(h[b.decide].float(), h[b.opts].float()) for h, b in zip(hs, branches)]
                for hs, branches in zip(hidden, branch_lists)]

    def logits(self, clip, questions):
        """questions: list of API-format question dicts -> list of fp32 logit vectors (option order)."""
        return self.logits_batch([clip], [questions])[0]

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    # --- checkpoints: adapter + head only (the base is referenced by name/revision, never copied)

    def state(self):
        from peft import get_peft_model_state_dict
        adapter = get_peft_model_state_dict(self.thinker.model) if self.lora_config else {}
        state = {"adapter": {k: v.detach().cpu() for k, v in adapter.items()},
                 "head": {k: v.detach().cpu() for k, v in self.head.state_dict().items()}}
        if self.encoder_lora:
            state["encoder_adapter"] = {f"{attr}.{k}": v.detach().cpu() for attr in self.encoder_lora
                                        for k, v in getattr(self.thinker, attr).state_dict().items() if "lora_" in k}
        return state

    def load_state(self, state):
        if self.lora_config:
            from peft import set_peft_model_state_dict
            result = set_peft_model_state_dict(self.thinker.model, state["adapter"])
            if getattr(result, "unexpected_keys", None):
                raise ValueError(f"unexpected adapter keys: {result.unexpected_keys[:3]}")
        saved = state.get("encoder_adapter", {})
        for attr in self.encoder_lora:
            encoder = getattr(self.thinker, attr)
            want = {k for k in encoder.state_dict() if "lora_" in k}
            sub = {k[len(attr) + 1:]: v for k, v in saved.items() if k.startswith(attr + ".")}
            if set(sub) != want:
                raise ValueError(f"{attr} adapter keys don't match: missing {sorted(want - set(sub))[:3]}, "
                                 f"unexpected {sorted(set(sub) - want)[:3]}")
            encoder.load_state_dict(sub, strict=False)
        if saved and not self.encoder_lora:
            raise ValueError("checkpoint has encoder adapters but the model has no encoder LoRA")
        self.head.load_state_dict(state["head"])

    def save(self, path, meta=None):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state(), path / "decision.pt")
        (path / "decision.json").write_text(json.dumps({"lora": self.lora_config, **(meta or {})}, indent=1))
