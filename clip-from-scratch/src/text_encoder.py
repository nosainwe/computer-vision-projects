"""
text_encoder.py
---------------
Text encoder built on top of sihab/slm-1.0 (1.5B causal LM),
quantized to 4-bit NF4 with bitsandbytes and fine-tuned via QLoRA.

Based on:
  - Hu et al. (2021) "LoRA: Low-Rank Adaptation of Large Language Models"
    https://arxiv.org/abs/2106.09685
  - Dettmers et al. (2023) "QLoRA: Efficient Finetuning of Quantized LLMs"
    https://arxiv.org/abs/2305.14314

Key design choice: causal LM with last-token pooling.
  The model is not bidirectional, so we pool the last non-padding token
  rather than a [CLS] token. This is a reasonable approximation and
  avoids the need for a separate encoder architecture.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, PeftModel


BASE_MODEL_ID = "sihab/slm-1.0"


# ---------------------------------------------------------------------------
# QLoRA Configuration
# ---------------------------------------------------------------------------

def get_bnb_config() -> BitsAndBytesConfig:
    # NF4 is specifically designed for normally distributed weights — which LLM weights are
    # double quantization quantizes the quantization constants themselves, saves ~0.4 bits/param
    # bfloat16 compute keeps matmuls numerically stable while weights stay in 4-bit storage
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def get_lora_config(rank: int = 16, alpha: int = 32,
                    dropout: float = 0.05) -> LoraConfig:
    # targeting all attention projections and MLP gate/up/down — standard for causal LMs
    # FEATURE_EXTRACTION task type because we're pulling hidden states, not generating text
    # bias="none" is standard — adding bias to LoRA layers doesn't help much and adds params
    return LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )


# ---------------------------------------------------------------------------
# Text Encoder
# ---------------------------------------------------------------------------

class TextEncoder(nn.Module):
    """
    Wraps sihab/slm-1.0 (1.5B causal LM) with QLoRA adapters.

    Forward pass:
        tokens -> QLoRA-adapted LM -> last-token hidden state -> (B, hidden_dim)
    """

    def __init__(
        self,
        model_id: str = BASE_MODEL_ID,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        max_length: int = 77,        # matches original CLIP token limit
    ):
        super().__init__()
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # causal LMs don't have a pad token by default — using eos as pad is the standard workaround
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading {model_id} in 4-bit NF4...")
        base = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=get_bnb_config(),
            device_map="auto",          # lets accelerate decide GPU/CPU placement automatically
            trust_remote_code=True,
        )
        # use_cache caches key/value states for generation — incompatible with gradient checkpointing
        base.config.use_cache = False

        # wrapping base with LoRA — only adapter weights are trainable, base stays frozen in 4-bit
        self.model = get_peft_model(base, get_lora_config(lora_rank, lora_alpha, lora_dropout))
        self.model.print_trainable_parameters()   # sanity check: should be a small % of total

        self.hidden_dim = base.config.hidden_size

    def tokenize(self, texts: list[str], device: torch.device) -> dict:
        # padding to max_length so all sequences in a batch have the same shape
        # truncation=True handles captions longer than 77 tokens — rare but possible
        return self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids:      (B, seq_len)
            attention_mask: (B, seq_len)  — 1 for real tokens, 0 for padding

        Returns:
            (B, hidden_dim) — last non-padding token hidden state
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,  # need this to access intermediate layers, not just logits
            return_dict=True,
        )

        # last layer hidden states — this is what carries the semantic representation
        last_hidden = outputs.hidden_states[-1]     # (B, seq_len, hidden_dim)

        # last-token pooling: sum attention mask to find where each sequence ends
        # -1 converts length to 0-indexed position of the last real token
        seq_lengths = attention_mask.sum(dim=1) - 1   # (B,)
        batch_idx   = torch.arange(last_hidden.size(0), device=last_hidden.device)
        pooled      = last_hidden[batch_idx, seq_lengths]   # (B, hidden_dim)

        return pooled

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # WARNING: do NOT use torch.save() on the full model — 4-bit quantized
    # weights don't serialize cleanly through torch. always save adapters only.
    # ------------------------------------------------------------------

    def save(self, path: str):
        # saving adapters + tokenizer only — base model is re-quantized fresh at load time
        # this keeps checkpoints small and avoids the quantization serialization issue
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"LoRA adapters saved to {path}")

    @classmethod
    def load(cls, adapter_path: str, model_id: str = BASE_MODEL_ID,
             **kwargs) -> "TextEncoder":
        # correct QLoRA reload pattern: quantize a fresh base, then mount saved adapters on top
        # don't try to deserialize the quantized weights directly — re-quantize from fp16 each time
        instance = cls.__new__(cls)
        super(TextEncoder, instance).__init__()

        instance.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        if instance.tokenizer.pad_token is None:
            instance.tokenizer.pad_token = instance.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=get_bnb_config(),
            device_map="auto",
            trust_remote_code=True,
        )
        # PeftModel.from_pretrained loads and attaches the saved LoRA weights onto the fresh base
        instance.model      = PeftModel.from_pretrained(base, adapter_path)
        instance.hidden_dim = base.config.hidden_size
        instance.max_length = kwargs.get("max_length", 77)

        return instance
