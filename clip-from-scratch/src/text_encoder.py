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
    """
    4-bit NF4 quantization config.
    NF4 (NormalFloat4) is information-theoretically optimal for
    normally distributed weights — which most LLM weights are.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,   # nested quantization saves ~0.4 bits/param
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def get_lora_config(rank: int = 16, alpha: int = 32,
                    dropout: float = 0.05) -> LoraConfig:
    """
    LoRA adapter config.
    We target the attention projection matrices (q, k, v, o) and
    the MLP layers — standard practice for causal LMs.
    """
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
        tokens → QLoRA-adapted LM → last-token hidden state → (B, hidden_dim)
    """

    def __init__(
        self,
        model_id: str = BASE_MODEL_ID,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        max_length: int = 77,        # matches original CLIP
    ):
        super().__init__()
        self.max_length = max_length

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model in 4-bit
        print(f"Loading {model_id} in 4-bit NF4...")
        base = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=get_bnb_config(),
            device_map="auto",
            trust_remote_code=True,
        )
        base.config.use_cache = False   # incompatible with gradient checkpointing

        # Wrap with LoRA adapters
        self.model = get_peft_model(base, get_lora_config(lora_rank, lora_alpha, lora_dropout))
        self.model.print_trainable_parameters()

        self.hidden_dim = base.config.hidden_size

    def tokenize(self, texts: list[str], device: torch.device) -> dict:
        """Tokenize a list of strings, pad/truncate to max_length."""
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
            output_hidden_states=True,
            return_dict=True,
        )

        # Last-token pooling: find the position of the last real token
        # for each sequence and extract its hidden state.
        last_hidden = outputs.hidden_states[-1]     # (B, seq_len, hidden_dim)

        # Index of last non-padding token per sequence
        seq_lengths = attention_mask.sum(dim=1) - 1  # (B,)
        batch_idx = torch.arange(last_hidden.size(0), device=last_hidden.device)
        pooled = last_hidden[batch_idx, seq_lengths]  # (B, hidden_dim)

        return pooled

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ⚠️  DO NOT use torch.save() on the full model — the 4-bit
    #     quantized weights don't serialize cleanly.
    #     Always save/load adapters separately.
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save LoRA adapters only. Base model is reloaded fresh at load time."""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"LoRA adapters saved to {path}")

    @classmethod
    def load(cls, adapter_path: str, model_id: str = BASE_MODEL_ID,
             **kwargs) -> "TextEncoder":
        """
        Reload by quantizing a fresh base and mounting saved adapters on top.
        This is the correct pattern for QLoRA checkpoints.
        """
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
        instance.model = PeftModel.from_pretrained(base, adapter_path)
        instance.hidden_dim = base.config.hidden_size
        instance.max_length = kwargs.get("max_length", 77)

        return instance
