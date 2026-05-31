"""Stub — QLoRA com Unsloth (não executa).

Recomendado vs Ollama para treino: backward + mask na última resposta.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class QloraTrainConfig:
    model_path: Path
    train_json: Path
    output_dir: Path
    max_seq_length: int = 4096
    lora_r: int = 16
    lora_alpha: int = 32
    load_in_4bit: bool = True
    num_epochs: int = 3
    learning_rate: float = 2e-5
    # Treino: loss só no último turno assistant (equivalente a 6 msgs contexto + 7ª label)
    train_on_responses_only: bool = True


def prepare_dataset_stub(train_json: Path) -> None:
    """
    Implementação futura:
      - Ler messages[0:6] como prompt, messages[6] como completion
      - Ou usar dataset ShareGPT com mask_history no Unsloth/TRL
    """
    raise NotImplementedError


def run_qlora_train_stub(cfg: QloraTrainConfig) -> Path:
    """
    Implementação futura:
      from unsloth import FastLanguageModel
      model, tokenizer = FastLanguageModel.from_pretrained(..., load_in_4bit=True)
      model = FastLanguageModel.get_peft_model(...)
      trainer.train()
    """
    raise NotImplementedError
