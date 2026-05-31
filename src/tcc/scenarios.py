"""Cenários experimentais alinhados à metodologia do TCC (I0, RAG, SFT, SFT+RAG)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ScenarioId(str, Enum):
    I0 = "i0"
    RAG = "rag"
    SFT = "sft"
    SFT_RAG = "sft_rag"


@dataclass(frozen=True)
class Scenario:
    id: ScenarioId
    use_rag: bool
    use_sft_checkpoint: bool
    worfbench_few_shot_flag: bool
    description: str

    @property
    def pred_suffix(self) -> str:
        return self.id.value


def load_scenarios(cfg: dict[str, Any]) -> dict[ScenarioId, Scenario]:
    raw = cfg.get("scenarios", {})
    mapping: dict[ScenarioId, Scenario] = {}
    for sid in ScenarioId:
        block = raw.get(sid.value, {})
        mapping[sid] = Scenario(
            id=sid,
            use_rag=bool(block.get("use_rag", False)),
            use_sft_checkpoint=bool(block.get("use_sft_checkpoint", False)),
            worfbench_few_shot_flag=bool(block.get("worfbench_few_shot_flag", False)),
            description=str(block.get("description", sid.value)),
        )
    return mapping


ALL_SCENARIOS = list(ScenarioId)
