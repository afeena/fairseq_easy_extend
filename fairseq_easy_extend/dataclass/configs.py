from dataclasses import dataclass, field
from typing import Any, Optional

from fairseq.dataclass.configs import (
    FairseqDataclass,
    CommonEvalConfig,
    DistributedTrainingConfig,
    DatasetConfig,
    OptimizationConfig,
    CheckpointConfig,
    FairseqBMUFConfig,
    EvalLMConfig,
    InteractiveConfig,
    EMAConfig,
    CommonConfig,
    GenerationConfig
)
from omegaconf import MISSING



@dataclass
class FEETextgenConfig(FairseqDataclass):
    common: CommonConfig = CommonConfig()
    common_eval: CommonEvalConfig = CommonEvalConfig()
    distributed_training: DistributedTrainingConfig = DistributedTrainingConfig()
    dataset: DatasetConfig = DatasetConfig()
    optimization: OptimizationConfig = OptimizationConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    bmuf: FairseqBMUFConfig = FairseqBMUFConfig()
    generation: GenerationConfig = GenerationConfig()
    eval_lm: EvalLMConfig = EvalLMConfig()
    interactive: InteractiveConfig = InteractiveConfig()
    model: Any = MISSING
    task: Any = None
    criterion: Any = None
    optimizer: Any = None
    lr_scheduler: Any = None
    scoring: Any = None
    bpe: Any = None
    tokenizer: Any = None
    ema: EMAConfig = EMAConfig()