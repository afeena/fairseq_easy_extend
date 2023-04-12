# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This file implements:
Ghazvininejad, Marjan, et al.
"Constant-time machine translation with conditional masked language models."
arXiv preprint arXiv:1904.09324 (2019).
"""
import argparse
import collections
from dataclasses import field, dataclass

import omegaconf
from fairseq.models import register_model
from fairseq.models.nat import CMLMNATransformerModel
from fairseq.models.transformer import TransformerConfig

from fairseq_easy_extend.dataclass.utils import gen_parser_from_dataclass
from fairseq_easy_extend.dataclass.utils import convert_omegaconf_to_namesapce


@dataclass
class CMLMTransformerConfig(TransformerConfig):
    # --- special arguments ---
    sg_length_pred: bool = field(
        default=False,
        metadata={
            "help": "stop gradients through length"
        }
    )
    pred_length_offset: bool = field(
        default=False,
        metadata={
            "help": "predict length offset"
        },
    )
    length_loss_factor: float = field(
        default=0.1,
        metadata={"help": "loss factor for length"},
    )
    ngram_predictor: int = field(
        default=1, metadata={"help": "maximum iterations for iterative refinement."},
    )
    src_embedding_copy: bool = field(
        default=False,
        metadata={
            "help": "copy source embeddings"
        },
    )
    label_smoothing: float = field(default=0.1, metadata={"help": "label smoothing"})

@register_model("cmlm_transformer_base", dataclass=CMLMTransformerConfig)
class BaseCMLMNATransformerModel(CMLMNATransformerModel):

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""
        # we want to build the args recursively in this case.
        gen_parser_from_dataclass(
            parser, CMLMTransformerConfig(), delete_default=False, with_prefix=""
        )

    @classmethod
    def build_model(cls, cfg, task):
        if isinstance(cfg, omegaconf.DictConfig):
            cfg = convert_omegaconf_to_namesapce(cfg)
        model = super().build_model(cfg, task)
        return model