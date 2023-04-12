#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""
import fairseq_cli.train as train
from fairseq_easy_extend.dataclass.configs import FEETextgenConfig as FairseqConfig
from fairseq_easy_extend.dataclass.utils import convert_namespace_to_omegaconf as fee_convert_namespace_to_omegaconf
from fairseq_easy_extend.dataclass.initialize import add_defaults as fee_add_defaults

train.add_defaults = fee_add_defaults
train.convert_namespace_to_omegaconf = fee_convert_namespace_to_omegaconf

def main(cfg: FairseqConfig) -> None:
    train.main(cfg)


if __name__ == "__main__":
    train.cli_main()
