#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import fairseq_cli.hydra_train as fairseq_hydra_train

from fairseq_easy_extend_cli.train import main as fee_pre_main
from fairseq_easy_extend.dataclass.initialize import add_defaults as fee_add_defaults
from fairseq_easy_extend.dataclass.initialize import hydra_init as fee_hydra_init
from fairseq_easy_extend.dataclass.configs import FEETextgenConfig

fairseq_hydra_train.pre_main = fee_pre_main
fairseq_hydra_train.add_defaults = fee_add_defaults
fairseq_hydra_train.hydra_init = fee_hydra_init
fairseq_hydra_train.FairseqConfig = FEETextgenConfig


def cli_main():
    fairseq_hydra_train.cli_main()


if __name__ == "__main__":
    fairseq_hydra_train.cli_main()
