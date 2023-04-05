#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import fairseq_cli.hydra_train as fairseq_hydra_train

from fairseq_easy_extend_cli.train import main as cdgm_pre_main

fairseq_hydra_train.pre_main = cdgm_pre_main


def cli_main():
    fairseq_hydra_train.cli_main()


if __name__ == "__main__":
    fairseq_hydra_train.cli_main()
