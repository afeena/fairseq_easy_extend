#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Train a new model on one or across multiple GPUs.
"""
import fairseq_cli.train as train
from fairseq.dataclass.configs import FairseqConfig


def main(cfg: FairseqConfig) -> None:
    train.main(cfg)


if __name__ == "__main__":
    train.cli_main()
