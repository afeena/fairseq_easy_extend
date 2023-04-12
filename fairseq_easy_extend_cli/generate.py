#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""
from fairseq_cli.generate import cli_main as fairseq_generate_cli_main
from fairseq_easy_extend.dataclass.utils import convert_namespace_to_omegaconf
from fairseq_easy_extend import options
fairseq_generate_cli_main.convert_namespace_to_omegaconf = convert_namespace_to_omegaconf
fairseq_generate_cli_main.options = options

def cli_main():
    fairseq_generate_cli_main()
