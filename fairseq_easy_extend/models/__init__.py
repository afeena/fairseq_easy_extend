# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""isort:skip_file"""

import os

import fairseq.models
from fairseq.dataclass import FairseqDataclass


def build_model(cfg: FairseqDataclass, task, from_checkpoint=False):
    return fairseq.models.build_model(cfg, task, from_checkpoint)


def register_model(name, dataclass=None):
    return fairseq.models.register_model(name, dataclass)


def import_models(models_dir, namespace):
    fairseq.models.import_models(models_dir, namespace)


# automatically import any Python files in the models/ directory
models_dir = os.path.dirname(__file__)
import_models(models_dir, "fairseq_easy_extend.models")
