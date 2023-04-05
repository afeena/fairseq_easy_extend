# initialize hydra

from fairseq_easy_extend.dataclass.initialize import hydra_init

hydra_init()

import fairseq_easy_extend.criterions  # noqa
import fairseq_easy_extend.models  # noqa
import fairseq_easy_extend.modules  # noqa
import fairseq_easy_extend.tasks  # noqa
import fairseq_easy_extend.dataclass  # noqa
import fairseq_easy_extend.logging  # noqa

import fairseq.distributed  # noqa
import fairseq.optim  # noqa
import fairseq.optim.lr_scheduler  # noqa
import fairseq.pdb  # noqa
import fairseq.scoring  # noqa
import fairseq.tasks  # noqa
import fairseq.token_generation_constraints  # noqa

import fairseq.benchmark  # noqa
import fairseq.model_parallel  # noqa
