# DGMs for text generation

## Requirements

This repository is a template for extending a [fairseq](https://github.com/facebookresearch/fairseq) for training continuous
seq2seq models.
Please follow the installation instruction of `fairseq`.


## Quick Start

```bash

git clone https://github.com/ltl-uva/cdgm_textgen/
cd fairseq_easy_extend

#train the model
python  ~/develop/fairseq_easy_extend/train.py --config-dir <config_dir> \
--config-name <config_name> <any additional hydra parameters>

#decode and obtain results

python decode.py path_to_data_folder --task name_task --source-lang src_lang --target-lang tgt_lang \
--path path_to_the_model_checkpoint

```



