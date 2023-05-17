# Reinforcement Learning for Non-Autoregressive Neural Machine Translation

## Requirements

This repository is an example of [fairseq](https://github.com/facebookresearch/fairseq) extension for NLP2 course. 
Please follow the installation instruction of `fairseq`

0. Get access to project drive (ask your TA)
1. Read project description pdf NLP2_ET_2023.pdf and suggested papers
2. Follow GroupC-fairseq notebook in order to get familiar with running model training and evaluation
   1. data for training is in iwslt14 folder
   2. you have pretrained checkpoint at checkpoint_best.pt
   3. if you are not familiar with NMT you can read https://evgeniia.tokarch.uk/blog/neural-machine-translation/
   4. Some notes on fairseq extension https://evgeniia.tokarch.uk/blog/extending-fairseq-incomplete-guide/
3. Objective implementation:
   1. check `rl_criterion.py` in `criterion` folder it gives you a hint how to start working on your objective
   2. Nice explanation of RL for NMT https://www.cl.uni-heidelberg.de/statnlpgroup/blog/rl4nmt/
   3. You can pick any metric and import any library of your choice
4. Run the training with your new objective function:
   1. You can strat fine-tuning to get better/faster results, find `checkpoint_best.pt` in the drive
   2. set `criterion._name` to the name of your implemented criterion
   3. It's enough to fine-tune for <1k steps

### Some notes on training and checking your model
here is the case when you apply mask after you calculate reward, B is a batch size, T is a sequence length

- Print the samples you generate. Does they look like plausible sentence? Print target to compare. Make sure your sentence is a string without bpe and tokenization. it should have shape BxT
- print reward. what is the range? do you so any unexpected numbers (e.g. a lot of 0-s). Make sure your reward has dimensionality BxT (same value along T, same value for each token in a sentence)
- when you gather from the log_prob, check your dimensions, you need to select along vocabulary dimension according to previously sampled sentences.
- check you loss sign, it should be positive and should go down during training
- generate and check model performance. if BLEU is 0, print the hypothesis
- you might want to remove learning rate warmup for fine-tuning
- learning rate and batch size are the most important params
- ou can change checkpoint saving interval by using checkpoint.save_interval=#epochs or checkpoint.save_interval_updates=# of steps
- you can reduce log_interval to see the results faster
- save the log file so you can access it after training in colab, you can use it for your report (plots, etc)
