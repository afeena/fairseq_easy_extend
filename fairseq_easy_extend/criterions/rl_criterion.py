import math
from argparse import Namespace
import torch
import torch.nn.functional as F
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.data import encoders
from fairseq.dataclass import FairseqDataclass
from dataclasses import dataclass, field
from fairseq.logging import metrics
from sacrebleu.metrics import BLEU
from evaluate import load
import jiwer
from comet import download_model, load_from_checkpoint
import wandb
from typing import Optional

@dataclass
class RLCriterionConfig(FairseqDataclass):
    sentence_level_metric: str = field(default="bleu", metadata={"help": "sentence level metric"}) 
    sampling: str = field(default="multinomial", metadata={"help": "sampling method"})
    detokenization: bool = field(default=True,  metadata={"help": "whether to detokenize the output"})
    wandb_project: str = field(default="nlp2-nanmt", metadata={"help": "wandb project"})
    wandb_run: Optional[str] = field(default=None, metadata={"help": "wandb run name"})
    
@register_criterion("rl_loss", dataclass=RLCriterionConfig)
class RLCriterion(FairseqCriterion):
    def __init__(self, task, sentence_level_metric, sampling="multinomial", wandb_project='nlp2-nanmt', wandb_run=None):
        super().__init__(task)
        self.metric = sentence_level_metric
        self.tokenizer = encoders.build_tokenizer(Namespace(tokenizer='moses'))
        self.tgt_dict = task.target_dictionary
        self.src_dict = task.source_dictionary
        self.tgt_lang = "en"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.sampling = sampling
        self.bleu = BLEU(effective_order="True")
        self.meteor = load('meteor')
        self.rouge = load('rouge')
        # self.ter = load('ter')
        self.bert = load('bertscore')
        self.bleurt = load('bleurt', module_type='metric', checkpoint='bleurt-large-128')
        if self.metric == "comet":
          model_path = download_model("Unbabel/wmt22-comet-da")
          self.comet = load_from_checkpoint(model_path)
        # init wandb project
        wandb.init(project=wandb_project)
        if wandb_run:
            wandb.run.name = wandb_run
        
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        nsentences, ntokens = sample["nsentences"], sample["ntokens"]

        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        tgt_tokens, prev_output_tokens = sample["target"], sample["prev_target"]

        outputs = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens)
        
        #get loss only on tokens, not on lengths
        outs = outputs["word_ins"].get("out", None)
        masks = outputs["word_ins"].get("mask", None)

        loss, reward = self._compute_loss(outs, tgt_tokens, src_tokens, masks)

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1
        logging_output = {
            "loss": loss.detach(),
            "nll_loss": loss.detach(),
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
            "reward": reward.detach()
        }
        
        # if reduce:
            # self.reduce_metrics(logging_output)

        return loss, sample_size, logging_output

    def decode(self, toks, ref="tgt", escape_unk=False):
        """
        Decode a tensor of token ids into a string.
        """
        decode_dict = self.tgt_dict if ref == "tgt" else self.src_dict
        with torch.no_grad():
            s = decode_dict.string(
                toks.int().cpu(),
                "@@ ",
                # The default unknown string in fairseq is `<unk>`, but this is tokenized by sacrebleu as `< unk >`, inflating BLEU scores. Instead, we use a somewhat more verbose alternative that is unlikely to appear in the real reference, but doesn't get split into multiple tokens.
                unk_string=(
                    "UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"
                ),
            )
            s = self.tokenizer.decode(s)
        return s

    def compute_reward(self, preds, targets, sources=None):
        """
        Compute reward metric for a batch of prediction and target sentences
        """
        # detokenize (convert to str) preds & targets
        preds_str = [self.decode(pred) for pred in preds]
        targets_str = [self.decode(target) for target in targets]
        sources_str = [self.decode(source, ref="src") for source in sources] if sources is not None else None
        print(f'1st target sent: {targets_str[0]}')
        print(f'1st pred sent: {preds_str[0]}')

        # compute reward metric
        seq_len = preds.shape[1]
        if self.metric == "bleu":
            reward = [[self.bleu.sentence_score(pred, [target]).score] * seq_len for pred, target in zip(preds_str, targets_str)]
        
        elif self.metric == "meteor":
            meteor_scores = [self.meteor.compute(predictions=[preds], references=[targets])['meteor'] for preds, targets in zip(preds_str, targets_str)]
            reward = [[score] * seq_len for score in meteor_scores]
            
        elif self.metric == "rouge":
            rouge_scores = self.rouge.compute(predictions=preds_str, references=targets_str, use_aggregator=False)['rougeL']
            reward = [[score] * seq_len for score in rouge_scores]
            
        elif self.metric == "wer":
            wer_scores = [jiwer.wer(target, pred) for pred, target in zip(targets_str, preds_str)]
            reward = [[score] * seq_len for score in wer_scores]
            
        elif self.metric == "bert":
            bert_scores = self.bert.compute(predictions=preds_str, references=targets_str, lang='en')['f1']
            reward = [[score] * seq_len for score in bert_scores]

        elif self.metric == "bleurt":
            bleurt_scores = self.bleurt.compute(predictions=preds_str, references=targets_str)['scores']
            reward = [[score] * seq_len for score in bleurt_scores]
            
        elif self.metric == "comet":
            data = [{"src": source, "mt": pred, "ref": target} for source, pred, target in zip(sources_str, preds_str, targets_str)]
            reward = self.comet.predict(data, batch_size=8, gpus=1)['scores']
            reward = [[score] * seq_len for score in reward]
        else:
            raise ValueError(f"metric {self.metric} not supported")
        reward = torch.tensor(reward).to(self.device)
        return reward

    def _compute_loss(self, outputs, targets, sources, masks=None):
        """
        outputs: batch x len x d_model
        targets: batch x len
        sources: batch x len
        masks:   batch x len
        """
        # input to device
        outputs = outputs.to(self.device)
        targets = targets.to(self.device)
        
        bsz, seq_len, vocab_size = outputs.size()

        # sample predictions and convert predictions and targets to strings without tokenization and bpe
        probs = F.softmax(outputs, dim=-1)
        with torch.no_grad():
            # multinomial or argmax sampling
            if self.sampling == "multinomial":
                preds = torch.multinomial(probs.view(-1, vocab_size), 1, replacement=True).view(bsz, seq_len)
            elif self.sampling == "argmax":
                preds = torch.argmax(probs.view(-1, vocab_size), dim=-1).view(bsz, seq_len)
            
            # compute reward metric
            reward = self.compute_reward(preds, targets, sources)
            # print(f'reward: {reward}')
        # print(f'shape of probs: {probs.shape}, shape of targets: {targets.shape}, shape of masks: {masks.shape}, shape of preds: {preds.shape}, shape of rewards: {reward.shape}')
        # apply mask
        if masks is not None:
            masks = masks.to(self.device)
            probs, targets = probs[masks], targets[masks]
            # outputs, targets = outputs[masks], targets[masks]
            reward, preds = reward[masks], preds[masks]

        # get the log probs of the samples
        # log_probs = F.log_softmax(probs, dim=-1)
        log_probs = torch.log(probs)
        sample_log_probs = torch.gather(log_probs, -1, preds.unsqueeze(1)).squeeze()
        
        # calculate loss of all samples and average for batch loss
        loss = -sample_log_probs * reward
        loss, reward = loss.mean(), reward.mean()
        print(f'loss: {loss.item():.3f} | reward: {reward:.3f}')    
        
        return loss, reward

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        reward_sum= sum(log.get("reward", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar("reward", reward_sum / sample_size, sample_size, round=3)
        
        # log to wandb
        wandb.log({
            "loss": loss_sum / sample_size / math.log(2),
            "reward": reward_sum / sample_size,
        })