import torch
from fairseq import search
from fairseq.dataclass import FairseqDataclass


def build_model(self, cfg: FairseqDataclass, from_checkpoint=False):
    """
    Build the :class:`~fairseq.models.BaseFairseqModel` instance for this
    task.

    Args:
        cfg (FairseqDataclass): configuration object

    Returns:
        a :class:`~fairseq.models.BaseFairseqModel` instance
    """
    from fairseq_easy_extend import models
    from fairseq import quantization_utils

    model = models.build_model(cfg, self, from_checkpoint)
    model = quantization_utils.quantize_model_scalar(model, cfg)
    return model


def build_model_levenshtein(self, cfg, from_checkpoint=False):
    model = super().build_model(cfg, from_checkpoint=from_checkpoint)
    model.decoder.setup_search_matrix()
    return model


def valid_step(self, sample, model, criterion):
    EVAL_BLEU_ORDER = 4
    model.eval()
    with torch.no_grad():
        sample["prev_target"] = self.inject_noise(sample["target"])
        loss, sample_size, logging_output = criterion(model, sample)
    if self.cfg.eval_bleu:
        bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
        logging_output["_bleu_sys_len"] = bleu.sys_len
        logging_output["_bleu_ref_len"] = bleu.ref_len
        # we split counts into separate entries so that they can be
        # summed efficiently across workers using fast-stat-sync
        assert len(bleu.counts) == EVAL_BLEU_ORDER
        for i in range(EVAL_BLEU_ORDER):
            logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
            logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]

    return loss, sample_size, logging_output


def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
        prefix_allowed_tokens_fn=None,
):
    """
    Build a :class:`~fairseq.SequenceGenerator` instance for this
    task.

    Args:
        models (List[~fairseq.models.FairseqModel]): ensemble of models
        args (fairseq.dataclass.configs.GenerationConfig):
            configuration object (dataclass) for generation
        extra_gen_cls_kwargs (Dict[str, Any]): extra options to pass
            through to SequenceGenerator
        prefix_allowed_tokens_fn (Callable[[int, torch.Tensor], List[int]]):
            If provided, this function constrains the beam search to
            allowed tokens only at each step. The provided function
            should take 2 arguments: the batch ID (`batch_id: int`)
            and a unidimensional tensor of token ids (`inputs_ids:
            torch.Tensor`). It has to return a `List[int]` with the
            allowed tokens for the next generation step conditioned
            on the previously generated tokens (`inputs_ids`) and
            the batch ID (`batch_id`). This argument is useful for
            constrained generation conditioned on the prefix, as
            described in "Autoregressive Entity Retrieval"
            (https://arxiv.org/abs/2010.00904) and
            https://github.com/facebookresearch/GENRE.
    """
    if getattr(args, "score_reference", False):
        from fairseq.sequence_scorer import SequenceScorer

        return SequenceScorer(
            self.target_dictionary,
            compute_alignment=getattr(args, "print_alignment", False),
        )

    from fairseq_easy_extend.sequence_generator import SequenceGenerator

    # Choose search strategy. Defaults to Beam Search.
    sampling = getattr(args, "sampling", False)
    sampling_topk = getattr(args, "sampling_topk", -1)
    sampling_topp = getattr(args, "sampling_topp", -1.0)
    diverse_beam_groups = getattr(args, "diverse_beam_groups", -1)
    diverse_beam_strength = getattr(args, "diverse_beam_strength", 0.5)
    match_source_len = getattr(args, "match_source_len", False)
    diversity_rate = getattr(args, "diversity_rate", -1)
    constrained = getattr(args, "constraints", False)
    if prefix_allowed_tokens_fn is None:
        prefix_allowed_tokens_fn = getattr(args, "prefix_allowed_tokens_fn", None)
    if (
            sum(
                int(cond)
                for cond in [
                    sampling,
                    diverse_beam_groups > 0,
                    match_source_len,
                    diversity_rate > 0,
                ]
            )
            > 1
    ):
        raise ValueError("Provided Search parameters are mutually exclusive.")
    assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
    assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

    if sampling:
        search_strategy = search.Sampling(
            self.target_dictionary, sampling_topk, sampling_topp
        )
    elif diverse_beam_groups > 0:
        search_strategy = search.DiverseBeamSearch(
            self.target_dictionary, diverse_beam_groups, diverse_beam_strength
        )
    elif match_source_len:
        # this is useful for tagging applications where the output
        # length should match the input length, so we hardcode the
        # length constraints for simplicity
        search_strategy = search.LengthConstrainedBeamSearch(
            self.target_dictionary,
            min_len_a=1,
            min_len_b=0,
            max_len_a=1,
            max_len_b=0,
        )
    elif diversity_rate > -1:
        search_strategy = search.DiverseSiblingsSearch(
            self.target_dictionary, diversity_rate
        )
    elif constrained:
        search_strategy = search.LexicallyConstrainedBeamSearch(
            self.target_dictionary, args.constraints
        )
    elif prefix_allowed_tokens_fn:
        search_strategy = search.PrefixConstrainedBeamSearch(
            self.target_dictionary, prefix_allowed_tokens_fn
        )
    else:
        search_strategy = search.BeamSearch(self.target_dictionary)

    extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}

    return SequenceGenerator(
        models,
        self.target_dictionary,
        beam_size=getattr(args, "beam", 5),
        max_len_a=getattr(args, "max_len_a", 0),
        max_len_b=getattr(args, "max_len_b", 200),
        min_len=getattr(args, "min_len", 1),
        normalize_scores=(not getattr(args, "unnormalized", False)),
        len_penalty=getattr(args, "lenpen", 1),
        unk_penalty=getattr(args, "unkpen", 0),
        temperature=getattr(args, "temperature", 1.0),
        match_source_len=getattr(args, "match_source_len", False),
        no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
        search_strategy=search_strategy,
        metric=getattr(args, "decoding_measure", "cosine"),
        **extra_gen_cls_kwargs,
    )
