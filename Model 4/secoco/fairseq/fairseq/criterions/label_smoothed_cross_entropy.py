# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

def nll_loss(lprobs, target, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    nll_loss = nll_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
    return nll_loss


@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg, label_smoothing):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, loss_translation_data, nll_loss_translation_data, \
            nll_loss_delete_data, nll_loss_insert_data = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': loss.data,
            'loss_translation': loss_translation_data,
            'nll_loss_translation': nll_loss_translation_data,
            'nll_loss_delete': nll_loss_delete_data,
            'nll_loss_insert': nll_loss_insert_data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        losses, nll_losses = [], []
        lprobs_translation = model.get_normalized_probs(net_output['translation']['out'], log_probs=True)
        lprobs_translation = lprobs_translation.view(-1, lprobs_translation.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss_translation, nll_loss_translation = label_smoothed_nll_loss(
            lprobs_translation, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )

        # delete
        outputs = net_output['delete']['out'].view(-1, net_output['delete']['out'].size(-1))
        masks = ~(net_output['delete']['mask'].view(-1))
        targets = sample['ref1'].view(-1, 1)
        outputs,targets = outputs[masks], targets[masks]
        nll_loss_delete = nll_loss(
            outputs, targets, reduce=reduce
        )

        # insert
        outputs = net_output['insert']['out'].view(-1, net_output['insert']['out'].size(-1))
        masks = ~(net_output['insert']['mask'].view(-1))
        targets = sample['ref2'].view(-1, 1)
        outputs, targets = outputs[masks], targets[masks]
        nll_loss_insert = nll_loss(
            outputs, targets, reduce=reduce
        )

        return loss_translation+nll_loss_delete+nll_loss_insert, \
                    loss_translation.data, nll_loss_translation.data, \
                        nll_loss_delete.data, nll_loss_insert.data

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        loss_translation_sum = sum(log.get('loss_translation', 0) for log in logging_outputs)
        nll_loss_translation_sum = sum(log.get('nll_loss_translation', 0) for log in logging_outputs)
        nll_loss_delete_sum = sum(log.get('nll_loss_delete', 0) for log in logging_outputs)
        nll_loss_insert_sum = sum(log.get('nll_loss_insert', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('loss_translation', loss_translation_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_scalar('nll_loss_translation', nll_loss_translation_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_scalar('nll_loss_delete', nll_loss_delete_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_scalar('nll_loss_insert', nll_loss_insert_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss_translation'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
