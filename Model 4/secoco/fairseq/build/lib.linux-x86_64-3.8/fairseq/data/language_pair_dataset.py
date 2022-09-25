# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch

from . import data_utils, FairseqDataset


logger = logging.getLogger(__name__)


def collate(
    samples, pad_idx, eos_idx, left_pad_source=True, left_pad_target=False,
    input_feeding=True,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src1_tokens = merge('src1', left_pad=left_pad_source)
    if 'src2' in samples[0]:
        src2_tokens = merge('src2', left_pad=left_pad_source)
        src3_tokens = merge('src3', left_pad=left_pad_source)
    # sort by descending source length
    if 'src2' in samples[0]:
        src_lengths = torch.LongTensor([s['src3'].numel() for s in samples])
    else:
        src_lengths = torch.LongTensor([s['src1'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src1_tokens = src1_tokens.index_select(0, sort_order)
    if 'src2' in samples[0]:
        src2_tokens = src2_tokens.index_select(0, sort_order)
        src3_tokens = src3_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None

    if 'src2' in samples[0]:
        target = merge('ref3', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor([s['ref3'].numel() for s in samples]).index_select(0, sort_order)
        ntokens = sum(len(s['ref3']) for s in samples)

        ref1_tokens = merge('ref1', left_pad=left_pad_source)
        ref2_tokens = merge('ref2', left_pad=left_pad_source)
        ref1_tokens = ref1_tokens.index_select(0, sort_order)
        ref2_tokens = ref2_tokens.index_select(0, sort_order)

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'ref3',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)

        batch = {
            'id': id,
            'nsentences': len(samples),
            'ntokens': ntokens,
            'net_input': {
                'src1_tokens': src1_tokens,
                'src2_tokens': src2_tokens,
                'src3_tokens': src3_tokens,
                'src_lengths': src_lengths,
            },
            'ref1': ref1_tokens,
            'ref2': ref2_tokens,
            'target': target,
        }
        if prev_output_tokens is not None:
            batch['net_input']['prev_output_tokens'] = prev_output_tokens

    else:
        ntokens = sum(len(s['src1']) for s in samples)
        batch = {
            'id': id,
            'nsentences': len(samples),
            'ntokens': ntokens,
            'net_input': {
                'src1_tokens': src1_tokens,
                'src_lengths': src_lengths,
            },
            'target': target,
        }
    return batch


class LanguagePairDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        tgt_dict (~fairseq.data.Dictionary, optional): target vocabulary
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        max_source_positions (int, optional): max number of tokens in the
            source sentence (default: 1024).
        max_target_positions (int, optional): max number of tokens in the
            target sentence (default: 1024).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
        align_dataset (torch.utils.data.Dataset, optional): dataset
            containing alignments.
        append_bos (bool, optional): if set, appends bos to the beginning of
            source/target sentence.
    """

    def __init__(
        self, src1, src1_sizes, src_dict,
        ref1=None, ref1_sizes=None, tgt_dict=None,
        src2=None, src3=None, ref2=None, ref3=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, input_feeding=True,
        remove_eos_from_source=False, append_eos_to_target=False,
        align_dataset=None,
        append_bos=False, eos=None
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        self.src1 = src1
        self.ref1 = ref1
        self.src2 = src2
        self.ref2 = ref2
        self.src3 = src3
        self.ref3 = ref3
        self.src_sizes = np.array(src1_sizes)
        self.tgt_sizes = np.array(ref1_sizes) if ref1_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.align_dataset = align_dataset
        if self.align_dataset is not None:
            assert self.tgt_sizes is not None, "Both source and target needed when alignments are provided"
        self.append_bos = append_bos
        self.eos = (eos if eos is not None else src_dict.eos())

    def __getitem__(self, index):
        if self.ref1 is not None:
            ref1_item, ref2_item, ref3_item = self.ref1[index], self.ref2[index], self.ref3[index]
            src2_item, src3_item = self.src2[index], self.src3[index]
        src1_item = self.src1[index]
        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.ref1 is not None:
            example = {
                'id': index,
                'src1': src1_item,
                'ref1': ref1_item,
                'src2': src2_item,
                'ref2': ref2_item,
                'src3': src3_item,
                'ref3': ref3_item,
            }
        else:
            example = {
                'id': index,
                'src1': src1_item,
                'ref1': None
            }
        return example

    def __len__(self):
        return len(self.src1)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
        """
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.eos,
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src1, 'supports_prefetch', False)
            and (getattr(self.ref1, 'supports_prefetch', False) or self.ref1 is None)
        )

    def prefetch(self, indices):
        self.src1.prefetch(indices)
        self.src2.prefetch(indices)
        self.src3.prefetch(indices)
        self.ref1.prefetch(indices)
        self.ref2.prefetch(indices)
        self.ref3.prefetch(indices)
