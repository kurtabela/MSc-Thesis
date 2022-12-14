#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate raw text with a trained model. Batches data on-the-fly.
"""

from collections import namedtuple
import fileinput
import logging
import math
import sys
import os

import torch

from fairseq import checkpoint_utils, options, tasks, utils
from fairseq.data import encoders


logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger('fairseq_cli.interactive')


Batch = namedtuple('Batch', 'ids src1_tokens src_lengths')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')


def buffered_read(input, buffer_size):
    buffer = []
    with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h:
        for src_str in h:
            buffer.append(src_str.strip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, args, task, max_positions, encode_fn):
    tokens = [
        task.source_dictionary.encode_line(
            encode_fn(src_str), add_if_not_exist=False
        ).long()
        for src_str in lines
    ]
    lengths = [t.numel() for t in tokens]
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(tokens, lengths),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            ids=batch['id'],
            src1_tokens=batch['net_input']['src1_tokens'], src_lengths=batch['net_input']['src_lengths'],
        )

def remake_batches(tokens, src_tokens, args, task, max_positions):
    lengths = [t.numel() for t in tokens]
    res = src_tokens[-1].new(len(tokens), max(lengths)).fill_(1)
    for idx, token in enumerate(tokens):
        res[idx, max(lengths)-lengths[idx]:].copy_(token)
    res_lengths = torch.LongTensor(lengths)
    return res, res_lengths


def main(args):
    utils.import_user_module(args)

    if args.buffer_size < 1:
        args.buffer_size = 1
    if args.max_tokens is None and args.max_sentences is None:
        args.max_sentences = 1

    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
        '--max-sentences/--batch-size cannot be larger than --buffer-size'

    logger.info(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Setup task, e.g., translation
    task = tasks.setup_task(args)

    # Load ensemble
    logger.info('loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(os.pathsep),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Initialize generator
    generator = task.build_generator(models, args)

    # Handle tokenization and BPE
    tokenizer = encoders.build_tokenizer(args)
    bpe = encoders.build_bpe(args)

    def encode_fn(x):
        if tokenizer is not None:
            x = tokenizer.encode(x)
        if bpe is not None:
            x = bpe.encode(x)
        return x

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        *[model.max_positions() for model in models]
    )

    if args.buffer_size > 1:
        logger.info('Sentence buffer size: %s', args.buffer_size)
    logger.info('NOTE: hypothesis and token scores are output in base 2')
    logger.info('Type the input sentence and press return:')
    start_id = 0
    for inputs in buffered_read(args.input, args.buffer_size):
        results = []
        for batch in make_batches(inputs, args, task, max_positions, encode_fn):
            src_tokens = batch.src1_tokens
            src_lengths = batch.src_lengths
            #logger.info(task.source_dictionary.string(src_tokens))
            #logger.info(src_tokens)
            if use_cuda:
                src_tokens = src_tokens.cuda()
                src_lengths = src_lengths.cuda()
            
            for _ in range(3):
                # predict delete
                delete, delete_mask = models[0].forward_delete(src_tokens)
                delete = torch.argmax(delete, dim=-1)
                delete = delete*(~delete_mask)
                #print(delete)
                # edit src_tokens
                src_tokens_list = src_tokens.tolist()
                new_src_tokens_list = []
                delete_list = delete.tolist()
                for idx, line_delete in enumerate(delete_list):
                    temp = []
                    for jdx, token_delete in enumerate(line_delete):
                        if token_delete == 1 or src_tokens_list[idx][jdx] == 1:
                            continue
                        else:
                            temp.append(src_tokens_list[idx][jdx])
                    new_src_tokens_list.append(torch.IntTensor(temp))
                src_tokens, src_lengths = remake_batches(new_src_tokens_list, src_tokens, args, task, max_positions)
                #print(src_tokens)
                #print(task.source_dictionary.string(src_tokens))
                #print('finish delete\n\n')

                # predict insert
                insert, insert_mask = models[0].forward_insert(src_tokens)
                insert= torch.argmax(insert, dim=-1)
                insert = (insert*(~insert_mask))
                #print(insert)
                # edit src_tokens
                insert_list = insert.tolist()
                src_tokens_list = src_tokens.tolist()
                new_src_tokens_list = []
                for idx, line_insert in enumerate(insert_list):
                    temp = []
                    for jdx, token_insert in enumerate(line_insert):
                        if token_insert != 0:
                            temp.append(token_insert)
                        temp.append(src_tokens_list[idx][jdx])
                    new_src_tokens_list.append(torch.IntTensor(temp))
                src_tokens, src_lengths = remake_batches(new_src_tokens_list, src_tokens, args, task, max_positions)
            #logger.info(src_tokens)
            #logger.info(task.source_dictionary.string(src_tokens))
            #logger.info('finish insert\n\n')
            
            sample = {
                'net_input': {
                    'src3_tokens': src_tokens,
                    'src_lengths': src_lengths,
                },
            }
            
            #logger.info(sample)
            translations = task.inference_step(generator, models, sample)
            for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
                src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
                results.append((start_id + id, src_tokens_i, hypos))

        # sort output to match input order
        for id, src_tokens, hypos in sorted(results, key=lambda x: x[0]):
            if src_dict is not None:
                src_str = src_dict.string(src_tokens, args.remove_bpe)
                print('S-{}\t{}'.format(id, src_str))

            # Process top predictions
            for hypo in hypos[:min(len(hypos), args.nbest)]:
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'],
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=args.remove_bpe,
                )
                detok_hypo_str = decode_fn(hypo_str)
                score = hypo['score'] / math.log(2)  # convert to base 2
                # original hypothesis (after tokenization and BPE)
                print('H-{}\t{}\t{}'.format(id, score, hypo_str))
                # detokenized hypothesis
                print('D-{}\t{}\t{}'.format(id, score, detok_hypo_str))
                print('P-{}\t{}'.format(
                    id,
                    ' '.join(map(
                        lambda x: '{:.4f}'.format(x),
                        # convert from base e to base 2
                        hypo['positional_scores'].div_(math.log(2)).tolist(),
                    ))
                ))
                if args.print_alignment:
                    alignment_str = " ".join(["{}-{}".format(src, tgt) for src, tgt in alignment])
                    print('A-{}\t{}'.format(
                        id,
                        alignment_str
                    ))

        # update running id counter
        start_id += len(inputs)


def cli_main():
    parser = options.get_generation_parser(interactive=True)
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
