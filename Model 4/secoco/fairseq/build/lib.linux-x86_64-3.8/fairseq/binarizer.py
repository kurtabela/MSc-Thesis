# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import Counter

from fairseq.tokenizer import tokenize_line
import torch


def safe_readline(f):
    pos = f.tell()
    while True:
        try:
            return f.readline()
        except UnicodeDecodeError:
            pos -= 1
            f.seek(pos)  # search where this character begins


class Binarizer:
    @staticmethod
    def binarize(
        filename,
        dict,
        consumer,
        tokenize=tokenize_line,
        append_eos=True,
        reverse_order=False,
        offset=0,
        end=-1,
        already_numberized=False,
    ):
        nseq, ntok = 0, 0
        replaced = Counter()

        def replaced_consumer(word, idx):
            if idx == dict.unk_index and word != dict.unk_word:
                replaced.update([word])

        with open(filename, "r", encoding="utf-8") as f:
            f.seek(offset)
            # next(f) breaks f.tell(), hence readline() must be used
            line = safe_readline(f)
            while line:
                if end > 0 and f.tell() > end:
                    break
                if already_numberized:
                    id_strings = line.strip().split()
                    id_list = [int(id_string) for id_string in id_strings]
                    if reverse_order:
                        id_list.reverse()
                    if append_eos:
                        id_list.append(dict.eos())
                    ids = torch.IntTensor(id_list)
                else:
                    ids = dict.encode_line(
                        line=line,
                        line_tokenizer=tokenize,
                        add_if_not_exist=False,
                        consumer=replaced_consumer,
                        append_eos=append_eos,
                        reverse_order=reverse_order,
                    )
                nseq += 1
                ntok += len(ids)
                # noisy input repeat
                consumer(ids)
                consumer(ids)
                line = f.readline()
        return {
            "nseq": nseq,
            "nunk": sum(replaced.values()),
            "ntok": ntok,
            "replaced": replaced,
        }

    @staticmethod
    def binarize_alignments(filename, alignment_parser, consumer, offset=0, end=-1):
        nseq = 0

        with open(filename, "r") as f:
            f.seek(offset)
            line = safe_readline(f)
            while line:
                if end > 0 and f.tell() > end:
                    break
                ids = alignment_parser(line)
                nseq += 1
                consumer(ids)
                line = f.readline()
        return {"nseq": nseq}

    @staticmethod
    def binarize_noise(
        filename, filename_noise, filename_pos,
        dict,
        consumer_src1, consumer_ref1, consumer_src2, consumer_ref2, consumer_src3,
        tokenize=tokenize_line,
        append_eos=True,
        reverse_order=False,
        offset=0,
        end=-1,
    ):
        nseq, ntok = 0, 0
        replaced = Counter()

        def replaced_consumer(word, idx):
            if idx == dict.unk_index and word != dict.unk_word:
                replaced.update([word])

        with open(filename, "r", encoding="utf-8") as f:
            f.seek(offset)
            # next(f) breaks f.tell(), hence readline() must be used
            line = safe_readline(f)

            f_noise = open(filename_noise, 'r', encoding='utf-8')
            f_pos = open(filename_pos, 'r', encoding='utf-8')

            line_num = 1
            while line:
                print(line_num)
                line_num += 1
                if end > 0 and f.tell() > end:
                    break

                # noise 输入
                line_noise = f_noise.readline().strip()
                # 错误标示
                # <Wrong-[竞争对手]-[竞争@@|怼@@|手]-[1]>|30 <DropPunc-[。]>|32
                line_pos = f_pos.readline().strip()
                line = line.strip()

                if len(line_pos) == 0:
                    # no error
                    # assert line == line_noise
                    ids_src1 = dict.encode_line(
                        line=line,
                        line_tokenizer=tokenize,
                        add_if_not_exist=False,
                        consumer=replaced_consumer,
                        append_eos=append_eos,
                        reverse_order=reverse_order,
                    )
                    ids_ref1 = torch.zeros_like(ids_src1)
                    ids_src2 = ids_src1
                    ids_ref2 = ids_ref1
                    ids_src3 = ids_src1

                else:
                    # line 正确的输入
                    line_noise_tokens = line_noise.split()
                    line_noise_tokens.append('<eos>')
                    ids_src1 = dict.encode_line(
                        line=line_noise,
                        line_tokenizer=tokenize,
                        add_if_not_exist=False,
                        consumer=replaced_consumer,
                        append_eos=append_eos,
                        reverse_order=reverse_order,
                    )
                    ids_ref1 = torch.zeros_like(ids_src1)
                    pos_all = line_pos.split()
                    ids_ref1_list = torch.zeros_like(ids_src1)
                    for pos in pos_all:
                        pos, pos_idx = pos.rsplit('|', 1)
                        pos_idx = int(pos_idx)
                        ts = pos[1:-1].split('-')
                        if ts[0] == 'Wrong':
                            raw, error, idx = ts[1][1:-1], ts[2][1:-1], ts[3][1:-1]
                            idx = int(idx)
                            raw_id = dict.index(raw)
                            
                            error_tokens = error.split('|')
                            if len(error_tokens) == 1:
                                ids_ref1[pos_idx] = 1
                                ids_ref1_list[pos_idx] = 1
                                ids_ref1_list[pos_idx+1] = raw_id
                            else:
                                for i in range(pos_idx-idx, pos_idx+len(error_tokens)-idx):
                                    ids_ref1[i] = 1
                                    ids_ref1_list[i] = 1
                                ids_ref1_list[pos_idx+len(error_tokens)-idx] = raw_id
                        elif ts[0] == 'DropPunc':
                            raw = ts[1][1:-1]
                            raw_id = dict.index(raw)
                            ids_ref1_list[pos_idx] = raw_id
                        elif ts[0] == 'DropPro':
                            raw = ts[1][1:-1]
                            raw_id = dict.index(raw)
                            ids_ref1_list[pos_idx] = raw_id
                    indices = (ids_ref1_list!=1).nonzero().reshape(1, -1)[0]
                    ids_src2 = ids_src1.index_select(dim=-1, index=indices)
                    ids_ref2 = ids_ref1_list.index_select(dim=-1, index=indices)

                    
                    ids_src3 = dict.encode_line(
                        line=line,
                        line_tokenizer=tokenize,
                        add_if_not_exist=False,
                        consumer=replaced_consumer,
                        append_eos=append_eos,
                        reverse_order=reverse_order,
                    )
     
                nseq += 2
                ntok += len(ids_src1*2)

                consumer_src1(ids_src1)
                consumer_ref1(ids_ref1)
                consumer_src2(ids_src2)
                consumer_ref2(ids_ref2)
                consumer_src3(ids_src3)

                # exist errors, add noisy input -> clean output
                #if len(line_pos) != 0:
                consumer_src1(ids_src1)
                consumer_ref1(ids_ref1)
                consumer_src2(ids_src2)
                consumer_ref2(ids_ref2)
                consumer_src3(ids_src1)


                line = f.readline()
        return {
            "nseq": nseq,
            "nunk": sum(replaced.values()),
            "ntok": ntok,
            "replaced": replaced,
        }

    @staticmethod
    def find_offsets(filename, num_chunks):
        with open(filename, "r", encoding="utf-8") as f:
            size = os.fstat(f.fileno()).st_size
            chunk_size = size // num_chunks
            offsets = [0 for _ in range(num_chunks + 1)]
            for i in range(1, num_chunks):
                f.seek(chunk_size * i)
                safe_readline(f)
                offsets[i] = f.tell()
            return offsets
