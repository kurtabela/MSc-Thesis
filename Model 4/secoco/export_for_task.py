import re

import argparse

'''
<InsertRandom-[with]>
<Delete-[&apos;]>
<InsertRepeat-[a|a|a]>
'''


def export_for_robust(f_zh, fw_zh, fw_pos):
    print("READING FROM: " + str(f_zh) + " AND WRITING TO " + str(fw_zh) + " AND " + str(fw_pos))
    with open(f_zh, "r", encoding="utf-8") as f:
        with open(fw_zh, "w+", encoding="utf-8") as w:
            with open(fw_pos, "w+", encoding="utf-8") as w_pos:
                data = f.read()

                data_en = data.split('\n')
                for line in data_en:
                    line = line.strip()
                    tokens = line.split()

                    sps = re.findall(r'<[Delete|InsertRepeat|InsertRandom].*?>', line)
                    idx = 0
                    i = 0
                    temp = []
                    noise = []
                    for token in tokens:
                        if idx >= len(sps):
                            temp.append(token)
                            i += 1
                        elif token == sps[idx]:
                            idx += 1
                            ts = token[1:-1].split('-')
                            if ts[0] == 'Delete':
                                raw = ts[1][1:-1]
                                noise.append(f'{token}|{i}')
                            elif ts[0] == 'InsertRepeat':
                                raw = ts[1][1:-1]
                                raw_tokens = raw.split('|')
                                temp.extend(raw_tokens)
                                noise.append(f'{token}|{i}to{i + len(raw_tokens)}')
                                i += len(raw_tokens)
                            elif ts[0] == 'InsertRandom':
                                raw = ts[1][1:-1]
                                temp.append(raw)
                                noise.append(f'{token}|{i}')
                                i += 1
                            else:
                                print('error!!!!!')
                                print(ts[0])
                                temp.append(token)
                                i += 1

                        else:
                            temp.append(token)
                            i += 1

                    w.write(' '.join(temp) + '\n')
                    w_pos.write(' '.join(noise) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--data_out', type=str)
    args = parser.parse_args()
    for split in ['train', 'dev', 'test']:
        export_for_robust(args.data_out + split + '.bpe.tag.' + args.src, args.data_out + split + '.bpe.tag.noise.' + args.src, args.data_out + split + '.bpe.tag.noise.' + args.src+ '.pos')
