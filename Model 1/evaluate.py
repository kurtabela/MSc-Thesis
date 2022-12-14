import sacrebleu
import argparse
from subprocess import call

def calculate_score_report(sys, ref, score_only, srcfile, reffile):

    command = "sacrebleu " + reffile + " -i " + srcfile + " -m bleu chrf "
    res = call(command, shell=True)

#    command = "sacrebleu " + reffile + " -i " + srcfile + " -m chrf --confidence"
#    res = call(command, shell=True)
    #chrf = sacrebleu.corpus_chrf(sys, ref)
    #bleu = sacrebleu.corpus_bleu(sys, ref)

    #prefix = 'BLEU = ' if score_only else ''

    #print('#### Score Report ####')
    #print(chrf)
    #print('{}{}'.format(prefix, bleu.format(score_only=score_only)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--system_output', '--sys', type=str, help='File with each line-by-line model outputs')
    parser.add_argument('--gold_reference', '--ref', type=str, help='File with corresponding line-by-line references')
    parser.add_argument('--detailed_output', action='store_const', const=True, default=False, help='(sacrebleu) Print additional BLEU information (default=False)')
    args = parser.parse_args()


    system_lines = []
    with open(args.system_output, 'r') as f:
        for line in f:
            system_lines.append(line.strip())

    gold_lines = []
    with open(args.gold_reference, 'r') as f:
        for line in f:
            gold_lines.append(line.strip())

    calculate_score_report(system_lines, [gold_lines], score_only=not args.detailed_output, srcfile=args.system_output, reffile=args.gold_reference)

