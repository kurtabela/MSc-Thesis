import sacrebleu
import argparse
from subprocess import check_call

def calculate_score_report(score_only, srcfile, reffile):
    print("in calculate score report")
    
    check_call("split " + reffile + " ref_ --lines 500000 ", shell=True)
    check_call("split " + srcfile + " src_ --lines 500000", shell=True)
    check_call("sacrebleu ref_aa -i src_aa -m chrf", shell=True)
    check_call("sacrebleu ref_ab -i src_ab -m chrf", shell=True)
    check_call("sacrebleu ref_aa -i src_aa -m bleu", shell=True)
    check_call("sacrebleu ref_ab -i src_ab -m bleu", shell=True)
#    print("doing final command")
#    check_call("sacrebleu " + reffile + " -i " + srcfile + " -m bleu", shell=True)
    #chrf = sacrebleu.corpus_chrf(sys, ref)
    #bleu = sacrebleu.corpus_bleu(sys, ref)

    #prefix = 'BLEU = ' if score_only else ''

    #print('#### Score Report ####')
    #print(chrf)
    #print('{}{}'.format(prefix, bleu.format(score_only=score_only)))


if __name__ == '__main__':
    print("IN MAIN")
    parser = argparse.ArgumentParser()

    parser.add_argument('--system_output', '--sys', type=str, help='File with each line-by-line model outputs')
    parser.add_argument('--gold_reference', '--ref', type=str, help='File with corresponding line-by-line references')
    parser.add_argument('--detailed_output', action='store_const', const=True, default=False, help='(sacrebleu) Print additional BLEU information (default=False)')
    args = parser.parse_args()
    calculate_score_report(score_only=not args.detailed_output, srcfile=args.system_output, reffile=args.gold_reference)
