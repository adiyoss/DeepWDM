import argparse
import os

from lib.textgrid import TextGrid
from predict import predict


def run_dir(in_path, out_path):
    for item in os.listdir(in_path):
        if item.endswith('.wav'):
            out_file_path = out_path + item.replace('.wav', '.TextGrid')
            predict(in_path + item, out_file_path, 'rnn')
            out_txt = out_file_path.replace('.TextGrid', '.txt')
            t = TextGrid()
            t.read(out_file_path)
            onset = int(t._TextGrid__tiers[0]._IntervalTier__intervals[1]._Interval__xmin*100)
            offset = int(t._TextGrid__tiers[0]._IntervalTier__intervals[1]._Interval__xmax*100)
            fid = open(out_txt, 'w')
            fid.write(str(onset)+'-'+str(offset))
            fid.close()

if __name__ == "__main__":
    # the first argument is the wav file path
    # the second argument is the TextGrid path
    # -------------MENU-------------- #
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir", help="The input directory")
    parser.add_argument("out_dir", help="The output directory")
    args = parser.parse_args()

    # main function
    run_dir(args.in_dir, args.out_dir)
