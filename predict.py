import argparse
import os
import shutil
import sys

from label2textgrid import create_text_grid
from lib import utils
from post_process import post_process


def predict(input_path, output_path, model):
    tmp_dir = 'tmp/'
    tmp_features = 'tmp.features'
    tmp_prob = 'tmp.prob'
    tmp_prediction = 'tmp.prediction'
    tmp_duration = 'tmp.dur'

    if not os.path.exists(input_path):
        print >> sys.stderr, "wav file does not exits"
        return

    t_model = model.upper()    
    if t_model == 'RNN':
        model_path = 'results/1_layer_model.net'
        print '==> using single layer RNN'
    elif t_model == '2RNN':
        model_path = 'results/2_layer_model.net'
        print '==> using 2 stacked layers RNN'
    elif t_model == 'BIRNN':
        model_path = 'results/bi_model.net'
        print '==> using bi-directional RNN'
    else:
        model_path = 'results/1_layer_model.net'
        print '==> unknown model, using default model: single RNN'

    length = utils.get_wav_file_length(input_path)
    prob_file = tmp_dir + tmp_prob
    predict_file = tmp_dir + tmp_prediction
    dur_file = tmp_dir+tmp_duration

    # remove tmo dir if exists
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.mkdir(tmp_dir)

    print '\n1) Extracting features and classifying ...'
    cmd = 'python predict_single_file.py %s %s ' % (
    os.path.abspath(os.path.abspath(input_path)), os.path.abspath(tmp_dir) + '/' + tmp_features)
    os.chdir("front_end/")
    utils.easy_call(cmd)
    os.chdir("..")

    print '\n2) Model predictions ...'
    cmd = 'th classify.lua -folder_path %s -x_filename %s -class_path %s -prob_path %s -model_path %s' % (
    os.path.abspath(tmp_dir), tmp_features, os.path.abspath(predict_file), os.path.abspath(prob_file), model_path)
    os.chdir("back_end/")
    utils.easy_call(cmd)
    os.chdir("..")

    print '\n3) Extracting duration'
    post_process(os.path.abspath(predict_file), dur_file)

    print '\n4) Writing TextGrid file to %s ...' % output_path
    create_text_grid(dur_file, output_path, length, float(0.0))

    # remove leftovers
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    # -------------MENU-------------- #
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="The path to the wav file")    
    parser.add_argument("output_path", help="The path to save new text-grid file")
    parser.add_argument("model", help="The type pf model: rnn | 2rnn | birnn")
    args = parser.parse_args()

    # main function
    predict(args.input_path, args.output_path, args.model)
