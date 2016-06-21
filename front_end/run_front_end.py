import argparse
import shutil
import sys
import os
import numpy as np

from lib import utility as utils
from lib import textgrid as tg
from lib.htkmfc import HTKFeat_read

__author__ = 'yossiadi'

# globals
features_dir = "mfcc/"
labels_dir = "labels/"
tmp_dir = "tmp/"
merge_file_x = "tmp.features"
merge_file_y = "tmp.label"
NO_STRING = '$$'


def extract_single_mfcc(in_path, out_path):
    """
    Extract mfcc features from one audio file
    :param in_path: the path to the audio file
    :param out_path: the path to save the mfcc features
    """
    import platform
    plat = platform.system().lower()
    if plat is 'darwin':
        sox_path = 'sbin/osx/sox'
        htk_path = 'sbin/osx'
    elif 'linux' in plat:
        sox_path = 'sox'
        htk_path = 'sbin/linux'
    else:
        sox_path = 'sbin/osx/sox'
        htk_path = 'sbin/osx'

    tmp_file = "tmp.wav"
    cmd = "%s %s -r 16000 -b 16 %s" % (sox_path, in_path, tmp_file)
    utils.easy_call(cmd)
    cmd = "%s/HCopy -C config/htk.config %s %s" % (htk_path, tmp_file, out_path)
    utils.easy_call(cmd)
    os.remove(tmp_file)


def extract_single_acoustic(in_path, out_path):
    """
    Extract mfcc features from one audio file
    :param in_path: the path to the audio file
    :param out_path: the path to save the mfcc features
    """
    tmp_input = "tmp.input"
    tmp_features = "tmp.features"
    tmp_label = "tmp.labels"
    zero = 0.01

    input_file = open(tmp_dir + tmp_input, 'wb')  # open the input file for the feature extraction
    features_file = open(tmp_dir + tmp_features, 'wb')  # open file for the feature list path
    labels_file = open(tmp_dir + tmp_label, 'wb')  # open file for the labels
    length = utils.get_wav_file_length(in_path)

    # write the data
    input_file.write(
            '"' + in_path + '" ' + str('%.8f' % 0) + ' ' + str(float(length) - zero) + ' ' + str(
                '%.8f' % 0) + ' ' + str(
                    '%.8f' % 0))
    features_file.write(out_path)

    input_file.close()
    features_file.close()
    labels_file.close()

    command = "sbin/fea_extract %s %s %s" % (tmp_dir + tmp_input, tmp_dir + tmp_features, tmp_dir + tmp_label)
    utils.easy_call(command)

    # remove leftovers
    os.remove(tmp_dir + tmp_input)
    os.remove(tmp_dir + tmp_features)
    os.remove(tmp_dir + tmp_label)


def extract_mfcc_dir(in_path, out_path):
    """
    Extract mfcc features from directory
    :param in_path: the path to the audio files directory
    :param out_path: the path to save the mfcc features (should be directory)
    """
    if not os.path.exists(in_path):
        print >> sys.stderr, "Directory does not exists"

    if not os.path.exists(out_path):
        print "Output directory does not exists"
        print "Creating output directory"
        os.mkdir(out_path)

    for item in os.listdir(in_path):
        if item.endswith(".wav"):
            abs_path = os.path.abspath(in_path + item)
            extract_single_mfcc(abs_path, out_path + item.replace(".wav", ".htk"))


def extract_acoustic_dir(in_path, out_path):
    """
    Extract acoustic features from directory
    :param in_path: the path to the audio files directory
    :param out_path: the path to save the features (should be directory)
    """
    if not os.path.exists(in_path):
        print >> sys.stderr, "Directory does not exists"

    if not os.path.exists(out_path):
        print "Output directory does not exists"
        print "Creating output directory"
        os.mkdir(out_path)

    for item in os.listdir(in_path):
        if item.endswith(".wav"):
            abs_path = os.path.abspath(in_path + item)
            extract_single_acoustic(abs_path, out_path + item.replace(".wav", ".data"))


def create_labels(in_path, out_path):
    """
    Extract the labels from the text grid files
    :param in_path: the path to the text grid files directory
    :param out_path: the path to save the labels
    """
    if not os.path.exists(in_path):
        print >> sys.stderr, "Directory does not exists"

    if not os.path.exists(out_path):
        print "Output directory does not exists"
        print "Creating output directory"
        os.mkdir(out_path)

    for item in os.listdir(in_path):
        if item.endswith(".TextGrid"):
            abs_path = os.path.abspath(in_path + item)
            textgrid = tg.TextGrid()
            textgrid.read(abs_path)

            max_time = utils.get_wav_file_length(abs_path.replace(".TextGrid", ".wav")) * 100 - 1
            onset = np.ceil(textgrid.tiers[1].intervals[1].minTime * 100)
            offset = np.ceil(textgrid.tiers[1].intervals[1].maxTime * 100)

            labels = np.zeros([int(max_time), 2])
            labels[:, 1] = 1

            labels[:, 0][int(onset): int(offset)] = 1
            labels[:, 1][int(onset): int(offset)] = 0

            np.savetxt(out_path + item.replace(".TextGrid", ".txt"), labels)


def merge_files(path, final_file_features, final_file_labels=NO_STRING):
    mfcc_suffix = ".htk"
    txt_suffix = ".txt"

    # create the final features and labels files
    if os.path.exists(final_file_features):
        os.remove(final_file_features)
    fid = open(final_file_features, 'w')
    fid.close()

    if final_file_labels != NO_STRING:
        if os.path.exists(final_file_labels):
            os.remove(final_file_labels)
        fid = open(final_file_labels, 'w')
        fid.close()
    # =================================== #

    features_path = path + features_dir
    labels_path = path + labels_dir

    if not os.path.exists(path):
        print >> sys.stderr, "Directory does not exists"
    item_num = 1
    for item in os.listdir(features_path):
        if item.endswith(mfcc_suffix):
            abs_path_features = os.path.abspath(features_path + item)
            # read the mfcc features
            reader = HTKFeat_read(abs_path_features)
            matrix = reader.getall()
            # write the merged files
            f_handle = file(final_file_features, 'a')
            np.savetxt(f_handle, matrix)
            f_handle.close()

            # handel labels if exists
            if final_file_labels != NO_STRING:
                abs_path_labels = os.path.abspath(labels_path + (item.replace(mfcc_suffix, txt_suffix)))
                labels = np.loadtxt(abs_path_labels)
                f_handle = file(final_file_labels, 'a')
                np.savetxt(f_handle, labels)
                f_handle.close()

            print("\rProcessing item number: %d" % item_num),
            item_num += 1


# the main function
def main(in_path_x, in_path_y):
    """
    Extract the mfcc features and labels from the audio and text grid files
    Both the TextGrid and audio files should be at the same directory
    :param in_path_x: the path to the audio files directory
    :param in_path_y: the path to the text grid files directory
    """
    if not os.path.exists(in_path_x):
        print >> sys.stderr, "X Directory does not exists"
        return

    extract_mfcc_dir(in_path_x, tmp_dir + features_dir)  # extract the features
    if os.path.exists(in_path_y):
        create_labels(in_path_y, tmp_dir + labels_dir)  # create the labels


# clean the tmp files
def clean(init=False):
    # clean tmp folder
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    if init:
        os.mkdir(tmp_dir)


if __name__ == "__main__":
    # command line arguments
    parser = argparse.ArgumentParser(description="This script extract the mfcc features from a given audio file.")
    parser.add_argument("in_path_x", help="The path to the audio files")
    parser.add_argument("out_path", help="The path to save the mfcc's and labels, the saved file will be a "
                                         "pickle file for both features and labels")
    parser.add_argument("--in_path_y", help="The path to the text grid files if possible", default='$$')
    args = parser.parse_args()

    # x, y = ds.get_data('data/files/db.test.naming')
    # np.savetxt(args.out_path + merge_file_x, x)
    # np.savetxt(args.out_path + merge_file_y, y)

    clean(init=True)
    main(args.in_path_x, args.in_path_y)
    if args.in_path_y != NO_STRING:
        merge_files(tmp_dir, args.out_path + merge_file_x, args.out_path + merge_file_y)
    else:
        merge_files(tmp_dir, args.out_path + merge_file_x)
    clean()
