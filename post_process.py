import argparse
import numpy as np
import os


def predict_duration_target(y):
    """
    This function get all the voice activities that were detected in the signal
    :param y: the network predictions
    :return: list of the voice activities
    """
    predictions = []
    onset_found = False
    onset = 0
    for i in range(1, len(y)):
        prev = y[i-1]
        curr = y[i]
        if prev is 1 and curr is 2:
            onset_found = True
            onset = i
        if prev is 2 and curr is 1 and onset_found:
            onset_found = False
            predictions.append([onset, i - 1])
    return predictions


def max_duration(predictions):
    max_value = 1
    max_idx = -1
    for i, p in enumerate(predictions):
        tmp = p[1] - p[0]
        if tmp > max_value:
            max_value = tmp
            max_idx = i
    if max_idx != -1:
        return predictions[max_idx]
    else:
        print('No predictions')
        return [0, 0]


def smooth_duration(predictions):
    max_idx = -1
    max_length = 0
    onset = 0
    offset = 0
    final_b = 0
    final_r = 0
    for i, p in enumerate(predictions):
        if onset == 0 and offset == 0:
            onset = p[0]
            offset = p[1]
        else:
            if p[0] - offset < 10:
                offset = p[1]
            else:
                tmp_len = offset - onset
                if offset - onset > max_length:
                    max_idx = i
                    final_b = onset
                    final_r = offset
                    max_length = tmp_len
                onset = 0
                offset = 0
    if max_idx == -1 and (offset != 0 or onset != 0):
        max_idx = 1
        final_b = onset
        final_r = offset

    if max_idx != -1:
        return [final_b, final_r]
    else:
        print('No predictions')
        return [0, 0]


def post_process(filename, output_path):
    """
    Computes the duration from the network predictions
    :param filename: the predictions file path
    :param output_path: the path to write the final duration
    """
    x_file = os.path.abspath(filename)
    output_path = os.path.abspath(output_path)

    classifications = list()
    # parsing the predictions file
    with open(x_file) as f:
        lines = f.readlines()
        for line in lines:
            classifications.append(int(line[:-1]))
    f.close()

    predictions = predict_duration_target(classifications)
    # prediction = max_duration(predictions)
    prediction = smooth_duration(predictions)
    with open(output_path, 'w') as fid:
        fid.write(str(prediction[0]) + ' ' + str(prediction[1])+'\n')
    fid.close()


if __name__ == "__main__":
    # -------------MENU-------------- #
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="The path to the prediction file that were generated from the nn")
    parser.add_argument("output_path", help="The path to save the label")
    args = parser.parse_args()

    # main function
    post_process(args.filename, args.output_path)
