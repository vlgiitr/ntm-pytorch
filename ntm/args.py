import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-task_json', default='tasks/copy.json',
                        help='path to json file with task specific parameters')
    parser.add_argument('-batch_size', default=1,
                        help='batch size of input sequence during training')
    parser.add_argument('-num_iters', default=100000,
                        help='number of iterations for training')

    # todo: only rmsprop optimizer supported yet, support adam too
    parser.add_argument('-lr', default=1e-4,
                        help='learning rate for rmsprop optimizer')
    parser.add_argument('-momentum', default=0.9,
                        help='momentum for rmsprop optimizer')
    parser.add_argument('-alpha', default=0.95,
                        help='alpha for rmsprop optimizer')
    return parser
