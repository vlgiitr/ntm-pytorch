import argparse


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-task_json', type=str, default='ntm/tasks/copy.json',
                        help='path to json file with task specific parameters')
    parser.add_argument('-saved_model', default='saved_model_copy.pt',
                        help='path to file with final model parameters')
    parser.add_argument('-batch_size', type=int, default=1,
                        help='batch size of input sequence during training')
    parser.add_argument('-num_iters', type=int, default=100000,
                        help='number of iterations for training')

    # todo: only rmsprop optimizer supported yet, support adam too
    parser.add_argument('-lr', type=float, default=1e-4,
                        help='learning rate for rmsprop optimizer')
    parser.add_argument('-momentum', type=float, default=0.9,
                        help='momentum for rmsprop optimizer')
    parser.add_argument('-alpha', type=float, default=0.95,
                        help='alpha for rmsprop optimizer')
    parser.add_argument('-beta1', type=float, default=0.9,
                        help='beta1 constant for adam optimizer')
    parser.add_argument('-beta2', type=float, default=0.999,
                        help='beta2 constant for adam optimizer')
    return parser
