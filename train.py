#!/usr/bin/env python3
import argparse
import csv
import numpy as np
import tensorflow as tf

from ids_conv import get_converter


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', '-e', required=True)
    parser.add_argument('--sources', '-s', nargs='+', type=int,
                        choices=range(5), required=True)
    parser.add_argument('--target', '-t', type=int,
                        choices=range(1, 5), required=True)
    parser.add_argument('--training_set', required=True)
    parser.add_argument('--dev_set')
    parser.add_argument('--test_set', required=True)
    parser.add_argument('--iters', '-i', type=int, default=20)
    parser.add_argument('--learning_rate', '-lr', type=float, required=True)
    parser.add_argument('--decay', '-l2', type=float)
    parser.add_argument('--batch_size', '-b', type=int, required=True)
    parser.add_argument('--idrop', '-id', type=float)
    parser.add_argument('--rdrop', '-rd', type=float)
    parser.add_argument('--hid_size', '-hs', type=int)
    parser.add_argument('--n_layers', '-n', type=int)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--stats')

    return parser.parse_args()


def main(args):
    sources_name = {0: 'CHAR', 1: 'mandarin', 2: 'cantonese',
                    3: 'korean', 4: 'vietnamese'}

    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    if '2' in args.experiment:
        import experiment.exp2x
        exp_type = getattr(experiment.exp2x, args.experiment)
    elif '1' in args.experiment:
        import experiment.exp1x
        exp_type = getattr(experiment.exp1x, args.experiment)

    print(exp_type.__name__, 'experiment')
    print('Sources:', *[sources_name[s] for s in args.sources])
    print('Target:', sources_name[args.target])

    conv = get_converter()
    args = vars(args)
    args.pop('experiment')
    if args['dev_set']:
        exp = exp_type(conv, **args).train(args['training_set'], args['dev_set'])
    else:
        exp = exp_type(conv, **args).train(args['training_set'])

    test_error = exp.test(args['test_set'])
    print('Test errors: SER %.1f WER %.1f %.1f %.1f %.1f' % tuple(test_error))

    if args['stats']:
        with open(args['stats']) as csvfile:
            header = csvfile.readline().strip().split(',')
        with open(args['stats'], 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=header)
            params = {f: args.get(f, None) for f in header}
            params.update({'ser': test_error[0], 'ter': test_error[1]})
            writer.writerow(params)


if __name__ == '__main__':
    main(get_args())
