"""
AFLite implements the AFLite (Algorithm 1) in https://arxiv.org/pdf/2002.04108.pdf
Run `pip install pyyaml` before running this script
"""
import torch
import yaml

def AFLite(phi, L, n, m, t, k, tau, output):
    """
    Input:
        phi: string path to load the data from.
        L: a trainable model.
        n: target dataset size.
        m: number of times the data is partitioned into two parts.
        t: size of the part used for training L.
        k: maximum number of rows to be deleted for each iteration.
            If an iteration chose less than k, we'll remove those rows and then terminate.
        tau: (0 to 1 ) predictability score to determine if a row should be removed. If a row has predictability score more or equal to tau, it'll be removed.
            1. the size of to be removed instances is less than k.
            2. size of the remaining data set is less than n.
        output: string path to write the filter to.

    Output
        Lx1 torch.Tensor, on each row, if true, keep the datapoint, if false, remove the datapoint.
    """
    print(f'Loading data from "{phi}" and output to "{output}"')
    print(f'Model = {L}, n={n}, m={m}, t={t}, k={k}, tau={tau}')
    print('done')
    pass

def run():
    with open('config.yaml', 'r') as y:
        config = {
            'phi': 'Location to load the embeddings',
            'L': 'linear',
            'm': 64,
            't': 40000,
            'tau': 0.75,
            'k': 10000,
            'n': 640000, # relavent to size of the input
            'output': 'Location to output the filter',
        }

        models = {
            'linear': torch.nn.Linear,
        }

        for k, v in yaml.safe_load(y)['2-aflite'].items():
            config[k] = v

        model = models[config['L']]
        AFLite(config['phi'], model, config['n'], config['m'], config['t'], config['k'], config['tau'], config['output'])

if __name__ == '__main__':
    run()