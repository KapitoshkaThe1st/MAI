import json
import subprocess
import argparse
import sys

def read_file(name):
    r = []
    with open(name) as f:
        for line in f:
            r.append([float(el) for el in line.strip().split()])
    return r

def get_cmap(n, name='prism'):
    return plt.cm.get_cmap(name, n)

parser = argparse.ArgumentParser(prog=sys.argv[0], description='Simple program for solving gas dynamics equation using MPI')
parser.add_argument('--config', type=str, default='config.json',
                    help='path to config file')
parser.add_argument('--verbose', '-v', action='store_true', default=False,
                    help='verbose output')
parser.add_argument('--target_t', '-t', type=float, default=0.0,
                    help='target t for to plot')
parser.add_argument('--only-plot', '-op', action='store_true', default=False,
                    help='only plot data, no recomputing needed')
parser.add_argument('--figure', '-f', default=None, type=str,
                    help='path to output figure file')
args = parser.parse_args()

with open(args.config) as config_file:
    config = json.loads(config_file.read())

executable = config['executable']
n_process = config['n_process']
hostfile = config['hostfile']
parameters = config['parameters']
grid_params = config['grid_params']

if not args.only_plot:
    cmd = ['mpirun', '-np', str(n_process), '--hostfile', hostfile, executable]
    if args.verbose:
        cmd.append('-v')

    for gp in grid_params:
        input = f"{parameters['r10']} {parameters['r20']} {parameters['eps1']} {parameters['eps2']} " \
            + f"{parameters['u1']} {parameters['u2']} {parameters['gamma']} {parameters['alpha']}" \
            + f"\n{parameters['max_t']} {gp['target_t']}\n{gp['N']} {gp['M']}\n{gp['output_file_prefix']}"

        if args.verbose:
            print(f'running: {" ".join(cmd)}')
            print(f'input:\n{input}')

        p = subprocess.run(cmd, stdout=subprocess.PIPE,
                input=input, encoding='ascii')

        if args.verbose:
            print(f'process ended with code {p.returncode}')
            print(f'process output:\n{p.stdout}')

if args.verbose:
    print('Plotting:')

import matplotlib.pyplot as plt

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True, figsize=(8, 8))

target_t = args.target_t

ax1.set_xlabel('x')
ax1.set_ylabel(f'r(x, {target_t})')

ax2.set_xlabel('x')
ax2.set_ylabel(f'u(x, {target_t})')

ax3.set_xlabel('x')
ax3.set_ylabel(f'E(x, {target_t})')

ax4.set_xlabel('x')
ax4.set_ylabel(f'eps(x, {target_t})')

ax5.set_xlabel('x')
ax5.set_ylabel(f'p(x, {target_t})')

cmap = get_cmap(len(parameters))

for i, gp in enumerate(grid_params):

    prefix = gp['output_file_prefix']
    with open(prefix + "x.txt") as f:
        x = [float(el) for el in f.readline().strip().split()]

    r = read_file(prefix + "r.txt")
    u = read_file(prefix + "u.txt")
    E = read_file(prefix + "E.txt")
    eps = read_file(prefix + "eps.txt")
    p = read_file(prefix + "p.txt")

    N = gp['N']
    M = gp['M']
    label = f'{N=} {M=}'

    if args.verbose:
        print('Plotting ' + label)

    index = int(target_t / parameters['max_t'] * len(x))
    ax1.plot(x, r[index], c=cmap(i), label=label)
    ax2.plot(x, u[index], c=cmap(i), label=label)
    ax3.plot(x, E[index], c=cmap(i), label=label)
    ax4.plot(x, eps[index], c=cmap(i), label=label)
    ax5.plot(x, p[index], c=cmap(i), label=label)

lines, labels = fig.axes[-1].get_legend_handles_labels()
    
fig.legend(lines, labels, loc = 'upper center')

if args.figure is not None:
    plt.savefig(args.figure)
    print(f'{args.figure} saved')

plt.show()

if args.verbose:
    print('Done')
