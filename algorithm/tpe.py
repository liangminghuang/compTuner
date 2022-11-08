import argparse
import os, sys
import time
import random
import numpy as np
from hyperopt import fmin, tpe, hp, space_eval, rand, Trials, partial, STATUS_OK
random.seed(123)
iters = 180
begin2end = 3


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Args needed for BOCA tuning compiler.")
    parser.add_argument('--bin-path',
                        help='Specify path to compilation tools.',
                        metavar='<directory>', required=True)
    parser.add_argument('--driver',
                        help='Specify name of compiler-driver.',
                        metavar='<bin>', required=True)
    parser.add_argument('--linker',
                        help='Specify name of linker.',
                        metavar='<bin>', required=True)
    parser.add_argument('--libs',
                        help='Pass comma-separated <options> on to the compiler-driver.',
                        nargs='*', metavar='<options>', default='')
    parser.add_argument('-o', '--output',
                        help='Write output to <file>.',
                        default='a.out', metavar='<file>')
    parser.add_argument('-p', '--execute-params',
                        help='Pass comma-separated <options> on to the executable file.',
                        nargs='+', metavar='<options>')
    parser.add_argument('-src', '--src-dir',
                        help='Specify path to the source file.',
                        required=True, metavar='<directory>')
    args = parser.parse_args()

    from .executor import Executor, LOG_DIR

    if not os.path.exists(LOG_DIR):
        os.system('mkdir ' + LOG_DIR)

    make_params = {}
    bin_path = args.bin_path
    if not bin_path.endswith(os.sep):
        make_params['bin_path'] = args.bin_path
    else:
        make_params['bin_path'] = args.bin_path[:-1]
    make_params['driver'] = args.driver
    make_params['linker'] = args.linker
    if args.libs:
        make_params['libs'] = args.libs
    make_params['output'] = args.output
    if args.execute_params:
        make_params['execute_params'] = args.execute_params
    make_params['src_dir'] = args.src_dir

    e = Executor(**make_params)
    space = {}
    stats = []
    times = []
    for option in e.o3_opts:
        space[option] = hp.choice(option, [0, 1])

    algo = partial(tpe.suggest, n_startup_jobs=1)
    for i in range(begin2end):
        process = []
        ts = []
        b = time.time()
        best = fmin(e.get_objective_score, space, algo=algo, max_evals=iters)
        stats.append(process)
        times.append(ts)
        print(best)
        print(e.get_objective_score(best))
        print(process)

    vals = []
    for j, v_tmp in enumerate(stats):
        max_s = 0
        for i, v in enumerate(v_tmp):
            max_s = min(max_s, v)
            v_tmp[i] = max_s
    print(times)
    print(stats)

    for i in range(iters):
        tmp = []
        for j in range(begin2end):
            tmp.append(times[j][i])
        vals.append(-np.mean(tmp))

    print(vals) 

    vals = []
    for i in range(iters):
        tmp = []
        for j in range(begin2end):
            tmp.append(stats[j][i])
        vals.append(-np.mean(tmp))

    print(vals)

    vals = []
    for i in range(iters):
        tmp = []
        for j in range(begin2end):
            tmp.append(stats[j][i])
        vals.append(-np.std(tmp))

    print(vals)
