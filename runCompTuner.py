# encoding: utf-8
import os
import random
import argparse

from algorithm.CompTuner import compTuner

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Args needed for BOCA tuning compiler.")
    # compilation params
    parser.add_argument('--bin-path',
                        help='Specify path to compilation tools.',
                        metavar='<directory>', required=True)
    parser.add_argument('--driver',
                        help='Specify name of compiler-driver.',
                        metavar='<bin>', required=True)
    parser.add_argument('--linker',
                        help='Specify name of linker.',
                        metavar='<bin>',required=True)
    parser.add_argument('--libs',
                        help='Pass comma-separated <options> on to the compiler-driver.',
                        nargs='*', metavar='<options>',default='')
    parser.add_argument('-o', '--output',
                        help='Write output to <file>.',
                        default='a.out', metavar='<file>')
    parser.add_argument('-p', '--execute-params',
                        help='Pass comma-separated <options> on to the executable file.',
                        nargs='+', metavar='<options>')
    parser.add_argument('-src', '--src-dir',
                        help='Specify path to the source file.',
                        required=True, metavar='<directory>')

    # compTuner params
    parser.add_argument('--c1',
                        help='Specify the scale of the PSO process (2 by default).',
                        type=float, default=2, metavar='<num>')
    parser.add_argument('--c2',
                        help='Specify the scale of the PSO process (2 by default).',
                        type=float, default=2, metavar='<num>')
    parser.add_argument('--w',
                        help='Specify the scale of the PSO process (0.6 by default).',
                        type=float, default=0.6, metavar='<num>')
    parser.add_argument('-i', '--iteration',
                        help='Number of total instances, including initial sampled ones (60 by default).',
                        type=int, default=600, metavar='<iteration>')
    parser.add_argument('--random',
                        help='Fix <random> for random process and model building.',
                        type=int, default=456, metavar='<random>')
                        

#
#
#
#
    args = parser.parse_args()
    if args.random:
        random.seed(args.random)
    from algorithm.executor import Executor, LOG_DIR

    if not os.path.exists(LOG_DIR):
        os.system('mkdir '+LOG_DIR)


    make_params = {}
    pso_params = {}
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
    # if args.obj_dir:
    #     make_params['obj_dir'] = args.obj_dir

    e = Executor(**make_params)
    tuning_list = e.o3_opts

    print(len(tuning_list), tuning_list)

    pso_params['dim'] = len(tuning_list)
    pso_params['get_objective_score'] = e.get_objective_score
    pso_params['c1'] = args.c1
    pso_params['c2'] = args.c2
    pso_params['w'] = args.w
    pso_params['iteration'] = args.iteration
    pso_params['random'] = args.random
#
#
    CompTuner = compTuner(**pso_params)
    #重复实验次数
    begin2end = 3
    stats = []
    times = []
    for _ in range(begin2end):
        dep, ts = CompTuner.run()
        print('middle result')
        print(dep)
        stats.append(dep)
        times.append(ts)
    for j, v_tmp in enumerate(stats):
        max_s = 0
        for i, v in enumerate(v_tmp):
            max_s = max(max_s, v)
            v_tmp[i] = max_s
    print(times)
    print(stats)

    time_mean = []
    time_std = []
    stat_mean = []
    stat_std = []
    import numpy as np
    for i in range(args.budget):
        time_tmp = []
        stat_tmp = []
        for j in range(begin2end):
            time_tmp.append(times[j][i])
            stat_tmp.append(stats[j][i])
        time_mean.append(np.mean(time_tmp))
        time_std.append(np.std(time_tmp))
        stat_mean.append(np.mean(stat_tmp))
        stat_std.append(np.std(stat_tmp))
    print(time_mean)
    print(time_std)
    print(stat_mean)
    print(stat_std)
