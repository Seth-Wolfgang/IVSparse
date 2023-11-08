import scipy.sparse as spp
import numpy as np
import math
import os
import matplotlib.pyplot as plt
from cycler import cycler
import argparse
import subprocess


def replace_with_sed(fname, var, num):
    subprocess.call(['sed', 
                     '-i',
                     f'/#define {var}/c\#define {var} {num}',
                     fname])

def update_cpp_file(fname, nrows, ncols, density_pct):
    replace_with_sed(fname, 'ROWS', nrows)
    replace_with_sed(fname, 'COLS', ncols)
    replace_with_sed(fname, 'NNZ', int(density_pct/100 * nrows))
    subprocess.call(f'g++ -O3 --std=c++17 -I ~/eigen {fname} -o {fname[:-4]}.out', shell=True)
    cmd_list = ['g++', '-O3', '--std=c++17', '-I', '~/eigen', fname, '-o', fname[:-4] + '.out']

def run_cpp(mat, redundancy, idnum, cpp_results_path, cpp_fname, which='all'):
    rowinds_fname = '/tmp/rowinds.csv'
    colptrs_fname = '/tmp/colptrs.csv'
    values_fname = '/tmp/values.csv'

    # save files
    np.savetxt(rowinds_fname, mat.indices, fmt='%d')
    np.savetxt(colptrs_fname, mat.indptr, fmt='%d')
    np.savetxt(values_fname, mat.data, fmt='%.16f')

    # convert which to number
    if which == 'all':
        which_num = -1
    elif which == 'vcsc':
        which_num = 0
    elif which == 'ivcsc':
        which_num = 1
    elif which == 'eigen':
        which_num = 2
    else:
        raise NotImplementedError

    # call cpp program
    #subprocess.call(['./' + cpp_fname[:-4] + '.out', values_fname, rowinds_fname, colptrs_fname, str(redundancy), str(idnum), str(which_num), cpp_results_path])
    cmd_list = ['./' + cpp_fname[:-4] + '.out', values_fname, rowinds_fname, colptrs_fname, str(redundancy), str(idnum), str(which_num), cpp_results_path]
    subprocess.call(' '.join(cmd_list), shell=True)
    subprocess.call(['rm', values_fname, rowinds_fname, colptrs_fname])

def adjust(spmat, mmr):
    for i in range(len(spmat.indptr) - 1):
        colvals = spmat.data[spmat.indptr[i]:spmat.indptr[i + 1]]
        nnz_col = len(colvals)
        nuniq_col = math.floor((1 - mmr) * nnz_col + 1)
        spmat.data[spmat.indptr[i]:spmat.indptr[i + 1]] = np.random.permutation(nnz_col) % nuniq_col + 1


def compute_mmr(spmat, bytes_per_index=4, bytes_per_val=8, debug=False):
    mmr_avg = 0
    ivcsc_size = 0
    vcsc = 0
    avg_difflog = 0  # new metric

    zero_col_count = 0
    nnz_col_avg = 0
    nuniq_col_avg = 0
    # print(spmat.tocoo())
    for i in range(len(spmat.indptr) - 1):
        colvals = spmat.data[spmat.indptr[i]:spmat.indptr[i + 1]]
        if len(colvals) == 0:
            zero_col_count += 1
            continue
        nnz_col = len(colvals)
        nuniq_col = len(np.unique(colvals))
        mmr_avg += 1 - (nuniq_col - 1) / nnz_col
        if debug:
            print(f'nnz_col = {nnz_col}, nuniq_col = {nuniq_col}')
        nnz_col_avg += nnz_col
        nuniq_col_avg += nuniq_col
        # vcsc size computation
        vcsc += nuniq_col * bytes_per_val
        vcsc += nnz_col * bytes_per_index
        vcsc += nuniq_col * bytes_per_index
        vcsc += bytes_per_index # extra chunk of memory to store number of unique values

        # ivcsc size computation
        rowindices = spmat.indices[spmat.indptr[i]:spmat.indptr[i + 1]]
        # print('rowindices:', rowindices, len(rowindices))
        if len(colvals) == nuniq_col:
            # handle the all unique values case
            runmax = np.ones(nuniq_col) * rowindices
            runcounts = np.ones(nuniq_col)
        elif nuniq_col == 1:
            # handle the fully redundant case (one unique value in the column)
            runmax = np.array([np.max(rowindices[1:] - rowindices[:-1])])
            runcounts = np.array([nnz_col])
        else:
            ordered = np.lexsort((rowindices, colvals))
            splits = np.where((colvals[ordered][1:] - colvals[ordered][:-1]) != 0)[0]
            splits += 1
            ordered_rowinds = rowindices[ordered]
            # stores maximum positive delta encoded index for each run
            runmax = np.zeros(len(splits) + 1)
            runcounts = np.zeros_like(runmax)
            for j in range(len(splits) + 1):
                # get split_inds -- row indices for this value
                if j == 0:
                    split_inds = ordered_rowinds[0:splits[j]]
                elif j == len(splits):
                    split_inds = ordered_rowinds[splits[j - 1]:]

                else:
                    split_inds = ordered_rowinds[splits[j - 1]:splits[j]]


                if len(split_inds) == 1:
                    # only 1 of this value, max index for this run is just the index
                    runmax[j] = split_inds[0]
                    runcounts[j] = 1
                else:
                    # get max index for this run as max delta or first index if larger
                    runmax[j] = max(split_inds[0], np.max(split_inds[1:] - split_inds[:-1]))
                    # runmax[j] = max(split_inds[0], np.max(np.diff(split_inds)))
                    if j == 0:
                        # on first split
                        runcounts[j] = splits[j]
                    elif j == len(splits):
                        # on last split
                        runcounts[j] = len(split_inds)
                    else:
                        # on a middle split
                        runcounts[j] = (splits[j] - splits[j - 1])
        if runmax.shape[0] > 50 and debug:
            print('fist 25 of runmax =', np.sort(runmax)[:25])
            print('last 25 of runmax =', np.sort(runmax)[-25:])

        bytes_needed = np.ceil(np.log(runmax + 1) / np.log(256))
        if debug:
            print('min bytes needed = {}, max bytes needed = {}, average bytes needed = {}'.format(bytes_needed.min(),
                                                                                               bytes_needed.max(),
                                                                                               bytes_needed.mean()))
        ivcsc_size += np.sum(bytes_needed * (runcounts + 1))
        ivcsc_size += bytes_per_val + nuniq_col * (bytes_per_val + 1)

        difflog = np.log10(nnz_col) - np.log10(nuniq_col)  # new metric involves difference of magnitudes
        avg_difflog += 1 - 1 / (difflog + 1)  # this scales so it is between 0 and 1


    avg_difflog /= (len(spmat.indptr) - 1 - zero_col_count)  # new metric
    mmr_avg /= (len(spmat.indptr) - 1 - zero_col_count)
    return mmr_avg, vcsc, ivcsc_size, avg_difflog, (bytes_needed.min(), bytes_needed.max(), bytes_needed.mean()), zero_col_count, (nnz_col_avg/(len(spmat.indptr) - 1), nuniq_col_avg/(len(spmat.indptr) - 1))

def calc_csc_size(spmat, bytes_per_index=4, bytes_per_val=8):
    return bytes_per_val * len(spmat.data) + bytes_per_index * len(spmat.indptr) + bytes_per_index * len(spmat.indices)


def calc_coo_size(spmat, bytes_per_index=4, bytes_per_val=8):
    # bytes_per_index for row for each, bytes_per_index for col for each, bytes_per_val for val for each
    return (bytes_per_val + 2 * bytes_per_index) * len(spmat.data)


def variable_mmr(nrows, ncols, density, nmats=50, save_data=False, start_mmr=0):
    mat = spp.rand(nrows, ncols, density=density, format='csc', random_state=42)
    mmrs = []
    ivcsc_sizes = []
    vcsc_sizes = []
    csc_sizes = []
    coo_sizes = []
    avg_difflogs = []
    temp1 = np.linspace(start_mmr, .6, nmats // 2)
    temp2 = np.linspace(0.6, 0.8, nmats // 2)
    temp3 = np.linspace(0.8, 0.9, nmats // 2)
    temp3 = np.linspace(0.9, 0.99, nmats // 2)
    temp3 = np.array([0.995, 0.999, 0.9995, 0.9999,
                      0.999925, 0.99995, 0.999975, 0.99999, 0.9999925, 0.999995, 0.9999975,
                      0.999999, 0.99999925, 0.9999995, 0.99999975, 0.9999999, 0.999999925,
                      0.99999995, 0.999999975, 0.99999999])
    mmrs_to_run = np.concatenate((temp1, temp2, temp3))
    # for mmr in np.linspace(start_mmr, 1, nmats):
    for mmr in mmrs_to_run:
        adjust(mat, mmr)
        computed_mmr, vcsc_size, ivcsc_size, avg_difflog, = compute_mmr(mat)
        mmrs.append(mmr)
        vcsc_sizes.append(vcsc_size)
        ivcsc_sizes.append(ivcsc_size)
        csc_sizes.append(calc_csc_size(mat))
        coo_sizes.append(calc_coo_size(mat))
        avg_difflogs.append(avg_difflog)
        print('target mmr = {}, actual mmr = {}, diff = {}'.format(mmr, computed_mmr, abs(mmr - computed_mmr)))
        if save_data:
            os.makedirs(f"./mmr/{computed_mmr}", exist_ok=True)
            np.savetxt(f"./mmr/{computed_mmr}/outer.csv", mat.indptr, delimiter=',')
            np.savetxt(f"./mmr/{computed_mmr}/inner.csv", mat.indices, delimiter=',')
            np.savetxt(f"./mmr/{computed_mmr}/vals.csv", mat.data, delimiter=',')

    size_names = ['vcsc', 'ivcsc', 'csc', 'coo']
    return mmrs, (vcsc_sizes, ivcsc_sizes, csc_sizes, coo_sizes), size_names, avg_difflogs


def nuniq_adjust(spmat, prev_nuniq=None):
    for i in range(len(spmat.indptr) - 1):
        colvals = spmat.data[spmat.indptr[i]:spmat.indptr[i + 1]]
        nnz_col = len(colvals)
        if prev_nuniq is None:
            nuniq_col = nnz_col
        elif prev_nuniq < 50:
            nuniq_col = prev_nuniq - 1
        elif prev_nuniq < 2500:
            nuniq_col = prev_nuniq - 10
        else:
            nuniq_col = math.floor(.9 * min(prev_nuniq, nnz_col)) + 1
        spmat.data[spmat.indptr[i]:spmat.indptr[i + 1]] = np.random.permutation(nnz_col) % nuniq_col + 1
    return nuniq_col



def variable_nuniq(nrows, ncols, density, save_data=False, bytes_per_index=4, bytes_per_val=8, benchmark=False, offset=0, cpp_results_path=None, cpp_fname='simulatedBench_COO.cpp'):
    state = offset

    mat = spp.rand(nrows, ncols, density=density, format='csc', random_state=state)
    mmrs = []
    ivcsc_sizes = []
    vcsc_sizes = []
    csc_sizes = []
    coo_sizes = []
    avg_difflogs = []
    first = True
    prev_nuniq_cpp = -1
    prev_bytes_needed = (-1,-1,-1)
    while True:
        # advance state
        state += 1

        # adjust random state
        np.random.seed(state)

        if first:
            nuniq_col = nuniq_adjust(mat)
            first = False
        else:
            nuniq_col = nuniq_adjust(mat, nuniq_col)

        computed_mmr, vcsc_size, ivcsc_size, avg_difflog, bytes_needed_info = compute_mmr(mat)

        skipped_too_many = abs(prev_nuniq_cpp - nuniq_col) > 0.1*nuniq_col
        changed_bytes_needed = abs(bytes_needed_info[2] - prev_bytes_needed[2]) > 1e-8

        if benchmark and (skipped_too_many or changed_bytes_needed):
            prev_nuniq_cpp = nuniq_col
            prev_bytes_needed = bytes_needed_info
            # Run C++ code
            run_cpp(mat, avg_difflog, state, cpp_results_path, cpp_fname, which='all')

        mmrs.append(computed_mmr)
        vcsc_sizes.append(vcsc_size)
        ivcsc_sizes.append(ivcsc_size)
        csc_sizes.append(calc_csc_size(mat,
                                       bytes_per_index=bytes_per_index,
                                       bytes_per_val=bytes_per_val))
        coo_sizes.append(calc_coo_size(mat,
                                       bytes_per_index=bytes_per_index,
                                       bytes_per_val=bytes_per_val))

        avg_difflogs.append(avg_difflog)
        print('mmr = {}, avg_difflog = {}'.format(computed_mmr, avg_difflog))
        if nuniq_col <= 1:
            break


    if save_data:
        os.makedirs(f"./nuniq/{nrows}_by_{ncols}/", exist_ok=True)
        np.savetxt(f"./nuniq/{nrows}_by_{ncols}/avg_difflogs.csv", np.array(avg_difflogs), delimiter=',')
        np.savetxt(f"./nuniq/{nrows}_by_{ncols}/csc_sizes.csv", np.array(csc_sizes), delimiter=',')
        np.savetxt(f"./nuniq/{nrows}_by_{ncols}/coo_sizes.csv", np.array(coo_sizes), delimiter=',')
        np.savetxt(f"./nuniq/{nrows}_by_{ncols}/vcsc_sizes.csv", np.array(vcsc_sizes), delimiter=',')
        np.savetxt(f"./nuniq/{nrows}_by_{ncols}/ivcsc_sizes.csv", np.array(ivcsc_sizes), delimiter=',')

    size_names = ['vcsc', 'ivcsc', 'csc', 'coo']
    return mmrs, (vcsc_sizes, ivcsc_sizes, csc_sizes, coo_sizes), size_names, avg_difflogs


def variable_density():
    mmr = .9
    state = 42
    min_density = .001
    start_idx = 30
    for density in np.linspace(0.001, 1 - min_density, 50)[start_idx:]:
        print('random_state = {}'.format(state + start_idx))
        mat = spp.rand(500_000, 1000, density=density, format='csc', random_state=state + start_idx)
        print('generated mat')
        adjust(mat, mmr)
        computed_mmr = compute_mmr(mat)
        print('target mmr = {}, actual mmr = {}, diff = {}'.format(mmr, computed_mmr, abs(mmr - computed_mmr)))
        os.makedirs(f"./density/{density}/", exist_ok=True)
        np.savetxt(f"./density/{density}/outer.csv", mat.indptr, delimiter=',')
        np.savetxt(f"./density/{density}/inner.csv", mat.indices, delimiter=',')
        np.savetxt(f"./density/{density}/vals.csv", mat.data, delimiter=',')
        state += 1



if __name__ == "__main__":
    plt.rcParams['text.usetex'] = True
    plt.rc('axes', prop_cycle=(cycler('color', list('mbgcr')) +
                               cycler('linestyle', ['-', '--', ':', '-.', '-.'])))
    
    # grab arguments
    parser = argparse.ArgumentParser(description='Run benchmarking')
    parser.add_argument('nrows', help='number of rows')
    parser.add_argument('density', help='density in [0,100]')
    parser.add_argument('cpp_results_path', help='path of results file')
    parser.add_argument('cpp_fname', help='cpp filename')
    args = parser.parse_args()

    bytes_per_index=4
    bytes_per_val=4
    ncols = 100
    # variable_density()
    # for num_rows in [1e2, 5e2, 1e3, 1e4, 1e5, 5e5, 1e6, 5e6, 1e7, 5e7]:
    #for num_rows in [1e4]:
    nrows = int(args.nrows)
    density_pct = int(args.density)

    # update C++ file with correct vals for nrows, ncols, nnz
    update_cpp_file(args.cpp_fname, nrows, ncols, density_pct)

    save_path = f"./figs/density_{density_pct}/new_metric/"
    os.makedirs(save_path, exist_ok=True)

    # mmrs, sizes, size_names = variable_mmr(nrows, ncols, 0.1, 50)
    mmrs, sizes, size_names, avg_difflogs = variable_nuniq(nrows, ncols,
                                                           density_pct / 100,
                                                           save_data=True,
                                                           bytes_per_index=bytes_per_index,
                                                           bytes_per_val=bytes_per_val,
                                                           benchmark=True,
                                                           offset=nrows,
                                                           cpp_results_path=args.cpp_results_path,
                                                           cpp_fname=args.cpp_fname)
    # mmrs, sizes, size_names, avg_difflogs = variable_nuniq(nrows, ncols, density_pct/100, 40, start_mmr=0)
    plt.figure()
    for i in range(len(sizes)):
        plt.plot(mmrs, np.array(sizes[i]) / (nrows * ncols * bytes_per_val), label=size_names[i], lw=2)
    plt.xlabel('mmr')
    plt.ylabel('ratio over dense')
    plt.title(f'{nrows} x {ncols}')
    plt.legend(loc=0)
    print(f'saving as figs/density_{density_pct}/dense_ratio_{nrows}_by_{ncols}.pdf')
    plt.savefig(f'figs/density_{density_pct}/dense_ratio_{nrows}_by_{ncols}.pdf')

    plt.figure()
    ivcsc_ind = size_names.index('ivcsc')
    vcsc_ind = size_names.index('vcsc')
    plt.plot(mmrs, np.array(sizes[ivcsc_ind]) / np.array(sizes[vcsc_ind]), lw=2)
    plt.title(f'{nrows} x {ncols}')
    plt.xlabel('mmr')
    plt.ylabel('ratio ivcsc/vcsc')
    plt.savefig(f'figs/density_{density_pct}/ivcsc_ratio_{nrows}_by_{ncols}.pdf')

    plt.figure()
    for i in range(len(sizes)):
        plt.plot(avg_difflogs, np.array(sizes[i]) / (nrows * ncols * bytes_per_val), label=size_names[i], lw=2)
    plt.xlabel(r'$\frac{1}{nCols} \sum_{i=1}^{nCols} \left(1 - \frac{1}{\log_{10}(nnz_i) - \log_{10}(nUniq_i) + 1} \right)$')
    plt.ylabel('ratio over dense')
    plt.title(f'{nrows} x {ncols}')
    plt.legend(loc=0)
    print(f'saving as figs/density={density_pct}/new_metric/dense_ratio_{nrows}_by_{ncols}.pdf')
    plt.savefig(f'figs/density_{density_pct}/new_metric/dense_ratio_{nrows}_by_{ncols}.pdf')

    plt.figure()
    plt.plot(avg_difflogs, np.array(sizes[ivcsc_ind])/sizes[vcsc_ind], label=size_names[i], lw=2)
    plt.xlabel(r'$\frac{1}{nCols} \sum_{i=1}^{nCols} \left(1 - \frac{1}{\log_{10}(nnz_i) - \log_{10}(nUniq_i) + 1} \right)$')
    plt.ylabel(r'$\frac{ivcsc}{vcsc}$')
    plt.title(f'{nrows} x {ncols}')
    plt.legend(loc=0)
    plt.savefig(f'figs/density_{density_pct}/new_metric/ivcsc_ratio_{nrows}_by_{ncols}.pdf')
    #print(mmrs, avg_difflogs)

