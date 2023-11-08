import importlib  
memcalc = importlib.import_module("mat-gen-mem-calc")

import itertools
import scipy.io as spio
import scipy.sparse as spp
import numpy as np

def read_single_cell(prefix='./'):
    mat = spio.mmread(prefix + 'single-cell.mtx')
    spmat = spp.csc_matrix(mat, dtype=np.uint16)
    return spmat

def read_pr02r(prefix='./'):
    mat = spio.mmread(prefix + 'PR02R.mtx')
    spmat = spp.csc_matrix(mat, dtype=np.float64)
    return spmat

def read_orkut(prefix='./'):
    mat = spio.mmread(prefix + 'com-Orkut/com-Orkut.mtx')
    spmat = spp.csc_matrix(mat, dtype=np.uint8)
    print(f'nrows = {spmat.shape[0]}, ncols = {spmat.shape[1]}, nnz = {spmat.nnz}\n')
    return spmat

def read_wos(prefix='./'):
    colptrs = np.loadtxt(prefix + 'web_of_science/WOS_colpointers.txt')
    rowinds = np.loadtxt(prefix + 'web_of_science/WOS_indices.txt')
    vals = np.loadtxt(prefix + 'web_of_science/WOS_vals.txt')
    spmat = spp.csc_matrix((vals, rowinds, colptrs), dtype=np.uint8)
    return spmat

def read_movielens_25mil(prefix='./'):
    alldata = np.loadtxt(prefix + 'movielens-ratings-25mil.csv', skiprows=1, delimiter=',')
    rows = alldata[:,0].astype(np.uint32)
    cols = alldata[:,1].astype(np.uint32)
    weights = alldata[:,2].astype(np.float32)

    uniq_rows = np.sort(np.unique(rows))
    uniq_cols = np.sort(np.unique(cols))


    mod_rows = np.copy(rows)
    for i in range(uniq_rows.shape[0]):
        mod_rows[rows == uniq_rows[i]] = i
    
    mod_cols = np.copy(cols)
    for i in range(uniq_cols.shape[0]):
        mod_cols[cols == uniq_cols[i]] = i

    spmat = spp.csc_matrix((weights, (mod_rows, mod_cols)),
                           shape=(uniq_rows.shape[0], uniq_cols.shape[0]),
                           dtype=np.float32)

    return spmat


if __name__ == '__main__':
    PREFIX = './' #PREFIX = 'real-problems/datasets/'

    f = open(PREFIX + 'data.csv', 'w')
    header = 'prob_name,avg_difflog,nrows,ncols,nnz,val_type,val_bytes,index_bytes,number_allzero_cols'
    header += ',dense_bytes,coo_bytes,csc_bytes,vcsc_bytes,ivcsc_bytes'
    header += ',coo_dense_ratio,csc_dense_ratio,vcsc_dense_ratio,ivcsc_dense_ratio'
    f.write(header + '\n')

    func_lst = [read_pr02r, read_wos, read_movielens_25mil, read_single_cell, read_orkut]
    name_lst = ['pr02r', 'web of science', 'movie lens', 'single cell', 'com-Orkut']

    for func, name in zip(func_lst, name_lst):
        spmat = func(PREFIX)
        valbytes = spmat.dtype.itemsize
        indbytes = 4
        nrows = spmat.shape[0]
        ncols = spmat.shape[1]
        nnz = spmat.nnz
        #mmr_avg, vcsc, ivcsc, avg_difflog, byte_info = memcalc.compute_mmr(spmat, 
        
        mmr_avg, vcsc, ivcsc, avg_difflog, byte_info, zero_col_count = memcalc.compute_mmr(spmat, 
                                                                           bytes_per_index=indbytes,
                                                                           bytes_per_val=valbytes,
                                                                           debug=False)
        csc = memcalc.calc_csc_size(spmat, bytes_per_index=indbytes, bytes_per_val=valbytes)
        coo = memcalc.calc_coo_size(spmat, bytes_per_index=indbytes, bytes_per_val=valbytes)
        dense = nrows*ncols*valbytes
        max_val = spmat.max()
        min_val = spmat.min()
  
        basic_info = f'{name},{avg_difflog:.16f},{nrows},{ncols},{nnz},{spmat.dtype},{valbytes},{indbytes},{zero_col_count}'
        sizes = f',{dense},{coo},{csc},{vcsc},{ivcsc}'
        ratios = f',{coo/dense},{csc/dense},{vcsc/dense},{ivcsc/dense}'
        f.write(basic_info + sizes + ratios + '\n')

    f.close()
