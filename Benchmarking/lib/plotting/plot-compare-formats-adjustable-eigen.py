import pandas as pd
import matplotlib.pyplot as plt
import os
from cycler import cycler


path = '../results-float-O3-25cols/'
eigen_suffix = ''
path_egien = path[:-1] + '-' + eigen_suffix
combos = ['5000_10',
          '10000_10', "50000_10",
          "100000_10", "500000_10",
          '1000000_10', '2500000_10', '5000000_10',
          '10000000_10']

FONTSIZE = 14

for combo in combos:

    clist = ['tab:orange', 'cyan', 'darkviolet']
    loose_dash_dot = (0, (4,4,1,4))

    plt.rc('axes', prop_cycle=(cycler('color', clist) + 
                               cycler('linestyle', ['--', '-', (0,(1,1))])))

    plt.rc('lines', linewidth=2.5)
    df1 = pd.read_csv(path[:-1] + '-' + eigen_suffix + f'{combo}_eigen_results.csv')
    noneigen_path = path

    df2 = pd.read_csv(noneigen_path + f'{combo}_ivcsc_results.csv')
    df3 = pd.read_csv(noneigen_path + f'{combo}_vcsc-results.csv')

    plot_path = noneigen_path + eigen_suffix
    os.makedirs(plot_path + 'plots/',  exist_ok=True)

    df1_std = df1.groupby('ID').std().reset_index()
    df2_std = df2.groupby('ID').std().reset_index()
    df3_std = df3.groupby('ID').std().reset_index()

    df1_avg = df1.groupby('ID').mean().reset_index()
    df2_avg = df2.groupby('ID').mean().reset_index()
    df3_avg = df3.groupby('ID').mean().reset_index()

    columns_to_plot = ['size','constructor_time', 'scalar_time', 'spmv_time', 'spmm_time', 'transpose_time', 'iterator_time']
    ylabels = ['Size (bytes)', 'Time (sec)', 'Time (sec)', 'Time (sec)', 'Time (sec)', 'Time (sec)', 'Time (sec)']

    columns_to_plot = ['size']
    ylabels = ['Size Ratio over Dense']

    for idx, column in enumerate(columns_to_plot):

        if column == 'spmv_time':
            #plt.figure(figsize=(5.6, 4.7))
            plt.figure(figsize=(5.5,4.5))
        elif column == 'spmm_time':
            plt.figure(figsize=(5.0, 4.4))
        elif column == 'size':
            plt.figure(figsize=(7, 5))
        else:
            plt.figure(figsize=(5.5, 4.5))
    
        if False:
            plt.plot(df2_avg['redundancy'], df2_std[column]/df2_avg[column], label='IVCSC')
            plt.plot(df3_avg['redundancy'], df3_std[column]/df3_avg[column], label='VCSC')
            plt.plot(df1_avg['redundancy'], df1_std[column]/df1_avg[column], label='Eigen')

        if True and column != 'size':
            plt.plot(df2_avg['redundancy'], df2_avg[column], label='IVCSC')#, marker='+')
            plt.plot(df3_avg['redundancy'], df3_avg[column], label='VCSC')#, marker='x')
            plt.plot(df1_avg['redundancy'], df1_avg[column], label='Eigen')#, marker='.')
        if column == 'size':
            dense_size = df1['rows'].values[0] * df1['cols'].values[0] * 4
            plt.plot(df2_avg['redundancy'], df2_avg[column]/dense_size, label='IVCSC')#, marker='+')
            plt.plot(df3_avg['redundancy'], df3_avg[column]/dense_size, label='VCSC')#, marker='x')
            plt.plot(df1_avg['redundancy'], df1_avg[column]/dense_size, label='CSC')#, marker='.')
            plt.legend(loc=0, prop={'size': FONTSIZE})


        #plt.title(f'{column} vs. Redundancy')
        plt.xlabel('Redundancy', fontsize=FONTSIZE+2)
        plt.ylabel(ylabels[idx], fontsize=FONTSIZE+2)
        #plt.legend(loc=1,prop={'size': FONTSIZE})
        plt.xticks(fontsize=FONTSIZE-1)
        plt.yticks(fontsize=FONTSIZE-1)
        plt.tight_layout()
        fname = plot_path + f"plots/{combo}_{column}.pdf"
        print(f'saving as {fname}')
       
        
        label_params = plt.gca().get_legend_handles_labels() 
        plt.savefig(fname, bbox_inches='tight')

    figl, axl = plt.subplots()
    axl.axis(False)
    axl.legend(*label_params, loc="center", bbox_to_anchor=(0.5, 0.5), prop={"size":FONTSIZE})
    figl.savefig(plot_path + "/plots/legend.pdf", bbox_inches='tight')
            
