import pandas as pd
import matplotlib.pyplot as plt
import os
from cycler import cycler

def plot_by_type(plot_name="", path="", fontsize=14):
    param_prefixes = ["50000_10", "100000_10","500000_10",
                      "1000000_10", "5000000_10"]
    labels = ['50k', '100k', '500k', '1M', '5M']
    columns_to_plot = ['size']

    file_mappings = {
        'IVCSC': "_ivcsc_results.csv",
        'VCSC': "_vcsc-results.csv"
    }

    for dtype, file_suffix in file_mappings.items():
        plt.figure(figsize=(7, 5))
        if dtype == 'IVCSC':
            clist = ['mediumblue', 'hotpink', 'lime', 'tab:orange',  'teal', 'crimson' ]

            plt.rc('axes', prop_cycle=(cycler('color', clist) +  
                                       cycler('linestyle', ['-', (0,(8,4)), '-.', '--', (0,(1,1)), (0,(3,5,1,5))])))
        elif dtype == 'VCSC':
            clist = ['mediumblue', 'lime', 'orchid', 'tab:orange',  'teal', 'crimson' ]

            plt.rc('axes', prop_cycle=(cycler('color', clist) +  
                                       cycler('linestyle', ['-', (0,(8,4)), '-.', '--', (0,(1,1)), (0,(3,5,1,5))])))

        for column in columns_to_plot:

            for i, param_prefix in enumerate(param_prefixes):
                df = pd.read_csv(path + f'{param_prefix}{file_suffix}')
                df_avg = df.groupby('ID').mean().reset_index()

                if column == 'size':
                    dense_size = df['rows'].values[0] * df['cols'].values[0] * 4
                    plt.plot(df_avg['redundancy'], df_avg[column]/dense_size, label=labels[i])
                    plt.ylabel('Size Ratio over Dense', fontsize=fontsize+2)
                    
                else:
                    plt.plot(df_avg['redundancy'], df_avg[column], label=labels[i])
                    plt.ylabel(column, fontsize=fontsize+2)
                    
                    

            plt.xlabel('Redundancy', fontsize=fontsize+2)
            plt.legend(handlelength=5, prop={'size': fontsize})
            plt.xticks(fontsize=fontsize-1)
            plt.yticks(fontsize=fontsize-1)

            print('saving in directory: ' + path + 'plots/')
            os.makedirs(path + 'plots/' + f"semilogy_together_plots_{plot_name}/", exist_ok=True)
            os.makedirs(path + 'plots/' + f"together_plots_{plot_name}/", exist_ok=True)
            plt.savefig(path + 'plots/' + f"together_plots_{plot_name}/{dtype}_{column}.pdf", bbox_inches='tight')
            plt.semilogy()
            plt.savefig(path + 'plots/' + f"semilogy_together_plots_{plot_name}/{dtype}_{column}.pdf", bbox_inches='tight')
            plt.clf()

if __name__ == '__main__':
    #plt.rcParams['text.usetex'] = True
    #clist = ['tab:orange', 'cyan', 'darkviolet', 'mediumseagreen']
    #clist = ['lightcoral', 'lime', 'mediumblue', 'tab:orange', 'orchid', 'teal' ]

    plt.rc('lines', linewidth=2.5)
    plot_by_type(plot_name='floats', path='../results-float-O3-25cols/')
