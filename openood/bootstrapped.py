import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse, pickle
from scipy.stats import ttest_ind
import pandas as pd
from utils import pp

parser = argparse.ArgumentParser()

pd.set_option('display.precision', 3)

parser.add_argument('--dataset', '-d', type=str, default='cifar10')
parser.add_argument('--better_postprocessor', '-b', type=str, default='vim')
parser.add_argument('--worse_postprocessor', '-w', type=str, default='gradmean')
parser.add_argument('--table', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--generator', '-g', type=str, default='gradcam')
parser.add_argument('--aggregator', '-a', type=str, default='Norm')

args = parser.parse_args()
print(args.table)

filename = f'saved_metrics/{args.dataset}_{args.worse_postprocessor}_bootstrapped.pkl'
with open(filename, 'rb') as file:
    data = pickle.load(file)
    dataframe = data[0][0]


def get_metrics(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)

    all_metrics, all_scores = data
    names = list(all_metrics[0].index)
    metrics = np.stack([metric.to_numpy() for metric in all_metrics])
    return metrics, names


if args.better_postprocessor == 'salagg' or args.better_postprocessor == 'salpluslogit':
    better_metric, names = get_metrics(
        f'saved_metrics/{args.dataset}_{args.better_postprocessor}_{args.generator}_{args.aggregator}_bootstrapped.pkl'
    )
elif args.better_postprocessor == 'salvim':
    better_metric, names = get_metrics(
        f'saved_metrics/{args.dataset}_{args.better_postprocessor}_{args.generator}_bootstrapped.pkl'
    )
    vim_metric, names = get_metrics(
        f'saved_metrics/{args.dataset}_vim_bootstrapped.pkl'
    )

else:
    better_metric, names = get_metrics(
        f'saved_metrics/{args.dataset}_{args.better_postprocessor}_bootstrapped.pkl'
    )

worse_metric, _ = get_metrics(
    f'saved_metrics/{args.dataset}_{args.worse_postprocessor}_bootstrapped.pkl'
)

msp_metric, _ = get_metrics(f'saved_metrics/{args.dataset}_msp_bootstrapped.pkl')

mls_metric, _ = get_metrics(f'saved_metrics/{args.dataset}_mls_bootstrapped.pkl')


print(args.worse_postprocessor)
print('-' * 52)
dataframe.values[:] = worse_metric.mean(axis=0)
print(dataframe)

print()

print(args.better_postprocessor)
print('-' * 52)
dataframe.values[:] = better_metric.mean(axis=0)
print(dataframe)

print(
    f'\n{args.worse_postprocessor} has higher AUROC than {args.better_postprocessor} with the following probabilities (null hypothesis)'
)

for i in range(len(names)):
    better = better_metric[:, i, 1]
    worse = worse_metric[:, i, 1]

    pval = ttest_ind(better, worse, alternative='greater').pvalue

    padding = 10 - len(names[i])

    mean_diff = better.mean() - worse.mean()
    mean_scale = better.mean() / worse.mean()
    print(
        f'{names[i]}: {" "*padding} pval: {pval:.5f}    mean_diff: {" " if mean_diff > 0 else ""}{mean_diff:.3f}%   mean_scale: {mean_scale:.3f}'
    )

if args.table and args.better_postprocessor != 'salvim':
    print(f"""
\\begin{{table}}[H]
\setlength\\tabcolsep{{3pt}}
\\begin{{center}}
\\begin{{tabular}}{{ |c|c|c|c|c|c| }}
    \\hline
    Dataset & \\ac{{auroc}} & $\\Delta$\\ac{{auroc}} \\ac{{mls}} & P-value \\ac{{mls}} & $\\Delta$\\ac{{auroc}} \\ac{{msp}} & P-value \\ac{{msp}} \\\\
    \\hline
    \\hline
    \\rowcolor{{near!50}}
    \\multicolumn{{6}}{{|c|}}{{Near-OOD}} \\\\
    \\hline""")

    rowcolor = 'near'

    for i in range(len(names)):
        better = better_metric[:, i, 1]
        mls = mls_metric[:, i, 1]
        msp = msp_metric[:, i, 1]

        pval_mls = ttest_ind(better, mls, alternative='greater').pvalue
        pval_msp = ttest_ind(better, msp, alternative='greater').pvalue

        padding = 10 - len(names[i])

        if pval_mls > 0.05:
            pval_mls = f'{pval_mls:.3f}'
        elif pval_mls > 0.01:
            pval_mls = f'{pval_mls:.3f} *'
        elif pval_mls > 0.001:
            pval_mls = f'{pval_mls:.3f} **'
        else:
            pval_mls = '<0.001 ***'

        if pval_msp > 0.05:
            pval_msp = f'{pval_msp:.3f}'
        elif pval_msp > 0.01:
            pval_msp = f'{pval_msp:.3f} *'
        elif pval_msp > 0.001:
            pval_msp = f'{pval_msp:.3f} **'
        else:
            pval_msp = '<0.001 ***'

        mls_diff = better.mean() - mls.mean()
        msp_diff = better.mean() - msp.mean()
        print(f'    \\rowcolor{{{rowcolor}!50}}')
        if names[i] == 'nearood':
            print(
                f'    Near-OOD Avg. & {better.mean():.2f} & {"+" if mls_diff > 0 else ""}{mls_diff:.3f} & {pval_mls} & {"+" if msp_diff > 0 else""}{msp_diff:.3f} & {pval_msp} \\\\'
            )
        elif names[i] == 'farood':
            print(
                f'    Far-OOD Avg. & {better.mean():.2f} & {"+" if mls_diff > 0 else ""}{mls_diff:.3f} & {pval_mls} & {"+" if msp_diff > 0 else""}{msp_diff:.3f} & {pval_msp} \\\\'
            )
        else:
            print(
                f'    {pp(names[i])} & {better.mean():.2f} & {"+" if mls_diff > 0 else ""}{mls_diff:.3f} & {pval_mls} & {"+" if msp_diff > 0 else""}{msp_diff:.3f} & {pval_msp} \\\\'
            )
        print('    \\hline')
        if names[i] == 'nearood':
            print('    \\hline')
            print('    \\rowcolor{far!50}')
            print('    \\multicolumn{6}{|c|}{Far-OOD} \\\\')
            print('    \\hline')
            rowcolor = 'far'

    print(f"""    \end{{tabular}}
    \caption[T-test for {pp(args.generator)}{args.aggregator.capitalize()} {args.better_postprocessor} on {pp(args.dataset)}]{{Results of performing a t-test on the \\ac{{auroc}} means of {pp(args.generator)}{args.aggregator.capitalize()} {args.better_postprocessor} against \\ac{{mls}} and \\ac{{msp}}, showing the mean \\ac{{auroc}} over 10 runs on {pp(args.dataset)}, the difference in means compared to the baselines, and the corresponding p-values. Each p-value is appended a significance code which follows the \\texttt{{R}}-standard.}}
    \label{{table:{args.dataset}_{args.generator}_{args.aggregator}_{args.better_postprocessor}_ttest}}
\end{{center}}
\setlength\\tabcolsep{{6pt}}
\end{{table}}
""")

elif args.table and args.better_postprocessor == 'salvim':
    print(f"""
\\begin{{table}}[H]
\setlength\\tabcolsep{{3pt}}
\\begin{{center}}
\\begin{{tabular}}{{ |c|c|c|c| }}
    \\hline
    Dataset & \\ac{{auroc}} & $\\Delta$\\ac{{auroc}} \\ac{{vim}} & P-value \\ac{{vim}} \\\\
    \\hline
    \\hline
    \\rowcolor{{near!50}}
    \\multicolumn{{4}}{{|c|}}{{Near-OOD}} \\\\
    \\hline""")

    rowcolor = 'near'

    for i in range(len(names)):
        better = better_metric[:, i, 1]
        vim = vim_metric[:, i, 1]

        pval = ttest_ind(better, vim, alternative='greater').pvalue

        padding = 10 - len(names[i])

        if pval > 0.05:
            pval = f'{pval:.3f}'
        elif pval > 0.01:
            pval = f'{pval:.3f} *'
        elif pval > 0.001:
            pval = f'{pval:.3f} **'
        else:
            pval = '<0.001 ***'

        diff = better.mean() - vim.mean()
        print(f'    \\rowcolor{{{rowcolor}!50}}')
        if names[i] == 'nearood':
            print(
                f'    Near-OOD Avg. & {better.mean():.2f} & {"+" if diff > 0 else ""}{diff:.3f} & {pval} \\\\'
            )
        elif names[i] == 'farood':
            print(
                f'    Far-OOD Avg. & {better.mean():.2f} & {"+" if diff > 0 else ""}{diff:.3f} & {pval} \\\\'
            )
        else:
            print(
                f'    {pp(names[i])} & {better.mean():.2f} & {"+" if diff > 0 else ""}{diff:.3f} & {pval} \\\\'
            )
        print('    \\hline')
        if names[i] == 'nearood':
            print('    \\hline')
            print('    \\rowcolor{far!50}')
            print('    \\multicolumn{4}{|c|}{Far-OOD} \\\\')
            print('    \\hline')
            rowcolor = 'far'

    print(f"""    \end{{tabular}}
    \caption[T-test for {pp(args.generator)}{args.aggregator.capitalize()} {args.better_postprocessor} on {pp(args.dataset)}]{{Results of performing a t-test on the \\ac{{auroc}} means of {pp(args.generator)}{args.aggregator.capitalize()} {args.better_postprocessor} against \\ac{{mls}} and \\ac{{msp}}, showing the mean \\ac{{auroc}} over 10 runs on {pp(args.dataset)}, the difference in means compared to the baselines, and the corresponding p-values. Each p-value is appended a significance code which follows the \\texttt{{R}}-standard.}}
    \label{{table:{args.dataset}_{args.generator}_{args.aggregator}_{args.better_postprocessor}_ttest}}
\end{{center}}
\setlength\\tabcolsep{{6pt}}
\end{{table}}
""")
