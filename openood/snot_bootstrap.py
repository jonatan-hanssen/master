import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import sys, argparse, pickle

from scipy.stats import wilcoxon as ttest_ind

# from scipy.stats import ttest_ind
import pandas as pd
from utils import get_palette, pp

parser = argparse.ArgumentParser()

pd.set_option('display.precision', 3)

parser.add_argument('--dataset', '-d', type=str, default='cifar10')
parser.add_argument('--pgf', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--postprocessor', '-p', type=str, default='salpluslogit')
parser.add_argument('--fpr', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--ylim', '-y', type=float, default=0.5)


args = parser.parse_args(sys.argv[1:])

if args.pgf:
    matplotlib.use('pgf')
    matplotlib.rcParams.update(
        {
            'pgf.texsystem': 'pdflatex',
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
        }
    )

filename = f'saved_metrics/{args.dataset}_mls_bootstrapped.pkl'
with open(filename, 'rb') as file:
    data = pickle.load(file)
    dataframe = data[0][0]


def get_metrics(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)

    all_metrics, all_scores = data
    metrics = np.stack([metric.to_numpy() for metric in all_metrics])
    return metrics


if args.postprocessor == 'salvim':
    lime_metric = get_metrics(
        f'saved_metrics/{args.dataset}_{args.postprocessor}_lime_bootstrapped.pkl'
    )

    occlusion_metric = get_metrics(
        f'saved_metrics/{args.dataset}_{args.postprocessor}_occlusion_bootstrapped.pkl'
    )

    gradcam_metric = get_metrics(
        f'saved_metrics/{args.dataset}_{args.postprocessor}_gradcam_bootstrapped.pkl'
    )

    vim_metric = get_metrics(f'saved_metrics/{args.dataset}_vim_bootstrapped.pkl')

    labels = ['VIM', 'LIMEVIM', 'OcclusionVIM', 'GradCAMVIM']
    metrics = [vim_metric, lime_metric, occlusion_metric, gradcam_metric]
    plt.axvspan(-0.5, 0.5, color='gray', alpha=0.2, label='Baseline')
    plt.ylim(bottom=args.ylim, top=1)

else:
    # lime_metric = get_metrics(
    #     f'saved_metrics/{args.dataset}_{args.postprocessor}_lime_norm_bootstrapped.pkl'
    # )

    gradcam_metric = get_metrics(
        f'saved_metrics/{args.dataset}_{args.postprocessor}_gradcam_norm_bootstrapped.pkl'
    )

    gbp_metric = get_metrics(
        f'saved_metrics/{args.dataset}_{args.postprocessor}_gbp_norm_bootstrapped.pkl'
    )

    ig_metric = get_metrics(
        f'saved_metrics/{args.dataset}_{args.postprocessor}_integratedgradients_mean_bootstrapped.pkl'
    )

    grad_metric = get_metrics(
        f'saved_metrics/{args.dataset}_{args.postprocessor}_grad_norm_bootstrapped.pkl'
    )

    mls_metric = get_metrics(f'saved_metrics/{args.dataset}_mls_bootstrapped.pkl')
    msp_metric = get_metrics(f'saved_metrics/{args.dataset}_msp_bootstrapped.pkl')

    labels = [
        'GradCAM-\nNorm',
        'IG-\nMean',
        'GBP-\nNorm',
        'Gradient-\nNorm',
    ]
    metrics = [
        gradcam_metric,
        ig_metric,
        gbp_metric,
        grad_metric,
    ]
    plt.ylim(bottom=args.ylim, top=1)

nears = list()
fars = list()
near_confs = list()
far_confs = list()

for metric in metrics:
    dataframe.values[:] = metric.mean(axis=0)
    near_auroc, far_auroc = dataframe.loc[['nearood', 'farood'], 'AUROC'].values / 100

    dataframe.values[:] = metric.std(axis=0)
    near_std, far_std = dataframe.loc[['nearood', 'farood'], 'AUROC'].values / 100

    nears.append(near_auroc)
    fars.append(far_auroc)

    near_confs.append(near_std * (2.28 / np.sqrt(10)))
    far_confs.append(far_std * (2.28 / np.sqrt(10)))


offset = 0.2
x_ticks = np.arange(len(labels))
plt.rc('axes', axisbelow=True)
plt.bar(x_ticks - offset, nears, 0.4, label='Near-OOD', color=get_palette()[1])
plt.bar(x_ticks + offset, fars, 0.4, label='Far-OOD', color=get_palette()[2])
# plt.scatter(x_ticks, avg, label='Average', color=get_palette()[3])
plt.grid(axis='y', linestyle='--')

for i in range(len(labels)):
    plt.errorbar(
        i - offset,
        nears[i],
        yerr=near_confs[i],
        fmt='none',
        ecolor='black',
        capsize=7,
    )

    plt.errorbar(
        i + offset,
        fars[i],
        yerr=far_confs[i],
        fmt='none',
        ecolor='black',
        capsize=7,
    )

plt.xticks(x_ticks, labels)
# plt.xlabel('OOD detector')
plt.ylabel('AUROC')

metric_index = 2 if args.fpr else 1


# handles, labels = plt.gca().get_legend_handles_labels()
# order = [1, 2, 0]
# plt.legend([handles[idx] for idx in order], [labels[idx] for idx in order])
plt.legend()

if args.pgf:
    plt.savefig(
        f'../master/figure/{args.dataset}_{args.postprocessor}_bootstrap_barplot.pgf'
    )
    plt.clf()
    print(
        f'scp uio:master/master/figure/{args.dataset}_{args.postprocessor}_bootstrap_barplot.pgf figure/'
    )

else:
    plt.show()

if args.postprocessor != 'salvim':
    names = [
        'LIMENorm',
        'GradCAMNorm',
        'IGMean',
        'GBPNorm',
    ]

    print()

    print(f"""
\\begin{{table}}[H]
\setlength\\tabcolsep{{3pt}}
\\begin{{center}}
\\begin{{tabular}}{{ |c|c!{{\\vrule width 1pt}}c|c!{{\\vrule width 1pt}}c|c| }}
    \\hline
    Dataset & \\ac{{auroc}} & $\\Delta$\\ac{{auroc}} \\ac{{mls}} & P-value \\ac{{mls}} & $\\Delta$\\ac{{auroc}} \\ac{{msp}} & P-value \\ac{{msp}} \\\\ """)

    for metric, name in zip(metrics[2:], names):
        m = len(names)

        near_index = dataframe.index.get_loc('nearood')
        far_index = dataframe.index.get_loc('farood')

        mls_near = mls_metric[:, near_index, metric_index]
        mls_far = mls_metric[:, far_index, metric_index]

        msp_near = msp_metric[:, near_index, metric_index]
        msp_far = msp_metric[:, far_index, metric_index]

        metric_near = metric[:, near_index, metric_index]
        metric_far = metric[:, far_index, metric_index]

        if args.fpr:
            pval_near_mls = ttest_ind(metric_near, mls_near, alternative='less').pvalue
            pval_far_mls = ttest_ind(metric_far, mls_far, alternative='less').pvalue

            pval_near_msp = ttest_ind(metric_near, msp_near, alternative='less').pvalue
            pval_far_msp = ttest_ind(metric_far, msp_far, alternative='less').pvalue
        else:
            pval_near_mls = ttest_ind(
                metric_near, mls_near, alternative='greater'
            ).pvalue
            pval_far_mls = ttest_ind(metric_far, mls_far, alternative='greater').pvalue

            pval_near_msp = ttest_ind(
                metric_near, msp_near, alternative='greater'
            ).pvalue
            pval_far_msp = ttest_ind(metric_far, msp_far, alternative='greater').pvalue

        pval_dict = {
            'near_mls': pval_near_mls,
            'far_mls': pval_far_mls,
            'near_msp': pval_near_msp,
            'far_msp': pval_far_msp,
        }

        for key in pval_dict:
            pval = pval_dict[key]
            if pval > 0.05 / m:
                pval = f'{pval:.3f}'
            elif pval > 0.01 / m:
                pval = f'{pval:.3f} *'
            elif pval > 0.001 / m:
                pval = f'{pval:.3f} **'
            else:
                pval = f'{pval:.1e} ***'
            pval_dict[key] = pval

        padding = 10 - len(name)

        mean_diff_near_mls = metric_near.mean() - mls_near.mean()
        mean_diff_far_mls = metric_far.mean() - mls_far.mean()

        mean_diff_near_msp = metric_near.mean() - msp_near.mean()
        mean_diff_far_msp = metric_far.mean() - msp_far.mean()

        print(f"""    \\hline
    \\hline
    \\multicolumn{{6}}{{|c|}}{{{name}}} \\\\
    \\hline""")

        print('    \\rowcolor{near!50}')
        print(
            f'    Near-OOD & {metric_near.mean():.2f} & {"+" if mean_diff_near_mls > 0 else ""}{mean_diff_near_mls:.3f} & {pval_dict["near_mls"]} & {"+" if mean_diff_near_msp > 0 else""}{mean_diff_near_msp:.3f} & {pval_dict["near_msp"]} \\\\'
        )

        print('    \\rowcolor{far!50}')
        print(
            f'    Far-OOD & {metric_far.mean():.2f} & {"+" if mean_diff_far_mls > 0 else ""}{mean_diff_far_mls:.3f} & {pval_dict["far_mls"]} & {"+" if mean_diff_far_msp > 0 else""}{mean_diff_far_msp:.3f} & {pval_dict["far_msp"]} \\\\'
        )

    print(f"""    \\hline
    \end{{tabular}}
    \caption[]{{Results of performing a t-test on the \\ac{{auroc}} means of against \\ac{{mls}} and \\ac{{msp}}, showing the mean \\ac{{auroc}} over 10 runs on {pp(args.dataset)}, the difference in means compared to the baselines, and the corresponding p-values. Each p-value is appended a significance code which follows the \\texttt{{R}}-standard.}}
    \label{{table:{args.dataset}_{args.postprocessor}_ttest}}
\end{{center}}
\setlength\\tabcolsep{{6pt}}
\end{{table}}""")

else:
    names = [
        'LIMEVIM',
        'OcclusionVIM',
        'GradCAMVIM',
    ]

    print()

    print(f"""
\\begin{{table}}[H]
\setlength\\tabcolsep{{3pt}}
\\begin{{center}}
\\begin{{tabular}}{{ |c|c|c|c| }}
    \\hline
    Dataset & \\ac{{auroc}} & $\\Delta$\\ac{{auroc}} \\ac{{vim}} & P-value \\ac{{vim}} \\\\ """)

    for metric, name in zip(metrics[1:], names):
        m = len(names)

        near_index = dataframe.index.get_loc('nearood')
        far_index = dataframe.index.get_loc('farood')

        vim_near = vim_metric[:, near_index, metric_index]
        vim_far = vim_metric[:, far_index, metric_index]

        metric_near = metric[:, near_index, metric_index]
        metric_far = metric[:, far_index, metric_index]

        if args.fpr:
            pval_near = ttest_ind(metric_near, vim_near, alternative='less').pvalue
            pval_far = ttest_ind(metric_far, vim_far, alternative='less').pvalue
        else:
            pval_near = ttest_ind(metric_near, vim_near, alternative='greater').pvalue
            pval_far = ttest_ind(metric_far, vim_far, alternative='greater').pvalue

        pval_dict = {
            'near': pval_near,
            'far': pval_far,
        }

        for key in pval_dict:
            pval = pval_dict[key]
            if pval > 0.05 / m:
                pval = f'{pval:.3f}'
            elif pval > 0.01 / m:
                pval = f'{pval:.3f} *'
            elif pval > 0.001 / m:
                pval = f'{pval:.3f} **'
            else:
                pval = f'{pval:.1e} ***'
            pval_dict[key] = pval

        padding = 10 - len(name)

        mean_diff_near = metric_near.mean() - vim_near.mean()
        mean_diff_far = metric_far.mean() - vim_far.mean()

        print(f"""    \\hline
    \\hline
    \\multicolumn{{4}}{{|c|}}{{{name}}} \\\\
    \\hline""")

        print('    \\rowcolor{near!50}')
        print(
            f'    Near-OOD & {metric_near.mean():.2f} & {"+" if mean_diff_near > 0 else ""}{mean_diff_near:.3f} & {pval_dict["near"]} \\\\'
        )

        print('    \\rowcolor{far!50}')
        print(
            f'    Far-OOD & {metric_far.mean():.2f} & {"+" if mean_diff_far > 0 else ""}{mean_diff_far:.3f} & {pval_dict["far"]} \\\\'
        )

    print(f"""    \\hline
    \end{{tabular}}
    \caption[]{{Results of performing a t-test on the \\ac{{auroc}} means of against \\ac{{vim}}, showing the mean \\ac{{auroc}} over 10 runs on {pp(args.dataset)}, the difference in means compared to the baselines, and the corresponding p-values. Each p-value is appended a significance code which follows the \\texttt{{R}}-standard.}}
    \label{{table:{args.dataset}_{args.postprocessor}_ttest}}
\end{{center}}
\setlength\\tabcolsep{{6pt}}
\end{{table}}""")
