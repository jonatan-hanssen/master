#!/bin/sh

for dataset in cifar10 imagenet200; do
    for generator in gradcam lime; do
        echo "python3 visualize_saliencies.py -d ${dataset} -g ${generator} --no-auc --linewidth 2 --pgf"
        python3 visualize_saliencies.py -d ${dataset} -g ${generator} --no-auc --linewidth 2 --pgf
    done
done
