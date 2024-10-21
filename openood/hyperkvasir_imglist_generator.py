import os
import random

root = os.path.dirname(__file__)
data = os.path.join(root, 'data/images_classic/hyperkvasir')
files_dir = os.path.join(root, 'data/benchmark_imglist/hyperkvasir')

dirnames = [
    'lower-gi-tract/anatomical-landmarks/cecum',
    'lower-gi-tract/anatomical-landmarks/retroflex-rectum',
    'lower-gi-tract/pathological-findings/polyps',
    'lower-gi-tract/pathological-findings/ulcerative-colitis-grade-2',
    'lower-gi-tract/therapeutic-interventions/dyed-lifted-polyps',
    'lower-gi-tract/therapeutic-interventions/dyed-resection-margins',
    'lower-gi-tract/quality-of-mucosal-views/bbps-0-1',
    'lower-gi-tract/quality-of-mucosal-views/bbps-2-3',
    'upper-gi-tract/anatomical-landmarks/pylorus',
    'upper-gi-tract/anatomical-landmarks/retroflex-stomach',
    'upper-gi-tract/anatomical-landmarks/z-line',
    'upper-gi-tract/pathological-findings/esophagitis-a',
    'upper-gi-tract/pathological-findings/esophagitis-b-d',
]


all_files = list()

for i, dirname in enumerate(dirnames):
    path = os.path.join(data, dirname)
    files = os.listdir(path)
    files = [f"{os.path.join('hyperkvasir', dirname, file)} {i}" for file in files]
    all_files = all_files + files

random.shuffle(all_files)

print(len(all_files))

train = all_files[:8000]
val = all_files[8000:9000]
test = all_files[9000:]

with open(os.path.join(files_dir, 'train_hyperkvasir.txt'), 'w') as file:
    file.write('\n'.join(train))

with open(os.path.join(files_dir, 'val_hyperkvasir.txt'), 'w') as file:
    file.write('\n'.join(val))

with open(os.path.join(files_dir, 'test_hyperkvasir.txt'), 'w') as file:
    file.write('\n'.join(test))
