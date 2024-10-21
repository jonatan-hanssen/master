import os
import random

root = os.path.dirname(__file__)
data = os.path.join(root, 'data/images_largescale/hyperkvasir')
files_dir = os.path.join(root, 'data/benchmark_imglist/hyperkvasir')

hyperkvasir = [
    'lower-gi-tract/anatomical-landmarks/cecum',
    'lower-gi-tract/anatomical-landmarks/retroflex-rectum',
    'lower-gi-tract/pathological-findings/polyps',
    'lower-gi-tract/pathological-findings/ulcerative-colitis-grade-2',
    'lower-gi-tract/quality-of-mucosal-views/bbps-0-1',
    'lower-gi-tract/quality-of-mucosal-views/bbps-2-3',
]

lower = [
    'lower-gi-tract/therapeutic-interventions/dyed-lifted-polyps',
    'lower-gi-tract/therapeutic-interventions/dyed-resection-margins',
]

upper = [
    'upper-gi-tract/anatomical-landmarks/pylorus',
    'upper-gi-tract/anatomical-landmarks/retroflex-stomach',
    'upper-gi-tract/anatomical-landmarks/z-line',
    'upper-gi-tract/pathological-findings/esophagitis-a',
    'upper-gi-tract/pathological-findings/esophagitis-b-d',
]


all_files = list()

for i, dirname in enumerate(hyperkvasir):
    path = os.path.join(data, dirname)
    files = os.listdir(path)
    files = [f"{os.path.join('hyperkvasir', dirname, file)} {i}" for file in files]
    all_files = all_files + files

random.shuffle(all_files)

a = int(len(all_files) * 0.8)
b = int(len(all_files) * 0.9)

train = all_files[:a]
val = all_files[a:b]
test = all_files[b:]

with open(os.path.join(files_dir, 'train_hyperkvasir.txt'), 'w') as file:
    file.write('\n'.join(train))

with open(os.path.join(files_dir, 'val_hyperkvasir.txt'), 'w') as file:
    file.write('\n'.join(val))

with open(os.path.join(files_dir, 'test_hyperkvasir.txt'), 'w') as file:
    file.write('\n'.join(test))

#############################

all_files = list()

for i, dirname in enumerate(lower):
    path = os.path.join(data, dirname)
    files = os.listdir(path)
    files = [f"{os.path.join('hyperkvasir', dirname, file)} {i}" for file in files]
    all_files = all_files + files

random.shuffle(all_files)

a = int(len(all_files) * 0.8)

test = all_files[:a]
val = all_files[a:]

with open(os.path.join(files_dir, 'test_lower.txt'), 'w') as file:
    file.write('\n'.join(test))

with open(os.path.join(files_dir, 'val_lower.txt'), 'w') as file:
    file.write('\n'.join(val))

#############################

all_files = list()

for i, dirname in enumerate(upper):
    path = os.path.join(data, dirname)
    files = os.listdir(path)
    files = [f"{os.path.join('hyperkvasir', dirname, file)} {i}" for file in files]
    all_files = all_files + files

random.shuffle(all_files)


with open(os.path.join(files_dir, 'test_upper.txt'), 'w') as file:
    file.write('\n'.join(all_files))
