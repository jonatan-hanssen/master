import os
import random

root = os.path.join(os.path.dirname(__file__), '..')
data = os.path.join(root, 'data/images_largescale')
files_dir = os.path.join(root, 'data/benchmark_imglist/hyperkvasir_polyp')

hyperkvasir = [
    'hyperkvasir/lower-gi-tract/pathological-findings/polyps',
    'kvasir-sessile',
    'hyperkvasir/lower-gi-tract/anatomical-landmarks/retroflex-rectum',
    'hyperkvasir/upper-gi-tract/anatomical-landmarks/pylorus',
]

lower = [
    'hyperkvasir/lower-gi-tract/quality-of-mucosal-views/bbps-2-3',
    'hyperkvasir/lower-gi-tract/anatomical-landmarks/cecum',
    'hyperkvasir/lower-gi-tract/therapeutic-interventions/dyed-lifted-polyps',
    'hyperkvasir/lower-gi-tract/pathological-findings/ulcerative-colitis-grade-2',
    'hyperkvasir/lower-gi-tract/quality-of-mucosal-views/bbps-0-1',
    'hyperkvasir/lower-gi-tract/therapeutic-interventions/dyed-resection-margins',
    'hyperkvasir/upper-gi-tract/pathological-findings/esophagitis-a',
    'hyperkvasir/upper-gi-tract/pathological-findings/esophagitis-b-d',
]

upper = [
    'hyperkvasir/upper-gi-tract/anatomical-landmarks/retroflex-stomach',
]


all_files = list()

for i, dirname in enumerate(hyperkvasir):
    path = os.path.join(data, dirname)
    files = os.listdir(path)
    files = [f'{os.path.join(dirname, file)} {i}' for file in files]
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
    files = [f'{os.path.join(dirname, file)} -1' for file in files]
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
    files = [f'{os.path.join(dirname, file)} -1' for file in files]
    all_files = all_files + files

random.shuffle(all_files)


with open(os.path.join(files_dir, 'test_upper.txt'), 'w') as file:
    file.write('\n'.join(all_files))
