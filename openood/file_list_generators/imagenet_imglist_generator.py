import os
import random

random.seed(0)

root = os.path.join(os.path.dirname(__file__), '..')
data_dir = os.path.join(root, 'benchmark_imglist/imagenet')
files_dir = os.path.join(root, 'data/benchmark_imglist/imagenet')

with open(os.path.join(data_dir, 'test_imagenet.txt'), 'r') as file:
    test = file.read()

test = test.split('\n')[:-1]

with open(os.path.join(data_dir, 'val_imagenet.txt'), 'r') as file:
    val = file.read()

val = val.split('\n')[:-1]

data = val + test
random.shuffle(data)

train = data[:30000]
test = data[30000:40000]
val = data[40000:]


with open(os.path.join(files_dir, 'train_imagenet.txt'), 'w') as file:
    file.write('\n'.join(train))

with open(os.path.join(files_dir, 'val_imagenet.txt'), 'w') as file:
    file.write('\n'.join(val))

with open(os.path.join(files_dir, 'test_imagenet.txt'), 'w') as file:
    file.write('\n'.join(test))
