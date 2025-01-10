import os
import random

random.seed(0)

root = os.path.join(os.path.dirname(__file__), '..')
data = os.path.join(root, 'data/images_largescale/imagewoof')
files_dir = os.path.join(root, 'data/benchmark_imglist/imagewoof')

train = [
    'train/n02086240',  # shih zu
    'train/n02087394',  # rhodesian
    'train/n02088364',  # beagle
    'train/n02089973',  # english foxhound
    'train/n02093754',  # border terrier
    'train/n02096294',  # australian terrier
    'train/n02099601',  # golden retriever
    'train/n02105641',  # old english sheepdog
    'train/n02111889',  # samoyed
    'train/n02115641',  # dingo
]

val_test = [
    'val/n02086240',  # shih zu
    'val/n02087394',  # rhodesian
    'val/n02088364',  # beagle
    'val/n02089973',  # english foxhound
    'val/n02093754',  # border terrier
    'val/n02096294',  # australian terrier
    'val/n02099601',  # golden retriever
    'val/n02105641',  # old english sheepdog
    'val/n02111889',  # samoyed
    'val/n02115641',  # dingo
]

train_list = list()

for i, dirname in enumerate(train):
    path = os.path.join(data, dirname)
    files = os.listdir(path)
    files = [f'{os.path.join("imagewoof", dirname, file)} {i}' for file in files]
    train_list = train_list + files

print(len(train_list))

with open(os.path.join(files_dir, 'train_imagewoof.txt'), 'w') as file:
    file.write('\n'.join(train_list))

val_test_list = list()

for i, dirname in enumerate(val_test):
    path = os.path.join(data, dirname)
    files = os.listdir(path)
    files = [f'{os.path.join("imagewoof", dirname, file)} {i}' for file in files]
    val_test_list = val_test_list + files

print(len(val_test_list))

a = int(len(val_test_list) * 0.5)

random.shuffle(val_test_list)

test = val_test_list[:a]
val = val_test_list[a:]

with open(os.path.join(files_dir, 'test_imagewoof.txt'), 'w') as file:
    file.write('\n'.join(test))

with open(os.path.join(files_dir, 'val_imagewoof.txt'), 'w') as file:
    file.write('\n'.join(val))
