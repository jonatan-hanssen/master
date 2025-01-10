import os
import random

random.seed(0)

root = os.path.join(os.path.dirname(__file__), '..')
data = os.path.join(root, 'data/images_largescale/stanforddogs')
files_dir = os.path.join(root, 'data/benchmark_imglist/stanforddogs')

test = os.listdir(data)
test_list = list()

for i, dirname in enumerate(test):
    path = os.path.join(data, dirname)
    files = os.listdir(path)
    files = [f'{os.path.join("stanforddogs", dirname, file)} -1' for file in files]
    test_list = test_list + files

with open(os.path.join(files_dir, 'test_stanforddogs.txt'), 'w') as file:
    file.write('\n'.join(test_list))
