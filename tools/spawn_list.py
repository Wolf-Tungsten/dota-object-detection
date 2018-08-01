import os
import shutil

counter = 0

with open('./data/list.txt', 'w') as f:
    for i in os.walk('./data/images'):
        for filename in i[2]:
            f.write(filename.split('.')[0]+'\n')

        