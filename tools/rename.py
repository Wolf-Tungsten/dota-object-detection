import os
import shutil

counter = 0

for i in os.walk('./data/images'):
    for filename in i[2]:
        new_filename = str('P%05d'%counter)
        print(new_filename)
        counter += 1
        shutil.move(os.path.join('./data/images', filename), os.path.join('./data/images', new_filename+'.jpg'))
        shutil.move(os.path.join('./data/labels', filename.split('.')[0]+'.txt'), os.path.join('./data/labels', new_filename+'.txt'))