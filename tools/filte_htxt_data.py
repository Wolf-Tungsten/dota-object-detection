import os
import shutil

label_path = './dataset/htxt_dataset/labels'
output_path = './data'
for i in os.walk(label_path):
    for label_file in i[2]:
        if os.path.getsize(os.path.join(label_path, label_file)) > 0:
            image_name = label_file.split('.')[0]
            shutil.copyfile(os.path.join('./dataset/htxt_dataset/images', image_name+'.jpg'), os.path.join(output_path, 'images', image_name+'.jpg'))
            shutil.copyfile(os.path.join('./dataset/htxt_dataset/labels', image_name+'.txt'), os.path.join(output_path, 'labels', image_name+'.txt'))
