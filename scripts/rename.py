import os


read_dir = 'train/img'
names = os.listdir(read_dir)
for i in range(len(names)):
    read_name = os.path.join(read_dir, names[i])
    new_name = os.path.join(read_dir, str(i)+'.jpg')
    os.rename(read_name, new_name)

read_dir = 'train/label'
names = os.listdir(read_dir)
for i in range(len(names)):
    read_name = os.path.join(read_dir, names[i])
    new_name = os.path.join(read_dir, str(i)+'.txt')
    os.rename(read_name, new_name)