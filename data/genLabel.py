

with open("train/train.txt", 'w') as f:
    for i in range(11,501):
        line = str(i)+".jpg 1 2 3 4 5 6 7 8 9 10\n"
        f.write(line)