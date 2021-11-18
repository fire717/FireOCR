
import random


def countLines(lines):
    word_list = {}
    for line in lines:
        items = line.strip().split(" ")
        words = items[2:]
        for word in words:
            if word not in word_list:
                word_list[word] = 0
            else:
                word_list[word] += 1
    return word_list

def main(read_path, save_path, min_count):
    #不足min_count的复制到min_count
    with open(read_path, 'r') as f:
        lines = f.readlines()


    d = countLines(lines)
    print("before: ",d)

    new_lines = []
    for line in lines:
        for k,v in d.items():
            if v < min_count and k in line:
                mul_ratio = min_count//v//6
                for _ in range(mul_ratio):
                    new_lines.append(line)
            else:
                new_lines.append(line)



    d = countLines(new_lines)
    print("after: ",d)
    random.shuffle(new_lines)
    with open(save_path, 'w') as f:
        for line in new_lines:
            f.write(line)


if __name__ == '__main__':
    read_path= "train.txt"
    save_path = "train_balanced.txt"

    main(read_path, save_path,min_count=300)

    read_path= "val.txt"
    save_path = "val_balanced.txt"
    main(read_path, save_path,min_count=5)
