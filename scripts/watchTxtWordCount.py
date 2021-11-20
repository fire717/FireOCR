
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

def main(read_path):
    #不足min_count的复制到min_count
    with open(read_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()


    d = countLines(lines)
    print(d)



if __name__ == '__main__':
    read_path= "../data/challange/train_bal_add_gen.txt"

    main(read_path)

    read_path= "../data/challange/val_bal_add_gen.txt"
    main(read_path)
