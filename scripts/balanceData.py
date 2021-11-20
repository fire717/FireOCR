
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
    with open(read_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()


    d = countLines(lines)
    print("before: ",d)

    new_lines = []
    for line in lines:
        expand = 1
        for k,v in d.items():
            #print(k,v, min_count)
            if v < min_count and k in line:
                mul_ratio = min_count//v
                if mul_ratio>expand:
                    expand = mul_ratio


        # print(expand)   

        for _ in range(expand):
            new_lines.append(line)

        # b


    d = countLines(new_lines)
    print("after: ",d)
    random.shuffle(new_lines)
    with open(save_path, 'w', encoding='utf-8') as f:
        for line in new_lines:
            f.write(line)


if __name__ == '__main__':
    read_path= "../data/train.txt"
    save_path = "../data/train_balanced.txt"

    main(read_path, save_path,min_count=300)

    read_path= "../data/val.txt"
    save_path = "../data/val_balanced.txt"
    main(read_path, save_path,min_count=5)
