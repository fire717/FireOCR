



def parseJson(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    word_list = {}
    for line in lines[1:-1]:
        items = line.strip().split(": ")
        words = items[-1][1:-2]
        for word in words:
            if word not in word_list:
                word_list[word] = 0
            else:
                word_list[word] += 1
    print(path,word_list)
    return word_list

def main(read_path_list, save_path):
    total_dict = {}
    for read_path in read_path_list:
        word_list = parseJson(read_path)
        for word in word_list:
            if word not in total_dict:
                total_dict[word] = word_list[word]
            else:
                total_dict[word] += word_list[word]
    print("total: ", len(total_dict))

    print(total_dict)


if __name__ == '__main__':
    read_path_list = ["train/amount/gt.json",
                    "train/date/gt.json"]
    save_path = "mydict.txt"

    main(read_path_list, save_path)
