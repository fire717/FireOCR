import os
import time
import random
from random import choice, randint, randrange

from PIL import Image, ImageDraw, ImageFont


def getFileNames(file_dir, tail_list=['.png','.jpg','.JPG','.PNG']): 
        L=[] 
        for root, dirs, files in os.walk(file_dir):
            for file in files:
                if os.path.splitext(file)[1] in tail_list:
                    L.append(os.path.join(root, file))
        return L






class ImgGenerator():
    def __init__(self, 
                save_dir='dataset',
                save_img = 'train_gen',
                img_size = (840,32),
                max_char = 20,
                dicts = "../task2_dict.txt",
                font_dir='./fonts/', 
                sentence_file='sentences.txt', 
                sentence_gen_count = 500,
                N = '壹贰叁肆伍陆柒捌玖',
                min_count=10000, 
                bg_color_list=[(162, 198, 182), (250, 250, 250), (251, 253, 241)]):
        """
        save_dir: 保存文件夹
        img_size: 生成图片尺寸
        max_char： 单图最大字符数
        dicts: 词表路径
        font_dir: 字体路径即对应权重，合计为1
        sentence_file: 要生成的句子模板
        sentence_gen_count: 每个句子生成多少
        N: 填充sentences N的词表
        min_count: 每类最少的数字
        bg_color_list: 背景颜色
        """
        self.save_dir = save_dir
        self.img_size = img_size
        self.max_char = max_char
        self.img_dir = os.path.join(self.save_dir, save_img)


        self.dicts = dicts 
        self.alphabet = []

        self.font_dir = font_dir
        self.fonts = []

        self.sentence_file = sentence_file
        self.sentence_gen_count = sentence_gen_count
        self.sentences = []
        self.N = N

        self.min_count = min_count
        
        self.bg_color_list = bg_color_list

        self.loadData()


    def loadData(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)


        with open(self.dicts, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.alphabet = [x.strip() for x in lines]
        print("Load dict: ", len(self.alphabet), self.alphabet)

        self.fonts = getFileNames(self.font_dir, ['.ttf','.ttc'])
        print("Load fonts: ", len(self.fonts))

        with open(self.sentence_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.sentences = [x.strip() for x in lines]
        print("Load sentences: ", len(self.sentences))


    def readLabel(self, path):
        char_dict = {}
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            items = line.strip().split(" ")[2:]
            for item in items:
                c = self.alphabet[int(item)]
                if c not in char_dict:
                    char_dict[c] = 1
                else:
                    char_dict[c] += 1

        return char_dict

    def getCharColor(self):
        r = randint(0, 50)
        g = randint(0, 50)
        b = randint(0, 50)
        return (r, g, b)


    # def selectedCharacters(self, length):
    #     result = ''.join(choice(self.alphabet) for _ in range(length))
    #     return result


    def createLabel(self, label_path):

        imgs = os.listdir(self.img_dir)
        with open(label_path, 'w', encoding='utf-8') as f:
            for img_name in imgs:
                line = img_name+" "+str(self.img_size[0])+" "+" ".join(img_name[:-4].strip().split('_')[1:])+'\n'
                f.write(line)




    def getCharBySentence(self):
        # 获取要生成的文本
        text = random.choice(self.sentences)

        new_text = []
        for c in text:
            if c=="N":
                new_text.append(random.choice(self.N))
            else:
                new_text.append(c)

        
        return ''.join(new_text)

    def getCharRandom(self):
        # 获取要生成的文本
        #根据已有词表和生成的sentence词表补充

        text = []
        while len(text)<self.max_char:

            text.append(random.choice(self.alphabet[1:]))

        return ''.join(text)


    def genImg(self, text, img_name):
        # 根据文本生成图片
        # 创建空白图像和绘图对象

        img_size = self.img_size

        # 生成并计算随机字符串的宽度和高度
        imageTemp = Image.new('RGB', img_size, random.choice(self.bg_color_list))
        draw01 = ImageDraw.Draw(imageTemp)

        font_size = random.randint(self.img_size[1]-8, self.img_size[1]+4)
        font = ImageFont.truetype(random.choice(self.fonts), font_size)
        width, height = draw01.textsize(text, font)
        # if width + 2 * character_num > self.img_size[0] or height > self.img_size[1]:
        #     print('尺寸不合法')
        #     return

        

        character_num = len(text)
        if width > self.img_size[0]:
            #字长大于图片宽度，要么随机截取前后，要么先用大图再resize
            if random.random()<0.6:
                crop_len = len(text)//2+1
                if random.random()<0.5:
                    text = text[:crop_len]
                else:
                    text = text[-crop_len:]
                character_num = len(text)
            else:
                img_size = (font_size*len(text)+1,img_size[1])
                imageTemp = Image.new('RGB', img_size, random.choice(self.bg_color_list))
                draw01 = ImageDraw.Draw(imageTemp)
                # width, height = draw01.textsize(text, font)

        

       

        # 绘制随机字符串中的字符
        startX = 0
        widthEachCharater = font_size+random.randint(-1,3)
        for i in range(character_num):
            startX += widthEachCharater + 1
            position = (startX, (img_size[1] - height) // 2)
            draw01.text(xy=position, text=text[i], font=font, fill=self.getCharColor())


        if random.random()<0.5:
            # 对像素位置进行微调，实现扭曲的效果
            imageFinal = Image.new('RGB', img_size,  random.choice(self.bg_color_list))
            pixelsFinal = imageFinal.load()
            pixelsTemp = imageTemp.load()
            for y in range(img_size[1]):
                offset = randint(-1, 0)
                for x in range(img_size[0]):
                    newx = x + offset
                    if newx >= img_size[0]:
                        newx = img_size[0] - 1
                    elif newx < 0:
                        newx = 0
                    pixelsFinal[newx, y] = pixelsTemp[x, y]
        else:
            imageFinal = imageTemp

        if img_size[0]!=840:
            print(img_size,self.img_size)
            b
        if img_size[0]>self.img_size[0]:
            imageFinal = imageFinal.resize(self.img_size)

        if random.random()<0.2:
            # 绘制随机颜色随机位置的干扰像素
            draw02 = ImageDraw.Draw(imageFinal)
            for i in range(int(self.img_size[0] * self.img_size[1] * 0.07)):
                draw02.point((randrange(0, self.img_size[0]), randrange(0, self.img_size[1])), fill=self.getCharColor())

        # 保存并显示图片
        imageFinal.save(os.path.join(self.img_dir,img_name))


    def run(self, label_path):

        ### 1.按sentence生成
        gen_char_dict = {}
        for i in range(self.sentence_gen_count):
            chars = self.getCharBySentence()
            #print(chars)
            img_name = "sent-"+str(i)+"_"+'_'.join([str(self.alphabet.index(c)) for c in chars])+'.jpg'
            #print(img_name)
            self.genImg(chars, img_name)

            for char in chars:
                if char not in gen_char_dict:
                    gen_char_dict[char] = 1
                else:
                    gen_char_dict[char] += 1
        print(gen_char_dict)


        ### 2.补充生成
        for i in range(self.min_count):
            chars = self.getCharRandom()
            img_name = "rand-"+str(i)+"_"+'_'.join([str(self.alphabet.index(c)) for c in chars])+'.jpg'
            self.genImg(chars, img_name)


        ### 3.生成label文件
        self.createLabel(label_path)




if __name__ == '__main__':

    save_dir='dataset'


    img_gen1 = ImgGenerator(save_dir = save_dir,
                save_img = 'train_gen',
                sentence_gen_count = 10000,
                dicts = "../task2_dict.txt",
                min_count=500000)
    label_path = os.path.join(save_dir, 'gen_train.txt')
    img_gen1.run(label_path)





    img_gen2 = ImgGenerator(save_dir = save_dir,
                save_img = 'val_gen',
                sentence_gen_count = 50,
                dicts = "../task2_dict.txt",
                min_count=450)
    label_path = os.path.join(save_dir, 'gen_val.txt')
    img_gen2.run(label_path)


 