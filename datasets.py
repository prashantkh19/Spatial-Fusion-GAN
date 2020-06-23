import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image, ImageFont, ImageDraw
import torchvision.transforms as transforms
import torch

class MyDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_X = sorted(glob.glob(os.path.join(root, 'dataset/base_icdar13/background_updated') + '/*.*'))
        # self.files_Y = sorted(glob.glob(os.path.join(root, 'dataset/foreground_text') + '/*.*'))
        self.files_Z = sorted(glob.glob(os.path.join(root, 'dataset/base_icdar13/real') + '/*.*'))

    def __getitem__(self, index):
        item_X = self.transform(Image.open(self.files_X[index % len(self.files_X)]))
        # item_Y = self.transform(Image.open(self.files_Y[index % len(self.files_Y)]))
        item_Z = self.transform(Image.open(self.files_Z[index % len(self.files_Z)]))
        
        # if self.unaligned:
        #     item_Y = self.transform(Image.open(self.files_Y[random.randint(0, len(self.files_Y) - 1)]))
        # else:
        #     item_Y = self.transform(Image.open(self.files_Y[index % len(self.files_Y)]))

        return {'X': item_X,
                # 'Y': item_Y,
                'Z': item_Z}

    def __len__(self):
        return min(len(self.files_X), len(self.files_Z))

class TextUtils:
    def __init__(self, root, transforms_):
        self.fonts = glob.glob(root + 'data/fonts' + '/*.*')
        self.WORDS = self.load_words(root)
        assert len(self.fonts) > 0 and len(self.WORDS) > 0
        self.transform = transforms.Compose(transforms_)

    def load_words(self, root):
        filename = os.path.join(root, 'data/words.txt')
        WORDS = []
        with open(filename, 'r') as f:
            for line in f:
                WORDS += line.split()
        return WORDS

    def get_random_font(self):
        font_id = random.randint(0, len(self.fonts) - 1) 
        return self.fonts[font_id]

    def get_random_word(self):
        word_id = random.randint(0, len(self.WORDS) - 1) 
        return self.WORDS[word_id]

    def get_text_mask(self, shape=(256, 256), img_fraction=0.8):
        font_name = self.get_random_font()
        word = self.get_random_word()
        fontsize = 10
        font = ImageFont.truetype(font = font_name, size=fontsize)

        breakpoint = img_fraction * shape[0]
        jumpsize = 10
        while True:
            if font.getsize(word)[0] < breakpoint:
                fontsize += jumpsize
            else:
                jumpsize = int(jumpsize / 2)
                fontsize -= jumpsize
            font = ImageFont.truetype(font = font_name, size=fontsize)
            if jumpsize <= 1:
                break

        # mask = Image.new('RGB', shape, color = (255, 255, 255))
        # mask = Image.new('1', shape, color = (255, 255, 255))
        mask = Image.new('RGB', shape)
        draw = ImageDraw.Draw(mask)
        y = (shape[0] - font.getsize(word)[0])/2
        x = (shape[1] - font.getsize(word)[1])/2
        draw.text((y, x), word, font=font, fill='#4DFED1')
        # draw.text((y, x), word, font=font, fill='#ffffff')
        return self.transform(mask)

    def get_text_masks(self, bs, shape=(256, 256), img_fraction=0.8):
        masks = []
        for i in range(bs):
            masks.append(self.get_text_mask(shape, img_fraction)[None])
        return torch.cat(masks, dim=0)
