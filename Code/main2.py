import os
import random

import numpy as np
import cv2
import matplotlib.pyplot as plt
from Crypto.Cipher import AES
from Crypto.SelfTest.st_common import b2a_hex, a2b_hex
import math
import struct
from reedsolo import RSCodec, ReedSolomonError


class AES_RS:
    def __init__(self):
        self.encoding = 'utf-16'
        self.key = '9999999999999999'.encode('utf-8')
        self.mode = AES.MODE_ECB
        self.ecc = 24

        self.rsc = RSCodec(self.ecc)
        self.cryptos = AES.new(self.key, self.mode)

    def add_to_16(self, text):
        if len(text.encode(self.encoding)) % 16:
            add = (16 - (len(text.encode(self.encoding)) % 16)) / 2
        else:
            add = 0
        text = text + ('\0' * int(add))
        return text.encode(self.encoding)

    # 加密函数
    def encrypt(self, text):
        text = self.add_to_16(text)
        cryptos = AES.new(self.key, self.mode)
        cipher_text = cryptos.encrypt(text)
        return cipher_text

    # 解密后，去掉补足的空格用strip() 去掉
    def decrypt(self, text):
        return self.cryptos.decrypt(text).decode("utf16")

    def tobin(self, e):
        str = []
        for x in e:
            for bit in '{:08b}'.format(x):
                str.append(int(bit))
        return str

    def tobyte(self, bitslist):
        e = ''
        for bit in bitslist:
            e = e + str(int(bit))
        bytes = b''
        for i in range(0, len(e), 8):
            bytes = bytes + struct.pack('B', int(e[i:i + 8], 2))
        return bytes

    def getimage(self, path):
        with open(path, 'r', encoding='utf16') as f:
            str = f.read()
        aes_text = self.encrypt(str)
        bits = []
        for i in range(0, int(len(aes_text) / 16)):
            bits.extend(self.tobin(self.rsc.encode(aes_text[i * 16:i * 16 + 16])))
        image = []
        for t in range(0, 5):
            image.extend(bits)
        for i in range(0, len(image)):
            if image[i] == 1:
                image[i] = 255
        cv2.imwrite("../Material/watermark.png", np.array(image).reshape(80, 80))
        return image

    def getinfo(self, image):
        image[image < 128] = 0
        image[image > 128] = 1
        image = image.reshape(5, -1)
        info = np.int64(np.sum(image, axis=0) > 2.5)
        image = np.insert(image, 0, values=info, axis=0)
        for i in range(0,4):
            for row in range(0,6):
                try:
                    rsc_text = self.rsc.decode(self.tobyte(image[row,i*320:i*320+320]))[0]
                    print(self.decrypt(rsc_text))
                    break
                except:
                    pass


if __name__ == '__main__':
    ar = AES_RS()
    ar.getinfo(cv2.imread(r"C:\Users\Nevermore\PycharmProjects\HW\BlindWatermark\out_wm.png", cv2.IMREAD_GRAYSCALE))

    pass
