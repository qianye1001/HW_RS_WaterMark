import os
import random

import numpy as np
import cv2
import matplotlib.pyplot as plt
from Crypto.Cipher import AES
from Crypto.SelfTest.st_common import b2a_hex, a2b_hex
import math
import struct

from BlindWatermark.BlindWatermark import watermark
from BlindWatermark.ncc import test_ncc
from BlindWatermark.psnr import test_psnr
from Code.main2 import AES_RS


def gaussian_noise(img, mean, sigma):
    '''
    此函数用将产生的高斯噪声加到图片上
    传入:
        img   :  原图
        mean  :  均值
        sigma :  标准差
    返回:
        gaussian_out : 噪声处理后的图片
    '''
    # 将图片灰度标准化
    img = img / 255
    # 产生高斯 noise
    noise = np.random.normal(mean, sigma, img.shape)
    # 将噪声和图片叠加
    gaussian_out = img + noise
    # 将超过 1 的置 1，低于 0 的置 0
    gaussian_out = np.clip(gaussian_out, 0, 1)
    # 将图片灰度范围的恢复为 0-255
    gaussian_out = np.uint8(gaussian_out * 255)
    # 将噪声范围搞为 0-255
    # noise = np.uint8(noise*255)
    return gaussian_out  # 这里也会返回噪声，注意返回值


def sp_noise(noise_img, proportion):
    '''
    添加椒盐噪声
    proportion的值表示加入噪声的量，可根据需要自行调整
    return: img_noise
    '''
    height, width = noise_img.shape[0], noise_img.shape[1]  # 获取高度宽度像素值
    num = int(height * width * proportion)  # 一个准备加入多少噪声小点
    for i in range(num):
        w = random.randint(0, width - 1)
        h = random.randint(0, height - 1)
        if random.randint(0, 1) == 0:
            noise_img[h, w] = 0
        else:
            noise_img[h, w] = 255
    return noise_img


def crop(image, proportion):
    img_noise = image
    rows, cols = img_noise.shape[0], img_noise.shape[1]
    for i in range(0, int(proportion * rows)):
        img_noise[i,] = 255
    return img_noise


def bright_att(input_img, ratio=0.8):
    # 亮度调整攻击，ratio应当多于0
    # ratio>1是调得更亮，ratio<1是亮度更暗
    output_img = input_img * ratio
    output_img[output_img > 255] = 255
    return output_img


def resize_att(input_filename, output_file_name, out_shape=(500, 500)):
    # 缩放攻击：因为攻击和还原都是缩放，所以攻击和还原都调用这个函数
    input_img = cv2.imread(input_filename)
    output_img = cv2.resize(input_img, dsize=out_shape)
    cv2.imwrite(output_file_name, output_img)


def rot_att(input_img, angle=45):
    # 旋转攻击
    rows, cols, _ = input_img.shape
    M = cv2.getRotationMatrix2D(center=(cols / 2, rows / 2), angle=angle, scale=1)
    output_img = cv2.warpAffine(input_img, M, (cols, rows))
    return output_img


def resize_att(input_img, out_shape=(500, 500)):
    # 缩放攻击：因为攻击和还原都是缩放，所以攻击和还原都调用这个函数
    output_img = cv2.resize(input_img, dsize=out_shape)
    return output_img


def attack(attack):
    bwm1 = watermark(4399, 2333, 32, None, [80, 80])
    bwm1.read_ori_img(r"C:\Users\Nevermore\PycharmProjects\HW\Material\lena640grey.png")
    ar = AES_RS()
    cv2.imwrite("temp.png", attack)
    bwm1.extract(r"temp.png", "temp_wm.png")
    ar.getinfo(cv2.imread(r"temp_wm.png", cv2.IMREAD_GRAYSCALE))
    test_psnr(r"C:\Users\Nevermore\PycharmProjects\HW\Material\watermark.png", r"temp_wm.png")
    test_ncc(r"C:\Users\Nevermore\PycharmProjects\HW\Material\watermark.png", r"temp_wm.png")


def jpg_att(quality):
    bwm1 = watermark(4399, 2333, 32, None, [80, 80])
    bwm1.read_ori_img(r"C:\Users\Nevermore\PycharmProjects\HW\Material\lena640grey.png")
    ar = AES_RS()
    cv2.imwrite("temp.jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    bwm1.extract(r"temp.jpg", "temp_wm.png")
    ar.getinfo(cv2.imread(r"Y_U_V/Ytemp_wm.png", cv2.IMREAD_GRAYSCALE))
    test_psnr(r"C:\Users\Nevermore\PycharmProjects\HW\Material\watermark.png", r"Y_U_V/Ytemp_wm.png")
    test_ncc(r"C:\Users\Nevermore\PycharmProjects\HW\Material\watermark.png", r"Y_U_V/Ytemp_wm.png")


if __name__ == '__main__':
    img = cv2.imread("out.png", cv2.IMREAD_COLOR)
    attack(gaussian_noise(img, 0, 0.026))
    attack(sp_noise(img, 0.44))
    attack(bright_att(img, 1.3))
    jpg_att(45)
    rate = int(640 * 1)
    attack(resize_att(resize_att(img, (rate, rate)), (640, 640)))
    attack(rot_att(rot_att(img, -45), 45))
    attack(crop(img, 0.23))
    pass
