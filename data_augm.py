import os
import cv2
import glob
import numpy as np
import random
import string

def get_random_string(length):
    letters = string.ascii_lowercase + string.ascii_uppercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str

def colorjitter(img, cj_type="b"):

    if cj_type == "b":
        value = np.random.choice(np.array([-50, -40, -30, 30, 40, 50]))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if value >= 0:
            lim = 255 - value
            v[v > lim] = 255
            v[v <= lim] += value
        else:
            lim = np.absolute(value)
            v[v < lim] = 0
            v[v >= lim] -= np.absolute(value)

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

    elif cj_type == "s":
        value = np.random.choice(np.array([-50, -40, -30, 30, 40, 50]))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if value >= 0:
            lim = 255 - value
            s[s > lim] = 255
            s[s <= lim] += value
        else:
            lim = np.absolute(value)
            s[s < lim] = 0
            s[s >= lim] -= np.absolute(value)

        final_hsv = cv2.merge((h, s, v))
        img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return img

    elif cj_type == "c":
        brightness = 10
        contrast = random.randint(40, 100)
        dummy = np.int16(img)
        dummy = dummy * (contrast / 127 + 1) - contrast + brightness
        dummy = np.clip(dummy, 0, 255)
        img = np.uint8(dummy)
        return img


def noisy(img, noise_type="gauss"):

    if noise_type == "gauss":
        image = img.copy()
        mean = 0
        st = 0.7
        gauss = np.random.normal(mean, st, image.shape)
        gauss = gauss.astype('uint8')
        image = cv2.add(image, gauss)
        return image

    elif noise_type == "sp":
        image = img.copy()
        prob = 0.05
        if len(image.shape) == 2:
            black = 0
            white = 255
        else:
            colorspace = image.shape[2]
            if colorspace == 3:  # RGB
                black = np.array([0, 0, 0], dtype='uint8')
                white = np.array([255, 255, 255], dtype='uint8')
            else:  # RGBA
                black = np.array([0, 0, 0, 255], dtype='uint8')
                white = np.array([255, 255, 255, 255], dtype='uint8')
        probs = np.random.random(image.shape[:2])
        image[probs < (prob / 2)] = black
        image[probs > 1 - (prob / 2)] = white
        return image


def filters(img, f_type="blur"):

    if f_type == "blur":
        image = img.copy()
        fsize = 9
        return cv2.blur(image, (fsize, fsize))

    elif f_type == "gaussian":
        image = img.copy()
        fsize = 9
        return cv2.GaussianBlur(image, (fsize, fsize), 0)

    elif f_type == "median":
        image = img.copy()
        fsize = 9
        return cv2.medianBlur(image, fsize)

inputFolder = "data\\first"
inputFolder2 = "data\\second"
output = 'augm/first/'
output2 = 'augm/second/'
folderLen = len(inputFolder)
i = 0

for img in glob.glob(inputFolder + "/*.jpg"):

    image = cv2.imread(img)
    r1 = colorjitter(image, cj_type='b')
    r2 = colorjitter(image, cj_type='s')
    r3 = colorjitter(image, cj_type='c')
    cv2.imwrite(output + get_random_string(8) + '.jpg', r1)
    cv2.imwrite(output + get_random_string(8) + '.jpg', r2)
    cv2.imwrite(output + get_random_string(8) + '.jpg', r3)

    r1 = noisy(image, noise_type="gauss")
    r2 = noisy(image, noise_type="sp")
    cv2.imwrite(output + get_random_string(8) + '.jpg', r1)
    cv2.imwrite(output + get_random_string(8) + '.jpg', r2)

    r1 = filters(image, f_type="blur")
    r2 = filters(image, f_type="gaussian")
    r3 = filters(image, f_type="median")
    cv2.imwrite(output + get_random_string(8) + '.jpg', r1)
    cv2.imwrite(output + get_random_string(8) + '.jpg', r2)
    cv2.imwrite(output + get_random_string(8) + '.jpg', r3)


for img in glob.glob(inputFolder2 + "/*.jpg"):

    image = cv2.imread(img)
    r1 = colorjitter(image, cj_type='b')
    r2 = colorjitter(image, cj_type='s')
    r3 = colorjitter(image, cj_type='c')
    cv2.imwrite(output2 + get_random_string(8) + '.jpg', r1)
    cv2.imwrite(output2 + get_random_string(8) + '.jpg', r2)
    cv2.imwrite(output2 + get_random_string(8) + '.jpg', r3)

    r1 = noisy(image, noise_type="gauss")
    r2 = noisy(image, noise_type="sp")
    cv2.imwrite(output2 + get_random_string(8) + '.jpg', r1)
    cv2.imwrite(output2 + get_random_string(8) + '.jpg', r2)

    r1 = filters(image, f_type="blur")
    r2 = filters(image, f_type="gaussian")
    r3 = filters(image, f_type="median")
    cv2.imwrite(output2 + get_random_string(8) + '.jpg', r1)
    cv2.imwrite(output2 + get_random_string(8) + '.jpg', r2)
    cv2.imwrite(output2 + get_random_string(8) + '.jpg', r3)


# most of this scrip was made by tranleanh and his github can be found at https://github.com/tranleanh