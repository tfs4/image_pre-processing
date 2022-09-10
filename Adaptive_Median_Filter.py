import numpy as np
from numba import njit, prange
import pandas as pd
import cv2
from PIL import Image, ImageFilter


def rgb2gray(rgb):
    if(len(rgb.shape) == 3):
        return np.uint8(np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140]))
    else:#already a grayscale
        return rgb
@njit
def padding(img,pad):
    padded_img = np.zeros((img.shape[0]+2*pad,img.shape[1]+2*pad))
    padded_img[pad:-pad,pad:-pad] = img
    return padded_img

@njit(parallel=True)
def AdaptiveMedianFilter(img,s=3,sMax=50):
    if len(img.shape) == 3:
        raise Exception ("Single channel image only")

    H,W = img.shape
    a = sMax//2
    padded_img = padding(img,a)

    f_img = np.zeros(padded_img.shape)

    for i in prange(a,H+a+1):
        for j in range(a,W+a+1):
            value = Lvl_A(padded_img,i,j,s,sMax)
            f_img[i,j] = value

    return f_img[a:-a,a:-a]

@njit
def Lvl_A(mat,x,y,s,sMax):
    window = mat[x-(s//2):x+(s//2)+1,y-(s//2):y+(s//2)+1]
    Zmin = np.min(window)
    Zmed = np.median(window)
    Zmax = np.max(window)

    A1 = Zmed - Zmin
    A2 = Zmed - Zmax

    if A1 > 0 and A2 < 0:
        return Lvl_B(window, Zmin, Zmed, Zmax)
    else:
        s += 2
        if s <= sMax:
            return Lvl_A(mat,x,y,s,sMax)
        else:
             return Zmed

@njit
def Lvl_B(window, Zmin, Zmed, Zmax):
    h,w = window.shape

    Zxy = window[h//2,w//2]
    B1 = Zxy - Zmin
    B2 = Zxy - Zmax

    if B1 > 0 and B2 < 0 :
        return Zxy
    else:
        return Zmed

@njit
def crop_image1(img, tol=7):
    # img is image data
    # tol  is tolerance

    mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]

def circle_crop(img, sigmaX=10):
    #img = cv2.imread(p)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # clahe = cv2.createCLAHE(clipLimit = 5)
    # img = clahe.apply(img)+30


    height, width  = img.shape

    x = int(width / 2)
    y = int(height / 2)
    r = np.amin((x, y))

    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    if img.ndim == 2:
        mask = img > 7
        img = img[np.ix_(mask.any(1), mask.any(0))]

    return img


def run():
    df_messidor = pd.read_csv('pre_processing_4/messidor-2/data.csv')
    df_idrid_test = pd.read_csv('pre_processing_4/idrid_datast/test.csv')
    df_idrid_train = pd.read_csv('pre_processing_4/idrid_datast/train.csv')
    df_aptos = pd.read_csv('pre_processing_4/APTOS/train.csv')

    for idx, row in df_aptos.iterrows():
        path = f"pre_processing_4/APTOS/500/{row['id_code']}"

        print(path)
        image_org = Image.open(path)
        image = np.array(image_org)
        image = rgb2gray(image)
        image = circle_crop(image)
        image = AdaptiveMedianFilter(image)

        cv2.imwrite(f"pre_processing_4_to_0/APTOS/{row['id_code']}", image)

    for idx, row in df_idrid_train.iterrows():
        path = f"pre_processing_4/idrid_datast/500/train/{row['id_code']}"

        print(path)
        image_org = Image.open(path)
        image = np.array(image_org)
        image = rgb2gray(image)
        image = circle_crop(image)
        image = AdaptiveMedianFilter(image)

        cv2.imwrite(f"pre_processing_4_to_0/IDRID/train/{row['id_code']}", image)

    for idx, row in df_idrid_test.iterrows():
        path = f"pre_processing_4/idrid_datast/500/test/{row['id_code']}"

        print(path)
        image_org = Image.open(path)
        image = np.array(image_org)
        image = rgb2gray(image)
        image = circle_crop(image)
        image = AdaptiveMedianFilter(image)

        cv2.imwrite(f"pre_processing_4_to_0/IDRID/test/{row['id_code']}", image)

    for idx, row in df_messidor.iterrows():
        path = f"pre_processing_4/messidor-2/500/{row['id_code']}"

        print(path)
        image_org = Image.open(path)
        image = np.array(image_org)
        image = rgb2gray(image)
        image = circle_crop(image)
        image = AdaptiveMedianFilter(image)

        cv2.imwrite(f"pre_processing_4_to_0/messidor_2/{row['id_code']}", image)


















