import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import glob
from tkinter import filedialog, Tk

save_dir = '/home/nizar/shadow_removal/experiments/7x7/rgb'


Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
path_selected = filedialog.askdirectory()  # show an "Open" dialog box and return the path to the selected file
print(path_selected)
os.chdir(path_selected)

def gsr (img1):
    # --------------- Step 1: mmClose -----------------#
    kernel = np.ones((2, 2), np.uint8)
    closing = cv2.morphologyEx(img1, cv2.MORPH_CLOSE, kernel)
    # plt.figure(2)
    # plt.imshow(closing)

    # --------------- Step 2: smooth -----------------#
    blur = cv2.GaussianBlur(closing, (31, 31), 0)
    # plt.figure(3)
    # plt.imshow(blur)

    # --------------- Step 3: geolevel -----------------#
    N = 255  # the initial intensity number

    ng = blur.size / N

    i = 1
    sum = 0
    geolevel = blur
    geol = []
    for k in range(0, 256):
        Pk = np.count_nonzero(blur == k)  # number of pixels with k intensity
        sum = sum + Pk
        geolevel[geolevel == k] = i
        if sum > ng:
            i += 1
            sum = 0
            geol.append(k)
    N = i
    plt.figure(4)
    plt.imshow(geolevel)
    geol = np.append(0, geol)

    # --------------- Step 4: illumcomponsate -----------------#
    # blur = cv2.GaussianBlur(closing,(61,61), 10)
    L = 0.99 * N  # number of geodesic levels
    L = int(L)
    img1_reshape = img1.reshape(img1.shape[0] * img1.shape[1]).tolist()
    img_B = [elem for elem in img1_reshape if elem in range(geol[L], 256)]
    img_B = np.asarray(img_B)
    std_B = np.std(img_B)
    mean_B = np.mean(img_B)
    std_S = []
    mean_S = []
    for i in range(0, L):
        img_S = [elem for elem in img1_reshape if elem in range(geol[i], geol[i + 1])]
        print('done')
        img_S = np.asarray(img_S)
        std_S.append(np.std(img_S))
        mean_S.append(np.mean(img_S))
        print(i)

    std_S = np.where(std_S == 0, 0.00001, std_S)
    # alpha = [std_B / x for x in std_S]
    lmd = [mean_B - 1 * mean_S[i] for i in range(0, len(mean_S))]

    # img1 = cv2.imread('99.png', 0)
    img_processed = img1
    # for i in range (0,len(mean_S)):

    for i in range(0, L):
        h, w = np.where(geolevel == i)
        for k in range(0, len(h)):
            img_processed[h[k]][w[k]] = int(1 * img1[h[k]][w[k]] + lmd[i])
        print(i)

    return img_processed

for filename in glob.glob("*.png"):
    print(filename)
    img = cv2.imread(filename)
    r, g, b = cv2.split(img)
    img_r = gsr(r)
    img_g = gsr(g)
    img_b = gsr(b)
    img_rgb = cv2.merge((img_r, img_g, img_b))

    # img1 = cv2.imread(filename, 0)

    #plt.figure(1)
    # plt.imshow(img1)

    # #--------------- Step 1: mmClose -----------------#
    # kernel = np.ones((2,2),np.uint8)
    # closing = cv2.morphologyEx(img1, cv2.MORPH_CLOSE, kernel)
    # # plt.figure(2)
    # # plt.imshow(closing)
    #
    # #--------------- Step 2: smooth -----------------#
    # blur = cv2.GaussianBlur(closing, (7,7), 0)
    # # plt.figure(3)
    # # plt.imshow(blur)
    #
    # #--------------- Step 3: geolevel -----------------#
    # N = 255# the initial intensity number
    #
    # ng = blur.size / N
    #
    # i = 1
    # sum = 0
    # geolevel = blur
    # geol = []
    # for k in range (0,256):
    #     Pk = np.count_nonzero(blur == k)  # number of pixels with k intensity
    #     sum = sum + Pk
    #     geolevel[geolevel == k] = i
    #     if sum > ng:
    #         i += 1
    #         sum = 0
    #         geol.append(k)
    # N = i
    # plt.figure(4)
    # plt.imshow(geolevel)
    # geol = np.append(0, geol)
    #
    # #--------------- Step 4: illumcomponsate -----------------#
    # # blur = cv2.GaussianBlur(closing,(61,61), 10)
    # L = 0.99 * N  # number of geodesic levels
    # L = int(L)
    # img1_reshape = img1.reshape(img1.shape[0]*img1.shape[1]).tolist()
    # img_B = [ elem for elem in img1_reshape if elem in range(geol[L], 256)]
    # img_B = np.asarray(img_B)
    # std_B = np.std(img_B)
    # mean_B = np.mean(img_B)
    # std_S = []
    # mean_S = []
    # for i in range(0, L):
    #     img_S = [elem for elem in img1_reshape if elem in range(geol[i], geol[i+1])]
    #     print ('done')
    #     img_S = np.asarray(img_S)
    #     std_S.append(np.std(img_S))
    #     mean_S.append(np.mean(img_S))
    #     print (i)
    #
    # std_S = np.where(std_S==0, 0.00001, std_S)
    # # alpha = [std_B / x for x in std_S]
    # lmd = [mean_B - 1 * mean_S[i] for i in range (0, len(mean_S))]
    #
    # # img1 = cv2.imread('99.png', 0)
    # img_processed = img1
    # # for i in range (0,len(mean_S)):
    #
    # for i in range(0, L):
    #     h, w = np.where(geolevel == i)
    #     for k in range (0,len(h)):
    #         img_processed[h[k]][w[k]] = int(1 * img1[h[k]][w[k]] + lmd[i])
    #     print(i)
    plt.figure(5)
    plt.imshow(img_rgb)
    filename_processed = filename[:-4] + '.png'
    #cv2.imwrite(filename_processed, img_processed)
    # img_processed = cv2.cvtColor(img_processed, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(os.path.join(save_dir, filename_processed), img_rgb)


    # mean = 0
    # B = np.zeros((N-L)*len(np.where(geolevel == i)[1])+1, np.uint8)
    # for i in range(L, N):
    #     index_B = np.where(geolevel == i)
    #     for k in range(0, len(np.where(geolevel == i)[1])):
    #         # mean += img1[np.where(geolevel == i)[0][k]][np.where(geolevel == i)[1][k]]
    #         B[(i-L)*len(np.where(geolevel == i)[1])+k] = img1[np.where(geolevel == i)[0][k]][np.where(geolevel == i)[1][k]]
    #         print(k)
    # B = np.asarray(B, dtype=np.float32)
    # B_mean = np.mean(B)
    # b_std = np.std(B)