import numpy as np
import cv2
from matplotlib import pyplot as plt

class Histogram:
    def show_img_plt(self, img, title, pos):
        img_RGB = img[:, :, ::-1]
        plt.subplot(3, 4, pos)
        plt.imshow(img_RGB)
        plt.title(title)
        plt.axis('off')

    def show_Gradientimg_plt(self, img, title, pos):
        plt.subplot(3, 4, pos)
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')

    def equalize_color(self, img):
        H, S, V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        eq_V = cv2.equalizeHist(V)
        eq_image = cv2.cvtColor(cv2.merge([H, S, eq_V]), cv2.COLOR_HSV2BGR)
        return eq_image


class Gray_scale_equalization:
    def getGrayequalization(self):
        obj=Histogram()
        plt.figure(figsize=(18, 14))
        plt.suptitle("Grayscale histogram equalization", fontsize=16, fontweight='bold')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_image_eq = cv2.equalizeHist(gray_image)
        M = np.ones(gray_image.shape, dtype="uint8") * 35
        added_image = cv2.add(gray_image, M)
        added_image_eq = cv2.equalizeHist(added_image)
        subtracted_image = cv2.subtract(gray_image, M)
        subtracted_image_eq = cv2.equalizeHist(subtracted_image)
        # Plot the images and the histograms (without equalization first):
        obj.show_img_plt(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray", 1)
        obj.show_img_plt(cv2.cvtColor(added_image, cv2.COLOR_GRAY2BGR), "gray lighter", 5)
        obj.show_img_plt(cv2.cvtColor(subtracted_image, cv2.COLOR_GRAY2BGR), "gray darker", 9)

        # Plot the images and the histograms (with equalization):
        obj.show_img_plt(cv2.cvtColor(gray_image_eq, cv2.COLOR_GRAY2BGR), "grayscale equalized", 3)
        obj.show_img_plt(cv2.cvtColor(added_image_eq, cv2.COLOR_GRAY2BGR), "gray lighter equalized", 7)
        obj.show_img_plt(cv2.cvtColor(subtracted_image_eq, cv2.COLOR_GRAY2BGR), "gray darker equalized", 11)
        #plt.show()

class Color_scale_equalization:
    def getcolorequalization(self):
        obj = Histogram()
        plt.figure(figsize=(18, 14))
        plt.suptitle("Color histogram equalization", fontsize=14,
                     fontweight='bold')
        image_eq = obj.equalize_color(image)
        M = np.ones(image.shape, dtype="uint8") * 15
        added_image = cv2.add(image, M)
        added_image_eq = obj.equalize_color(added_image)
        subtracted_image = cv2.subtract(image, M)
        subtracted_image_eq = obj.equalize_color(subtracted_image)

        obj.show_img_plt(image, "image", 1)
        obj.show_img_plt(added_image, "image lighter", 5)
        obj.show_img_plt(subtracted_image, "image darker", 9)

        # Plot the images and the histograms (with equalization)
        obj.show_img_plt(image_eq, "image equalized", 3)
        obj.show_img_plt(added_image_eq, "image lighter equalized", 7)
        obj.show_img_plt(subtracted_image_eq, "image darker equalized", 11)

        # Show the Figure:
        #plt.show()
class Gradient_of_Images:
    def getGradientImage(self):
        obj=Histogram()
        plt.figure(figsize=(18, 14))
        plt.suptitle("Image Gradients Of Gray Scale Image", fontsize=16, fontweight='bold')
        image = cv2.imread('embark-golden-retriever-puppy.jpg')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gX = cv2.Sobel(gray_image, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gX = cv2.convertScaleAbs(gX)
        obj.show_Gradientimg_plt(gray_image, "Original Image", 1)
        obj.show_Gradientimg_plt(gX, "Gradient in X Direction", 3)
        gY = cv2.Sobel(gray_image, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)
        gY = cv2.convertScaleAbs(gY)
        obj.show_Gradientimg_plt(gray_image, "Original Image", 5)
        obj.show_Gradientimg_plt(gY, "Gradient in Y Direction", 7)
        magnitude = cv2.addWeighted(gX, 0.5, gY, 0.5, 0)
        obj.show_Gradientimg_plt(magnitude, "Gradient Magnitude", 9)
        threshA, threshB = cv2.threshold(magnitude, 100, 255, cv2.THRESH_BINARY)
        obj.show_Gradientimg_plt(threshB, "Threshold magnitude", 11)

global image
image = cv2.imread('embark-golden-retriever-puppy.jpg')
obj = Gray_scale_equalization()
obj.getGrayequalization()
obj2=Color_scale_equalization()
obj2.getcolorequalization()
obj3=Gradient_of_Images()
obj3.getGradientImage()
plt.show()
