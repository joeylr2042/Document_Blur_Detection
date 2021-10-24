import numpy as np
import cv2


class BlurOps:
    def __init__(self, img):
        self.img = img

    def gaussian_blur(self, kernel_size=9):
        blurred = cv2.GaussianBlur(self.img, ksize=(kernel_size, kernel_size), sigmaX=0, sigmaY=0)
        return blurred

    def motion_blur(self, degree=12, angle=135):
        img = np.array(self.img)

        matrix = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, matrix, (degree, degree))

        motion_blur_kernel = motion_blur_kernel / degree
        blurred = cv2.filter2D(img, -1, motion_blur_kernel)

        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        blurred = np.array(blurred, dtype=np.uint8)
        return blurred


if __name__ == "__main__":
    image = cv2.imread('./bean-license.png')
    blur_operator = BlurOps(image)
    motion_blurred = blur_operator.gaussian_blur()
    cv2.imwrite("./gaussian_blur.jpg", motion_blurred)

