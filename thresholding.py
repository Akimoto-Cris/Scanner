from cv2 import imread, imwrite, equalizeHist, adaptiveThreshold, waitKey
from cv2 import resize, imshow, IMREAD_GRAYSCALE, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_path", type=str, help="path to the image")
    parser.add_argument("-d", "--display", type=int, default=1,
                        help="whether showing the image after processing. (1 / 0). default is true. (optional)")
    parser.add_argument("-c", "--downsample", type=float,
                        help="the factor you want to shrink resolution of output. (optional)")
    return parser.parse_args()


def histeq(pic_path):
    image = imread(pic_path, IMREAD_GRAYSCALE)
    return equalizeHist(image)


def adaptive_thresholding(image, block_size, C):
    return adaptiveThreshold(image, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, block_size, C)


if __name__ == "__main__":
    args = get_args()
    histed_image = histeq(args.image_path)
    thresed_image = adaptive_thresholding(histed_image, 35, 20)
    if args.display:
        imshow("thresed_image", thresed_image)
        waitKey()

    if args.downsample:
        thresed_image = resize(thresed_image, (int(thresed_image.shape[1] // args.downsample),
                                   int(thresed_image.shape[0] // args.downsample)))
    imwrite("{}_1.jpg".format(args.image_path.split('.')[0]), thresed_image)
