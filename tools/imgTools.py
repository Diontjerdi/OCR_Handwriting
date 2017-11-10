""" This file contains all functions necessary to process images from scanned documents to characters that are suitable
to be fed to the CNN.
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from skimage.morphology import skeletonize
from skimage import img_as_ubyte  # To convert skimage skeleton to cv2


def binarise(img, kernel_size=3):
    """
    Binarise image by Otsu's thresholding after Gaussian filtering
    http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html
    :param img: (NumPy array) cv2.imread grayscale image
    :param kernel_size: in pixels
    :return: (NumPy array) binarised image
    """
    blur = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    _, binarised = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    return binarised


def deslant_char(img, im_size=32):
    """
    Remove skew from handwritten character.
    from: https://github.com/opencv/opencv/blob/master/samples/python/digits.py#L63
    :param img: (NumPy array) cv2.imread image
    :param im_size: (Int) Desired output image size
    :return: (NumPy array) Deslanted character image
    """
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5 * im_size * skew], [0, 1, 0]])
    img = cv2.resize(img, (im_size, im_size))
    img = cv2.warpAffine(img, M, (im_size, im_size), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


# def crop_border(img, tol=0):
#     """
#     Remove black rows and colums, EVERYWHERE, NOT JUST BORDERS.
#     img is binarised image data in a NumPy array.
#     tol is tolerance (higher values e.g. 80 give tighter boundary)
#     """
#     mask = img > tol
#     return img[np.ix_(mask.any(1), mask.any(0))]


def crop_border(img):
    """
    Crop outer black border from image.
    :param img: (NumPy array) binarised image.
    :return: (NumPy array) Cropped image.
    """
    img = Image.fromarray(img)

    image_box = img.getbbox()

    cropped = img.crop(image_box)
    cropped = np.array(cropped)

    return cropped


def process_char(img, skeleton=False, deslant=False, im_size=32):
    """
    Preprocess image -> convert to grayscale, binarize, crop borders, resize, skeletonize (optional)
    :param img: (NumPy array) Grayscale image of character
    :param skeleton: (bool) If the desired output is skeletonised or not
    :param deslant: (bool) If the desired output is deslanted or not
    :param im_size: (Int) Desired output image size
    :return: (NumPy array) Preprocessed image
    """

    # Binarise
    if not any([sum(row) % 255 == 0 for row in img]):  # Check if img is binary by checking if rows are multiple of 255
        img = binarise(img)

    # Invert black/white
    h, w = img.shape
    corners = [img[0][0], img[0][w-1], img[h-1][0], img[h-1][w-1]]
    if sum(corners) > (255 * 2):  # Check if background is white
        img = np.invert(img)

    # Crop black border
    img = crop_border(img)

    # Add padding
    img = add_padding(img)

    # Skeletonise
    if skeleton:
        img = img_as_ubyte(skeletonize(img / 255))

    # Deslant
    if deslant:
        img = deslant_char(img)

    # Resize to size*size
    if not img.shape == (im_size, im_size):
        img = cv2.resize(img, (im_size, im_size))

    return img


def segment_lines(img, show_proj=False):
    """
    Segment a document in to lines of text
    :param img: (NumPy array) cv2.imread grayscale image of document
    :param show_proj (Bool) whether to show a plot of the vertical projection
    :return: list of images (NumPy arrays) of lines of text
    """
    # Binarise
    img = binarise(img)

    # Invert black/white
    h, w = img.shape
    white_count = cv2.countNonZero(img)  # Count white pixels
    if white_count > (h * w - white_count):  # Check if background is white
        img = np.invert(img)
    # corners = [img[0][0], img[0][w - 1], img[h - 1][0], img[h - 1][w - 1]]  # Check corner pixels
    # if sum(corners) > (255 * 2):  # Check if background is white
    #     img = np.invert(img)

    # Project on line (1D)
    img_proj = img.max(axis=1)

    # Plot projection
    if show_proj:
        y = np.arange(len(img_proj))  # To swap x,y axes
        plt.gca().invert_xaxis()  # Mirror x
        plt.gca().invert_yaxis()  # Mirror y
        plt.xlabel("Pixel intensity (255=black, 0=white)")
        plt.ylabel("y-axis")
        plt.plot(img_proj, y)
        plt.show()

    # Find cuts where image projection goes from white to black.
    prev_pix = 0
    cuts_index = [0]  # Add 0 for begin first cut.
    for i in range(len(img_proj)):
        if prev_pix == 255 and img_proj[i] == 0:  # Projection goes from white (255) to black (0)
            cuts_index.append(i)
        prev_pix = img_proj[i]

    # Make horizontal cuts
    cuts = []
    for i in np.arange(1, len(cuts_index)):
        cut = img[cuts_index[i - 1]: cuts_index[i], :]  # Cut out the region of character

        # Crop only outer black border
        cut = crop_border(cut)
        cuts.append(cut)

    return cuts


def segment_words(img, kernel_w=37, show_proj=False):
    """
    Segment a line of text into words
    :param img: (NumPy array) grayscale image of line of text
    :param kernel_w: (Int) Kernel width, how much to dilate image horizontally
    :param show_proj: (Bool) Plot vertical projection of line
    :return: (List) list of images (NumPy arrays) of words
    """
    # Binarise
    img = binarise(img)

    # Dilation
    # kernel = np.ones((5, 5), np.uint8)
    kernel = np.ones((5, kernel_w), np.uint8)
    dil_img = cv2.dilate(img, kernel, iterations=1)

    # Project image on line (1D)
    img_proj = dil_img.max(axis=0)

    # Plot projection of image
    if show_proj:
        plt.ylabel("Pixel intensity (0=white, 255=black,)")
        plt.plot(img_proj)
        plt.show()

    # Find cuts where image projection goes from white to black.
    prev_pix = 0
    cuts_index = [0]  # Add 0 for begin first cut.
    for i in range(len(img_proj)):
        if prev_pix == 255 and img_proj[i] == 0:  # Projection goes from white (255) to black (0)
            cuts_index.append(i)
        prev_pix = img_proj[i]
    cuts_index.append(len(img_proj))

    # Make vertical cuts
    cuts = []
    for i in np.arange(1, len(cuts_index)):
        cut = img[:, cuts_index[i - 1]: cuts_index[i]]  # Cut out the region of character
        cuts.append(cut)

    return cuts


def segment_chars(img, padding=True, show_proj=False):
    """
    Segment characters by finding vertical cuts
    :param img: (NumPy array) Binarized grayscale image of word in cv2.imread format
    :param padding: (bool) Whether to add padding so that image becomes square
    :param show_proj: (Bool) Plot vertical projection of line
    :return: (NumPy array) Array of cut characters from the input image
    """
    # Binarise image
    img = binarise(img)

    # Normalise image
    # img[img == 255] = 1

    # Project image on line (1D)
    img_proj = img.max(axis=0)

    # Plot projection of image
    if show_proj:
        plt.ylabel("Pixel intensity (0=white, 255=black,)")
        plt.plot(img_proj)
        plt.show()

    # Find cuts where image projection goes from white to black.
    prev_pix = 0
    cuts_index = [0]  # Add 0 for begin first cut.
    for i in range(len(img_proj)):
        if prev_pix == 255 and img_proj[i] == 0:  # Projection goes from white (255) to black (0)
            cuts_index.append(i)
        prev_pix = img_proj[i]
    if img_proj[-1] == 255:
        cuts_index.append(len(img_proj))  # Add for end last cut.

    # Make vertical cuts
    cuts = []
    for i in np.arange(1, len(cuts_index)):
        cut = img[:, cuts_index[i - 1]: cuts_index[i]]  # Cut out the region of character
        cut = crop_border(cut)  # Crop cuts to fit tightly around characters

        if padding:
            cut_new = add_padding(cut)
            cuts.append(cut_new)
        else:
            cuts.append(cut)

    # # Draw cuts
    #     img = cv2.line(img, (cuts_index[i], 0), (cuts_index[i], img.shape[0]), (255, 0, 0), 1)
    # cv2.imshow('slices', img)
    # cv2.waitKey

    return cuts


def segment_chars_contours(img, show_cuts=False):
    """
    Segment characters by finding contours.
    TODO: deal with letter 'i' and small specks of ink
    :param img: Binarized grayscale image of word in cv2.imread format (NumPy array)
    :param show_cuts: (Bool) show the character cutouts
    :return: NumPy arrays of cut characters from the input image
    """

    # Blur for improving contours
    blurred_img = cv2.GaussianBlur(img, (3, 3), 0)
    blurred_img, contours, _ = cv2.findContours(blurred_img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

    # Grayscale to BGR to allow drawing of green line
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR, img)

    # cv2.drawContours(img, contours, -1, (0,255,0), 3)  # Draw actual contours, not bounding box

    cuts = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cut = img[y: y + h, x: x + w]  # Cut out the region of interest
        cuts.append(cut)

        if show_cuts:
            cv2.imshow("Cut-out", cut)
            cv2.waitKey()
            cv2.destroyAllWindows()

    cv2.imshow("Segmented characters", img)
    cv2.waitKey()

    return cuts


def add_padding(img):
    """
    Makes an image square by adding padding in the form of black borders.
    :param img: (NumPy array) binarised image with black background.
    :return: (NumPy array) image with padding.
    """
    h, w = img.shape
    size = max(h, w)
    img_new = np.zeros((size, size), np.uint8)
    img_new[int(size / 2 - h / 2): int(size / 2 + h / 2), int(size / 2 - w / 2): int(size / 2 + w / 2)] = img
    return img_new


def process_doc(doc, show_lines=False, show_words=False, show_chars=False, show_proj=False, char_padding=True,
                word_kernel_w=37):
    """
    Process a scanned document by segmenting it into lines, then words, then characters and preprocessing the characters
    :param doc: (NumPy array) cv2.imread grayscale image of document to be processed.
    :param show_lines: (Bool) show extracted lines
    :param show_words: (Bool) show extracted words
    :param show_chars: (Bool) show extracted characters
    :param show_proj: (Bool) show projections of the histograms
    :param char_padding: (Bool) add padding to characters or not
    :param word_kernel_w: (Int) kernel width for horizontal dilation
    :return: (NumPy array) Grayscale images of characters, (NumPy array) array of word lengths.
    """

    characters = []
    word_lens = []

    lines = segment_lines(doc, show_proj=show_proj)
    l = 0
    for line in lines:
        if show_lines:
            cv2.imshow("line", line)
            cv2.waitKey()

        words = segment_words(line, show_proj=show_proj, kernel_w=word_kernel_w)
        l += 1
        w = 0
        for word in words:
            if show_words:
                cv2.imshow("word", word)
                cv2.waitKey()

            chars = segment_chars(word, padding=char_padding, show_proj=show_proj)
            word_lens.append(len(chars))
            w += 1
            c = 0
            for char in chars:
                if show_chars:
                    cv2.imshow("char", char)
                    cv2.waitKey()

                # cv2.imwrite((r"data\test_dion\\" + str(l) + "_" + str(w) + "_" + str(c) + ".png"), char)
                characters.append(char)
                c += 1

    return np.asarray(characters, dtype=object), np.asarray(word_lens)


if __name__ == '__main__':
    doc = cv2.imread(r"test_doc.png", 0)
    chars, word_lens = process_doc(doc, show_lines=True, show_words=False, show_chars=False, show_proj=False)
