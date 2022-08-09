# from skimage.util import compare_images
from PIL import ImageFile, Image
import numpy as np
import os
from os import path, listdir
ImageFile.LOAD_TRUNCATED_IMAGES = True # used to stop errors with file size
from skimage import io, util
from skimage.util import compare_images
from skimage import data, transform, exposure, filters
from skimage.filters import threshold_otsu, threshold_local
# class fefe_image_processes:
class scikit:
    # from skimage import io, util
    # from skimage import data, transform, exposure, filters
    # from skimage.filters import threshold_otsu, threshold_local
    def resize_image_for_screen(starter_img, width, height, dir_path):
        new_img = starter_img.resize((width, height), Image.ANTIALIAS)
        return new_img

    # dir_path tells the function where to look for images
    # img1/2_path are the paths of the images that will be compared 
    # in the first loop both will be from the contoured folder
    # after the first loop the first will be from the frame differenced folder and the second will be from the contoured folder
    # the output path argument tells the function where to save the image
    def frame_difference_paths(dir_path, img1_path, img2_path, input_position, image_counter, image_counter2, output_path):
        image1_path = os.path.join(dir_path, img1_path, str(input_position), str(image_counter)+'.png')
        image2_path = os.path.join(dir_path, img2_path, str(input_position), str(image_counter2)+'.png')
        img1 = io.imread(image1_path, as_gray=True)
        img2 = io.imread(image2_path, as_gray=True)
        diff_photo = util.compare_images(img1, img2, method='diff')
        frameDiff_directory = os.path.join(dir_path, 'frameDiff')
        output_path=os.path.join(frameDiff_directory, str(input_position), str(output_path)+'.png')
        io.imsave(output_path, diff_photo)

    # ================================= #
    # Scikit processes 
    def difference_images(img1, img2):
        # images need to be numpy arrays 
        diff_photo = compare_images(img1, img2, method='diff')
        return diff_photo


    def blend_images(img1, img2):
        # images need to be numpy arrays 
        diff_photo = compare_images(img1, img2, method='blend')
        return diff_photo

    def edge_detection_standard(img1):
        edged_img = filters.roberts(img1)
        return edged_img

    def edge_detection_sobel(img):
        edged_img = filters.roberts(img)
        return edged_img

    def edge_detection_scharr(img):
        edged_img = filters.scharr(img)
        return edged_img

    def edge_detection_farid_h(img):
        edged_img = filters.farid_h(img)
        return edged_img

    def edge_detection_farid_v(img):
        edged_img = filters.farid_v(img)
        return edged_img
    # meijering, sato, frangi, hessian, roberts, sobel, scharr, prewitt 
    def edge_detection_meijering(img):
        edged_img = filters.meijering(img)
        return edged_img

    def edge_detection_sato(img):
        edged_img = filters.sato(img)
        return edged_img

    def edge_detection_frangi(img):
        edged_img = filters.frangi(img)
        return edged_img

    def contrast_stretch(img):
        p2, p98 = np.percentile(img, (2, 98))
        # Contrast stretching
        img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98)) # rescale image based on contrast stretching - this draws the detail in images
        return img_rescale

    def invert_numpy_array(img):
        contrasted_inverted = util.invert(img) # invert image
        return contrasted_inverted

    # dithering

    # image separations 

    def local_threshold_image(img):
        # does a better job than standard thresholding but is slower
        # requires a scikit image input
        block_size = 35
        local_thresh = threshold_local(img, block_size, offset=10)
        return local_thresh
