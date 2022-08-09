

# from skimage.util import compare_images
from PIL import ImageFile, Image
import numpy as np
import inspect
from os import path, listdir
ImageFile.LOAD_TRUNCATED_IMAGES = True # used to stop errors with file size



class general:
    # ================================= #
    # General Processes

    def PIL_to_scikit_or_openCV(pil_image):
        # Make Numpy array for scikit-image from "PIL Image"
        numpy_array_image = np.array(pil_image)
        return numpy_array_image

    def numpy_array_to_PIL(numpy_array):
        # Make "PIL Image" from Numpy array
        pil_image = Image.fromarray(numpy_array)
        return pil_image

    def print_functions(self):
        print(inspect.getmembers(predicate=inspect.isfunction))

    def load_images_in_folder(path):
        # return array of images

        imagesList = listdir(path)
        loadedImages = []
        for image in imagesList:
            img = Image.open(path + image)
            loadedImages.append(img)

        return loadedImages