

# from skimage.util import compare_images
from PIL import ImageFile, Image, ImageOps, ImageFilter, ImageEnhance
import numpy as np
import cv2
ImageFile.LOAD_TRUNCATED_IMAGES = True # used to stop errors with file size

class PIL_processes:
    # ================================= #
    # PIL processes
    def threshold_image(img, threshold_value):
        # standard method of thresholding - requires a PIL image 
        im = img.point( lambda p: 255 if p>threshold_value else 0)
        return im 

    # cv2 processes 
    def find_largest_contour(image, min_area):
        """
        This function finds all the contours in an image and return the largest
        contour area.
        :param image: a binary image
        """
        # add min area here
        image = image.astype(np.uint8)
        # contours, hierarchy
        (_, cnts, _) = cv2.findContours(
            image,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            # areaMin = cv2.getTrackbarPos("Area", "Parameters")
            if area > min_area:
                largest_contour = max(cnts, key=cv2.contourArea)
        return largest_contour

    def extract_foreground_from_frame(img, threshold1, threshold2, min_area, output_folder_path):

        image = img
        # show('Input image', image)
        # blur the image to smmooth out the edges a bit, also reduces a bit of noise
        # imgBlur = cv2.blur(image,(3, 3))
        imgBlur = cv2.GaussianBlur(image, (5, 5), 0)
        # convert the image to grayscale 
        gray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)

        # optional switch up
        imgCanny = cv2.Canny(gray, threshold1, threshold2)
        kernel = np.ones((5,5))

        imgDil = cv2.dilate(imgCanny, kernel, iterations = 1)
        contour = self.find_largest_contour(imgDil, min_area)

        # apply thresholding to conver the image to binary format
        # after this operation all the pixels below 200 value will be 0...
        # and all th pixels above 200 will be 255
        # I think I should change these values to threshold1 and 2
        #ret, gray = cv2.threshold(gray, 200 , 255, cv2.CHAIN_APPROX_NONE)

        # find the largest contour area in the image
        # contour = find_largest_contour(gray)
        image_contour = np.copy(image)
        cv2.drawContours(image_contour, [contour], 0, (0, 255, 0), 2, cv2.LINE_AA, maxLevel=1)
        # show('Contour', image_contour)

        # create a black `mask` the same size as the original grayscale image 
        mask = np.zeros_like(gray)
        # fill the new mask with the shape of the largest contour
        # all the pixels inside that area will be white 
        cv2.fillPoly(mask, [contour], 255)
        # create a copy of the current mask
        res_mask = np.copy(mask)
        res_mask[mask == 0] = cv2.GC_BGD # obvious background pixels
        res_mask[mask == 255] = cv2.GC_PR_BGD # probable background pixels
        res_mask[mask == 255] = cv2.GC_FGD # obvious foreground pixels

        # create a mask for obvious and probable foreground pixels
        # all the obvious foreground pixels will be white and...
        # ... all the probable foreground pixels will be black
        mask2 = np.where(
            (res_mask == cv2.GC_FGD) | (res_mask == cv2.GC_PR_FGD),
            255,
            0
        ).astype('uint8')


        # create `new_mask3d` from `mask2` but with 3 dimensions instead of 2
        new_mask3d = np.repeat(mask2[:, :, np.newaxis], 3, axis=2)
        mask3d = new_mask3d
        mask3d[new_mask3d > 0] = 255.0
        mask3d[mask3d > 255] = 255.0
        # apply Gaussian blurring to smoothen out the edges a bit
        # `mask3d` is the final foreground mask (not extracted foreground image)
        mask3d = cv2.GaussianBlur(mask3d, (5, 5), 0)
        # show('Foreground mask', mask3d)

        # create the foreground image by zeroing out the pixels where `mask2`...
        # ... has black pixels
        foreground = np.copy(image).astype(float)
        foreground[mask2 == 0] = 0
        # cv2.imshow('Foreground', foreground.astype(np.uint8))

        # TO DO - save the images to disk - TO DO!
        # save_name = os.path.basename(input_file_path).strip(".png")

        # cv2.imwrite(os.path.join(output_folder_path, f"{save_name}_foreground.png"), foreground)
        # cv2.imwrite(os.path.join(output_folder_path, f"{save_name}_foreground_mask.png"), mask3d)
        return foreground
    
    # def extract_foreground_real_python(img)

    def grayscale(img):
        grayscale_img = img.convert('L')
        return grayscale_img

    def solarize(img, threshold_val):
        # threshold 0 - 255  (start with 130 e.g)
        solarized_img = ImageOps.solarize(img, threshold = threshold_val)
        return solarized_img

    def resize_image(img, w, h):
        newsize = (w, h)
        im1 = img.resize(newsize)
        return im1

    def PIL_edges(img):
        img_gray = img.convert("L")
        edged = img_gray.filter(ImageFilter.FIND_EDGES)
        return edged    

    def PIL_edges_smooth(img):
        img_gray = img.convert("L")
        img_smooth = img_gray.filter(ImageFilter.SMOOTH)
        edged = img_smooth.filter(ImageFilter.FIND_EDGES)
        return edged
    
    def PIL_enhance_edges(img):
        img_gray = img.convert("L")
        edge_enhance = img_gray.filter(ImageFilter.EDGE_ENHANCE)
        return edge_enhance
    
    def PIL_contour(img, threshold):
        img_l = img.convert("L")
        img_l = img_l.point(lambda x: 255 if x > threshold else 0)
        img_l = img_l.filter(ImageFilter.CONTOUR)
        return img_l

    def PIL_invert(img):
        img = img.point(lambda x: 0 if x == 255 else 255)
        return img
    
    def PIL_contrast(img, contrast_amount):
        # contrast between 0 and 2
        enhancer = ImageEnhance.Contrast(img)
        im_output = enhancer.enhance(contrast_amount)
        return im_output

    def PIL_brightness(img, brightness_amount):
        enhancer = ImageEnhance.Brightness(img)
        new_image = enhancer.enhance(brightness_amount)
        return new_image

    def PIL_rotate(img, rotation):
        # 90 is counterclockwise by 90
        rotated = img.rotate(rotation, expand=True)
        return rotated
