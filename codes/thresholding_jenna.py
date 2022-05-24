# ----------------------------------------------------------------------------------------------- #
# Set the working directory, where all the data is stored:
wd = "S:/mdc_work/jenna"

# The threshold will be applied to the following folders/conditions:
folders_list = ["Jennaimages"]

# Choose a threshold mode
threshold_mode = "otsu"
# Other options:
#  - "super_low_intensities_filtered"
#  - "triangle_on_dapi_intensity_greater_1_on_rest"
#  - "otsu_on_dapi_intensity_greater_1_on_rest"
#  - "otsu_on_dapi_only",
#  - "otsu",
#  - "triangle",
#  - "adaptive"

# Want to apply a gaussian blur filter too?
gauss_blur_filter = False

file_format = ".tif"

## Names of the markers as in the file names.
# e.g. "C3" mis the notation for DAPI. 'C' stands apperantly for "channel" and '3' is its number, set by the microscope.
ch_prefix = "C"
ch1_suffix = "1"
ch2_suffix = "2"
ch3_suffix = "3"
ch4_suffix = "4"

# ----------------------------------------------------------------------------------------------- #

import os, glob
import cv2
from tqdm import tqdm

pic_folder_path = os.path.join(wd, folders_list[0])
os.chdir(pic_folder_path)

## Read a file
# input: "file name" string
def read_image(file):
    # read the image in while maintaining the original bit-depth ('-1')
    img = cv2.imread(file, -1)
    return img

## Read 4 corresponding greyscale images
def read_4_color_channels(file_name):
    base_channel = ch_prefix + ch1_suffix
    ch1 = cv2.imread(file_name, -1)
    ch2 = cv2.imread(file_name.replace(base_channel, ch_prefix + ch2_suffix), -1)
    ch3 = cv2.imread(file_name.replace(base_channel, ch_prefix + ch3_suffix), -1)
    ch4 = cv2.imread(file_name.replace(base_channel, ch_prefix + ch4_suffix), -1)
    return ch1, ch2, ch3, ch4

## Apply thresholding to every color channel of the image.
# input: "folder name" string
def thresholding(pic_folder_path, pic_sub_folder_name, mode = "low_intensities_filtered", gaussian_blur = True):
    # Set the folder up, in which the thresholded images will be saved:
    if not os.path.isdir(pic_folder_path + f"/../{pic_sub_folder_name}_thresholded_{mode}"):
        os.makedirs(pic_folder_path + f"/../{pic_sub_folder_name}_thresholded_{mode}")
    # We're gonna save the images here:
    os.chdir(pic_folder_path + f"/../{pic_sub_folder_name}_thresholded_{mode}")

    for file in tqdm(glob.glob(pic_folder_path+"/C1*"), desc=f"Applying {mode} thresholding"):

        if os.path.isfile(file.replace(file_format, f"gauss_filter_{gaussian_blur}_{mode}_thresholded{file_format}")):
            continue

        ch1, ch2, ch3, ch4 = read_4_color_channels(file)

        if gaussian_blur:
            # Apply a Gaussian blur filter to the image
            ch1 = cv2.GaussianBlur(ch1, (5, 5), 0)
            ch2 = cv2.GaussianBlur(ch2, (5, 5), 0)
            ch3 = cv2.GaussianBlur(ch3, (5, 5), 0)
            ch4 = cv2.GaussianBlur(ch4, (5, 5), 0)

        if mode == "triangle":
            # Apply triangle thresholding to every channel
            _, th1 = cv2.threshold(ch1, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_TRIANGLE)
            _, th2 = cv2.threshold(ch2, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_TRIANGLE)
            _, th3 = cv2.threshold(ch3, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_TRIANGLE)
            _, th4 = cv2.threshold(ch4, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_TRIANGLE)

        if mode == "adaptive":
            # Apply cv adaptive thresholding to every channel
            th1 = cv2.adaptiveThreshold(ch1, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 0)
            th2 = cv2.adaptiveThreshold(ch2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 0)
            th3 = cv2.adaptiveThreshold(ch3, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 0)
            th4 = cv2.adaptiveThreshold(ch4, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 0)

        if mode == "otsu":
            # Apply Otsu's thresholding to every channel
            _, th1 = cv2.threshold(ch1, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
            _, th2 = cv2.threshold(ch2, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
            _, th3 = cv2.threshold(ch3, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
            _, th4 = cv2.threshold(ch4, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)

        if mode == "otsu_on_dapi_only":
            # Apply Otsu's thresholding to only the DAPI channel
            th1 = ch1
            th2 = ch2
            _, th3 = cv2.threshold(ch3, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
            th4 = ch4

        if mode == "otsu_on_dapi_intensity_greater_1_on_rest":
            # Apply Otsu's thresholding to only the DAPI channel
            # Every value >1 remains the same, every value <=1 is set to 0
            th1 = ch1
            th2 = ch2
            _, th3 = cv2.threshold(ch3, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_OTSU)
            th4 = ch4
            th1[th1 < 2] = 0 
            th2[th2 < 2] = 0
            th4[th4 < 2] = 0

        if mode == "triangle_on_dapi_intensity_greater_1_on_rest":
            # Apply Otsu's thresholding to only the DAPI channel
            _, th1 = cv2.threshold(ch1, 0, 255, cv2.THRESH_TOZERO + cv2.THRESH_TRIANGLE)
            # Every value >1 remains the same, every value <=1 is set to 0
            th2 = ch2
            th3 = ch3
            th4 = ch4
            th2[th2 < 2] = 0
            th3[th3 < 2] = 0
            th4[th4 < 2] = 0

        if mode == "super_low_intensities_filtered":
            th1 = ch1
            th2 = ch2
            th3 = ch3
            th4 = ch4
            # Every value >1 remains the same, every value <=1 is set to 0
            th1[th1 < 2] = 0
            th2[th2 < 2] = 0
            th3[th3 < 2] = 0
            th4[th4 < 2] = 0

        if mode == "low_intensities_filtered":
            th1 = ch1
            th2 = ch2
            th3 = ch3
            th4 = ch4
            # Every value >1 remains the same, every value <=1 is set to 0
            th1[th1 < 5] = 0
            th2[th2 < 5] = 0
            th3[th3 < 5] = 0
            th4[th4 < 5] = 0

        ## Save images
        #for file, img in zip([file.replace(file_format, f"gauss_filter_{gaussian_blur}_{mode}_thresholded{file_format}")], [th1, th2, th3, th4]):
        #    cv2.imwrite(file, img)
        base_channel = ch_prefix + ch1_suffix
        cv2.imwrite(os.path.basename(file.replace(file_format, f"_gauss_filter_{gaussian_blur}_{mode}_thresholded{file_format}")), th1)
        file2_replaced_name = file.replace(base_channel, ch_prefix + ch2_suffix)
        cv2.imwrite(os.path.basename(file2_replaced_name.replace(file_format, f"_gauss_filter_{gaussian_blur}_{mode}_thresholded{file_format}")), th2)
        file3_replaced_name = file.replace(base_channel, ch_prefix + ch3_suffix)
        cv2.imwrite(os.path.basename(file3_replaced_name.replace(file_format, f"_gauss_filter_{gaussian_blur}_{mode}_thresholded{file_format}")), th3)
        file4_replaced_name = file.replace(base_channel, ch_prefix + ch4_suffix)
        cv2.imwrite(os.path.basename(file4_replaced_name.replace(file_format, f"_gauss_filter_{gaussian_blur}_{mode}_thresholded{file_format}")), th4)
    return

if 1:
    for sub_folder_name in folders_list:
        pic_folder_path = os.path.join(wd, sub_folder_name)
        os.chdir(pic_folder_path)
        thresholding(pic_folder_path, sub_folder_name, mode = threshold_mode, gaussian_blur = gauss_blur_filter)
