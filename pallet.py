"""
These were computed based on the original images and it is best to update these pellet sizes in pixels.
Measurement shows in the near field a pellet is an ellipse with dimensions 9px X 12px thus pellet area is 339.3 px2
Measurement shows in the far field a pellet is an ellipse with dimensions 5px X 6px thus pellet area is 94.24 px2

To operate on an image:
python3 pellet_counter.py white_image_1.png --debug

To operate from webcam
python3 pellet_counter.py 0 --debug
"""
import os
import argparse
import copy

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch


def is_raspberrypi():
    try:
        with io.open('/sys/firmware/devicetree/base/model', 'r') as m:
            if 'raspberry pi' in m.read().lower(): return True
    except Exception: pass
    return False


if is_raspberrypi():
    from picamera.array import PiRGBArray
    from picamera import PiCamera


checkpoint = "vinvino02/glpn-nyu"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)
depth_estimator = AutoModelForDepthEstimation.from_pretrained(checkpoint)

# depth_estimator = pipeline("depth-estimation", model=checkpoint)

'''
USER_SUPPLIED_CAGE_RADIUS, USER_SUPPLIED_PELLET_SIZE, USER_SUPPLIED_DISTANCE_OF_CAMERA_BASE_FROM_CAGE and
one of USER_SUPPLIED_MOUNT_HEIGHT or USER_SUPPLIED_CAMERA_ELEVATION must be provided by user
'''
USER_SUPPLIED_CAGE_RADIUS = 12.0  # In meters
USER_SUPPLIED_CAMERA_ELEVATION = 30.0  # In degrees
USER_SUPPLIED_PELLET_SIZE = 5.0  # In millimeters
USER_SUPPLIED_MOUNT_HEIGHT = 1.5  # In meters (difference between camera height and water level in the cage)
USER_SUPPLIED_DISTANCE_OF_CAMERA_BASE_FROM_CAGE = 0  # In meters


MIN_DEPTH = np.sqrt(USER_SUPPLIED_MOUNT_HEIGHT ** 2 + USER_SUPPLIED_DISTANCE_OF_CAMERA_BASE_FROM_CAGE ** 2)
MAX_DEPTH = np.sqrt(USER_SUPPLIED_MOUNT_HEIGHT ** 2 + (USER_SUPPLIED_DISTANCE_OF_CAMERA_BASE_FROM_CAGE +
                                                       USER_SUPPLIED_CAGE_RADIUS) ** 2)
CAGE_AREA = np.pi * USER_SUPPLIED_CAGE_RADIUS ** 2


class PelletCounter:
    # Should be listed from near-field to far-field
    PIXEL_AREAS = [339.3, 94.24]

    def __init__(self, args, image=None):
        super().__init__()
        self.DEBUG = args.debug
        # Get the camera matrix
        self.camera_model = args.matrix

        # Get the lower and upper values of HSV fish pellets
        self.hl, self.sl, self.ll = [int(num) for num in args.hsvlp.split(',')]
        self.hu, self.su, self.lu = [int(num) for num in args.hsvup.split(',')]

        # Get the lower and upper values of HSV for water segmentation
        self.hlw, self.slw, self.llw = [int(num) for num in args.hsvlw.split(',')]
        self.huw, self.suw, self.luw = [int(num) for num in args.hsvuw.split(',')]

        # Read image and compute the HSV
        if image is not None:
            self.img = image
        else:
            self.img = cv2.imread(args.img_path)

        self.hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HLS)
        out_dir = args.img_path.split('/')[-1].split('.')[0]
        os.makedirs(out_dir, exist_ok=True)
        os.chdir(out_dir)

        if self.DEBUG:
            print(f"Input image is {args.img_path}")
            print(f"Fish pellet HLS range: {self.hl, self.hu, self.ll, self.lu, self.sl, self.su}")
            print(f"Water HLS range: {self.hlw, self.huw, self.llw, self.luw, self.slw, self.suw}")
            cv2.imwrite(f"Step_0_HSV_output.png", self.hsv)
            arrays = self.hsv.reshape((-1, 3))
            plt.hist(arrays, bins=100, histtype='step', density=True, range=(1, 255))
            plt.legend(['hue', 'luminance', 'saturation'])
            plt.savefig("HSV_distribution.png")

    @staticmethod
    def compute_depth(img):
        """
        This function is used for monocular depth estimation using GLPN.
        :param img: A numpy.ndarray of shape (height, weight, 3)
        :return: An estimated depth of pixels in the scene
        """
        # Convert image to PIL Image
        image = Image.fromarray(img)

        # Downscale image for deep learning operations
        original_size = image.size
        downscale_ratio = max(original_size) / 300
        new_size = tuple(
            [int(round(i / downscale_ratio)) for i in original_size]
        )
        image = image.resize(new_size)
        pixel_values = image_processor(image, return_tensors="pt").pixel_values
        # Estimate the depth using GLPN and return it
        with torch.no_grad():
            outputs = depth_estimator(pixel_values)
            depth = outputs.predicted_depth
            depth = depth.squeeze()
            depth = Image.fromarray(depth.cpu().numpy())
        #depth = predictions['depth']

        # Upsample computed depth image to the original size
        depth = depth.resize(original_size)
        return np.asarray(depth)

    def segment_water(self):
        """
        Segments out the water body, masks out non-water pixels and returns the computed mask.
        :return: A tuple of
        """

        hsv = self.hsv
        img = self.img

        # Define HSV ranges to be used for color segmentation
        lower_color = np.array([self.hlw, self.slw, self.llw])
        upper_color = np.array([self.huw, self.suw, self.luw])

        # Create an entirely white image and mask the non-water pixels out
        white_img = np.ones_like(img) * 255

        # Find the mask that segments the water out
        water_mask = cv2.inRange(hsv, lower_color, upper_color)

        # Retain whiteness for the water pixels only
        res = cv2.bitwise_and(white_img, white_img, mask=water_mask)
        _, binary_res = cv2.threshold(res[..., 1], 127, 255, cv2.THRESH_BINARY)

        # kernel = np.ones((53, 53), np.uint8)
        # Run a morphological closing to fill holes inside the water while avoiding creating bridges between components
        # [CHANGE]: Kernel size hanged from 35 -> 5
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closing = cv2.morphologyEx(binary_res, cv2.MORPH_CLOSE, kernel)

        # Detect connected components and keep only the largest connected component
        # This is a more accurate approach but this can be achieved in a simpler way (simple contour detection) for
        # faster computation
        components = cv2.connectedComponentsWithStats(closing, 4, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = components

        water_mask = np.zeros(closing.shape, dtype="uint8")
        max_comp_area = stats[:, cv2.CC_STAT_AREA].max()

        # For loop: To increase the accuracy and robustness of the water mask detection
        for i in range(1, numLabels):
            # extract the connected component statistics for the current
            # label
            area = stats[i, cv2.CC_STAT_AREA]
            # keepWidth = w > 5 and w < 50
            # keepHeight = h > 45 and h < 65
            # Keep the connected component with the maximum area
            keepArea = (area == max_comp_area)

            if all((keepArea,)):
                # print("[INFO] keeping connected component '{}'".format(i))
                componentMask = (labels == i).astype("uint8") * 255
                water_mask = cv2.bitwise_or(water_mask, componentMask)

        # [CHANGE]: This dilation has been moved to later stages
        # Run a morphological dilation to fill penisulas inside the water
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
        # water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_DILATE, kernel)

        # Find the boundary of the water body, RETR_EXTERNAL so no contours inside the water contour are detected
        contours, hierarchy = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # print("#contours:", len(contours))

        img_with_cont = copy.deepcopy(img)

        # Panic! No water body detected or wrong input given
        if contours is None or len(contours) < 1:
            return img_with_cont, None, binary_res, water_mask

        # Sort the contours by their contour area
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Pick only bodies that are large enough, assume dimensions of the pond will be given as input
        filtered_contours = [sorted_contours[0]]
        # filtered_contours = list(
        #     filter(lambda g: cv2.contourArea(g) > 400000, sorted_contours)
        # )
        # [CHANGE]: Handle white images(generally holes inside water body) Fill the water contour to avoid segmented portions
        water_mask = np.zeros_like(img)
        cv2.fillPoly(water_mask, pts=filtered_contours, color=(255, 255, 255))
        water_mask = water_mask[..., 0]

        # [CHANGE]: This dilation was moved here form earlier stages
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (45, 45))
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_DILATE, kernel)

        # Compute the bounding rectangle of the water body, will be used later
        water_bounding_rect = cv2.boundingRect(filtered_contours[0])

        #
        # closing_rgb = cv2.cvtColor(closing, cv2.COLOR_GRAY2BGR)
        # cv2.drawContours(closing_rgb, filtered_contours, 0, (0, 255, 0), 1)
        # cv2.imwrite('morph_out.png', closing_rgb)

        water_only = cv2.bitwise_and(img, img, mask=water_mask)
        water_only_hsv_img = cv2.bitwise_and(hsv, hsv, mask=water_mask)

        depth_img = self.compute_depth(water_only)

        if self.DEBUG:
            cv2.imwrite(f"Step_1_water_mask.png", water_mask)
            cv2.drawContours(img_with_cont, contours, -1, (0, 255, 0), 1)
            cv2.imwrite(f"Step_3_water_contour.png", img_with_cont)
            cv2.imwrite(f"Step_4_segmented_water.png", water_only)
            cv2.imwrite(f"Step_5_depth_image.png", depth_img)

        # Ignore computed depth for non-water pixels
        depth_img = np.ma.array(depth_img.astype(np.float32), mask=~water_mask)
        depth_img = depth_img.filled(-1)
        water_area = cv2.contourArea(filtered_contours[0])

        return water_only, water_only_hsv_img, water_mask, water_bounding_rect, depth_img, water_area

    def segment_pellets(self, water_only_img, water_only_hsv_img):
        """

        :param water_only_img: An RGB image that contains pixels within the water body with the rest blacked out
        :param water_only_hsv_img: An HSV image that contains pixels within the water body with the rest blacked out
        :return:
        """
        hsv = water_only_hsv_img
        img = water_only_img

        # Define HSV ranges to be used for color segmentation
        # -------------------------------------------------------
        # This section is expected to be changed for infrared images
        lower_color = np.array([self.hl, self.sl, self.ll])
        upper_color = np.array([self.hu, self.su, self.lu])

        # Find the mask that segments the water out
        white_img = np.ones_like(img) * 255
        pellet_mask = cv2.inRange(hsv, lower_color, upper_color)
        # Replace with a grayscale thresholding operation
        # --------------------------------------------------------------

        # Adding the following to ensure robustness
        res = cv2.bitwise_and(white_img, white_img, mask=pellet_mask)
        _, binary_res = cv2.threshold(res[..., 1], 127, 255, cv2.THRESH_BINARY)

        # Find the boundaries of the detected pellets
        contours, hierarchy = cv2.findContours(binary_res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # print("#contours:", len(contours))
        img_with_cont = copy.deepcopy(img)

        if contours is None or len(contours) < 1:
            return img_with_cont, None, binary_res, pellet_mask

        # sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)

        if self.DEBUG:
            cv2.imwrite(f"Step_6_pellet_mask.png", pellet_mask)
            cv2.drawContours(img_with_cont, contours, -1, (0, 0, 255), 1)
            cv2.imwrite(f"Step_7_pellet_contour.png", img_with_cont)

        return img_with_cont, contours, binary_res, pellet_mask

    @staticmethod
    def contour_moment(contour):
        '''
        Determines the centroid of a contour
        :param contour:
        :return:
        '''
        M = cv2.moments(contour)
        # Some contours are singular
        if M['m00'] < 1e-8:
            return -1, -1

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        return cx, cy

    @staticmethod
    def is_centroid_in_region(centroid, depth_mask):
        cx, cy = centroid
        # If the contour had a bad moment return False
        if cx < 0 or cy < 0:
            return False
        # Otherwise
        return depth_mask[cy, cx] == 1  # Note, OpenCV and Numpy x- and y- axes subtlety

    def count_pellets(self, N=2):
        """
        This function counts fish pellets and
        :param N:
        :return:
        """
        # Segment the water first
        water_only_img, water_only_hsv_img, water_mask, water_bounding_rect, depth_img, water_area = self.segment_water()

        # Detect the pellets within the segmented water
        img_with_contour, contours, binary_res, pellet_mask = self.segment_pellets(water_only_img, water_only_hsv_img)

        # Find the start and end rows and columns of the bounding rectangle, OpenCV -> Numpy conversion has catches
        x, y, w, h = water_bounding_rect
        start_row, start_col, end_row, end_col = y, x, y + h, x + w

        depths_list = depth_img.flatten()

        # Depths less than zero are points in the image we don't care about (Not water body)
        valid_depths_list = depths_list[depths_list >= 0]

        # Split the field of view into N subregions when N=2 (split field at median) q = [0.5], when N=3
        # q = [0.33, 0.67], ...
        q = np.linspace(0, 1, N + 1, endpoint=True)[1:-1]
        quantiles = np.quantile(valid_depths_list, q).tolist()
        quantiles.insert(0, 0)
        quantiles.append(float('inf'))

        # This computes the centroids of the group of detected pellets
        contour_centroids = list(map(PelletCounter.contour_moment, contours))
        pellet_counts = 0
        total_pellet_area = 0.0
        # These following variables are used to compute the distribution of the pellets over the water body
        # Weights are needed because farther pixels contain more pellets
        weights = list()
        x_coords, y_coords = list(), list()

        # Example for two regions currently [0, median, inf]
        # For 4 regions: [0, q1, q2, q3, inf]
        # For each region (near-field, ..., far-field)
        for k in range(1, len(quantiles)):
            lb = quantiles[k - 1]
            ub = quantiles[k]

            # Use the depth image to define a mask for the region
            depth_mask = np.bitwise_and(depth_img >= lb, depth_img <= ub)

            # Find pellets that belong to the region defined by the depth mask
            depth_pellet_mask = np.bitwise_and(depth_mask, pellet_mask)

            # Discard regions outside the bounding rectangle of the segmented water
            depth_sub_mask = depth_pellet_mask[start_row:end_row, start_col:end_col]

            #
            region_weight = PelletCounter.PIXEL_AREAS[0] / PelletCounter.PIXEL_AREAS[k - 1]
            x_coords_region, y_coords_region = np.nonzero(depth_sub_mask)
            x_coords.extend(x_coords_region)
            y_coords.extend(y_coords_region)
            weights.extend([region_weight] * len(x_coords_region))

            # contours_in_region = filter(
            #     lambda centroid: PelletCounter.is_centroid_in_region(centroid, depth_mask),
            #     contour_centroids
            # )

            # Filter out pellet contours that lie within the current region
            # ---------------------------------------------------------------
            # cntrs_in_region_bool = [True, False, True, False, False]
            cntrs_in_region_bool = np.apply_along_axis(
                lambda centroid: PelletCounter.is_centroid_in_region(centroid, depth_mask),
                1,
                contour_centroids
            )

            # cntrs_in_region_idx = [0, 2]
            cntrs_in_region_idx = np.where(cntrs_in_region_bool == True)[0]

            # contours_in_region = [contour_0, contour_2]
            contours_in_region = [contours[i] for i in cntrs_in_region_idx]
            # --------------------------------------------------------

            # Compute the areas of the detections in the current region defined by depth
            # contour_areas = [area_of_contour_0, area_of_contour_2]
            contour_areas = np.array(
                list(
                    map(cv2.contourArea, contours_in_region)
                )
            )
            # Compute the count by dividing the total area by the area of one pixel
            pellet_count_in_region = sum(contour_areas) / PelletCounter.PIXEL_AREAS[k - 1]
            pellet_counts += pellet_count_in_region
            total_pellet_area += sum(contour_areas)

            region = cv2.bitwise_and(water_only_img, water_only_img, mask=depth_mask.astype(np.uint8))
            cv2.drawContours(region, contours_in_region, -1, (0, 0, 255), 1)
            if self.DEBUG:
                cv2.imwrite(f"Step_8_region_{k}.png", region)

        H, *_ = np.histogram2d(x_coords, y_coords, bins=200, weights=weights, density=True)
        plt.figure()
        plt.imshow(H)
        percent_covered = total_pellet_area / water_area

        if self.DEBUG:
            plt.savefig(f"Distribution_of_pellets.png", cmap='gray')
        return pellet_counts, H, percent_covered * 100

# TODO: Add capture from webcam
# To open a camera you can use the OpenCV function
# cv2.VideoCapture(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Detect fish feed pellets and visualize detection'
    )

    parser.add_argument(
        'img_path', type=str,
        help='path to the input image'
    )

    parser.add_argument('--hsvlp', default='1, 0, 0',
                        help='The lower bounds of the HSV for the fish pellets specified by '
                             'a string of three comma-separated integers')

    parser.add_argument('--hsvup', default='75, 255, 255',
                        help='The upper bounds of the HSV for the fish pellets specified by '
                             'a string of three comma-separated integers')

    parser.add_argument('--hsvlw', default='90, 0, 0',
                        help='The lower bounds of the HSV for the water specified by '
                             'a string of three comma-separated integers')

    parser.add_argument('--hsvuw', default='115, 255, 255',
                        help='The upper bounds of the HSV for the water specified by '
                             'a string of three comma-separated integers')

    parser.add_argument('--matrix', default='camera_params.npy',
                        help='The path to the NPY file that contains the camera matrix.')
    parser.add_argument('--debug', action='store_true',
                        help='Run application in debug mode. Writes all intermediate outputs to file.')
    args = parser.parse_args()
    os.environ['TRANSFORMERS_OFFLINE'] = '1'

    if args.img_path == '0':
        if not is_raspberrypi():
            raise OSError('Not running on a Raspberry Pi')

        camera = PiCamera()
        camera.resolution = (640, 480)
        camera.framerate = 32
        rawCapture = PiRGBArray(camera, size=(640, 480))
        camera.capture(rawCapture, format="bgr")
        image = rawCapture.array

        app = PelletCounter(args, image)
        counts, H, percentage = app.count_pellets()

    else:
        app = PelletCounter(args)
        counts, H, percentage = app.count_pellets()

    print(counts, percentage)