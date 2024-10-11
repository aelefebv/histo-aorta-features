import os

import numpy as np
import skimage.morphology
import tifffile
from histo_tools import deconvolution
import scipy


class Slide:
    def __init__(self, filepath):
        self.filepath = filepath
        self.file_name = filepath.split(os.sep)[-1][:-5]
        self.im_rgb_downsample = None
        self.roi_mask = None
        self.roi_labels = None
        self.bounding_boxes = None
        self.bounding_boxes_upscaled = None
        self.downsample_level = None
        self.downsample_scaling = None

    def get_downsample_level(self, downsample_level):
        """Given a specified level, will configure slide and downsample from that level."""
        self.downsample_level = downsample_level
        self.im_rgb_downsample = tifffile.imread(self.filepath, series=0, level=downsample_level)
        self._get_downsample_scale()

    def get_downsample_multiplier(self, downsample_multiplier=16):
        """Given a specified downsample multiplier of the original scale,
        will configure slide and downsample from that multiplier."""
        downsample_level = None
        with tifffile.TiffFile(self.filepath) as tif:
            level0_len = tif.series[0].levels[0].shape[0]
            for level in range(len(tif.series[0].levels)):
                downsample_scale = tif.series[0].levels[level].shape[0]
                scale_check = int(level0_len / downsample_scale)
                if scale_check == downsample_multiplier:
                    downsample_level = level
                    break
        if downsample_level is None:  # if no downsample available, make one
            print(f'[WARNING] Creating a downsample of scale {downsample_multiplier} for {self.filepath}')
            scaling_factor = scale_check / downsample_multiplier
            smallest_level_im = tifffile.imread(self.filepath, series=0, level=level)
            self.im_rgb_downsample = scipy.ndimage.zoom(smallest_level_im, [scaling_factor, scaling_factor, 1])
            self.downsample_scaling = downsample_multiplier
        else:
            self.get_downsample_level(downsample_level)

    def _get_downsample_scale(self):
        """Gets the downsample scaling factor."""
        with tifffile.TiffFile(self.filepath) as tif:
            level0_len = tif.series[0].levels[0].shape[0]
            level2_len = tif.series[0].levels[self.downsample_level].shape[0]
        self.downsample_scaling = level0_len/level2_len

    def _get_bg_mask(self, low_thresh=220, high_thresh=245):
        """Gets the bg mask for OD image."""
        bg_mask = np.median(self.im_rgb_downsample, axis=2)
        bg_mask = scipy.ndimage.minimum_filter(bg_mask, 10)
        bg_mask = (bg_mask > low_thresh) * (bg_mask < high_thresh)
        return bg_mask

    def _get_od_im(self):
        """Gets the OD image."""
        ch_i0 = deconvolution.get_i0(self.im_rgb_downsample, self._get_bg_mask())
        od_im = deconvolution.get_od(self.im_rgb_downsample, ch_i0)
        od_im[np.isinf(od_im)] = 0
        od_im[np.isnan(od_im)] = 0
        return od_im

    def get_roi_masks(self, od_threshold=0.1, object_size=None):
        """Gets the roi masks from the OD image."""
        if object_size is None:
            object_size = self.downsample_scaling/16 * 1000
        od_im = self._get_od_im()
        od_im = np.max(od_im, axis=2)
        mask = scipy.ndimage.gaussian_filter(od_im, 3) > od_threshold
        mask = scipy.ndimage.binary_fill_holes(mask)
        mask = scipy.ndimage.binary_dilation(mask, iterations=5)
        self.roi_mask = skimage.morphology.remove_small_objects(mask, object_size)

    def get_region_labels_and_bbox(self):
        """Gets ROI labels and bounding boxes (down and upscaled)."""
        label_im, _ = scipy.ndimage.label(self.roi_mask)
        objects = scipy.ndimage.find_objects(label_im)
        region_bounding_boxes = []
        region_bounding_boxes_upscaled = []
        for obj in objects:
            region_bounding_boxes.append(
                ((int(obj[0].start), int(obj[0].stop)),
                 (int(obj[1].start), int(obj[1].stop)))
            )
            region_bounding_boxes_upscaled.append(
                ((int(obj[0].start * self.downsample_scaling), int(obj[0].stop * self.downsample_scaling)),
                 (int(obj[1].start * self.downsample_scaling), int(obj[1].stop * self.downsample_scaling)))
            )
        self.roi_labels = label_im
        self.bounding_boxes = region_bounding_boxes
        self.bounding_boxes_upscaled = region_bounding_boxes_upscaled


def get_downsample_rois(im_path):
    """Given a path to a full sized rgb slide, returns a down-sampled version with ROIs saved."""
    slide = Slide(im_path)
    slide.get_downsample_multiplier(16)
    slide.get_roi_masks()
    slide.get_region_labels_and_bbox()
    return slide
