import os

import numpy as np
import scipy
import skimage
from histo_tools import deconvolution
from tifffile import tifffile

from histo_ECM.src import utils


class HE:
    # he stain:
    # - [x] number of cells / tissue area
    # - [x] area of cells / tissue area
    def __init__(self, slide, subsample_idx, bounding_box, savepath):
        self.filepath = slide.filepath
        self.file_name = slide.filepath.split(os.sep)[-1]
        self.subsample_num = subsample_idx
        self.bounding_box = bounding_box
        self.savepath = savepath
        self.im_rgb = None
        self.im_od = None
        self.im_tissue_mask = None
        self.tissue_area = None
        self.cell_area = None
        self.num_cells = None
        self.nuc_labels = None
        self.cells_per_tissue_area = None
        self.cell_area_per_tissue_area = None

    def get_full_rgb_im(self, full_slide):
        """Gets the full sample rgb image from the bounding boxes."""
        self.im_rgb = full_slide[
                       self.bounding_box[0][0]:self.bounding_box[0][1],
                       self.bounding_box[1][0]:self.bounding_box[1][1],
                       :]

    def get_deconv_od_im(self):
        rgb_mask = np.max(self.im_rgb, axis=2) < 230

        sum_rgb = np.sum(self.im_rgb[..., :3], axis=2)
        one_two = (self.im_rgb[..., 1].astype('float') + self.im_rgb[..., 2].astype('float')) / sum_rgb
        zero_two = (self.im_rgb[..., 0].astype('float') + self.im_rgb[..., 2].astype('float')) / sum_rgb
        zero_one = (self.im_rgb[..., 0].astype('float') + self.im_rgb[..., 1].astype('float')) / sum_rgb
        zero_two_mask = scipy.ndimage.gaussian_filter(zero_two, 1) > 0.7
        one_two_mask = scipy.ndimage.gaussian_filter(one_two, 1) < 0.67
        one_two_mask = scipy.ndimage.binary_erosion(one_two_mask, iterations=5)
        zero_one_mask = scipy.ndimage.gaussian_filter(zero_one, 1) < 0.67

        final_bg_mask = rgb_mask * one_two_mask * zero_two_mask * zero_one_mask

        self.im_od, _, _ = deconvolution.run_full(self.im_rgb, final_bg_mask)

    def get_nuclei(self):
        od2_mask = scipy.ndimage.gaussian_filter(self.im_od[..., 2], 1) < 0.1
        od0_mask = scipy.ndimage.gaussian_filter(self.im_od[..., 0], 1) < 0.2
        gauss_nuc = scipy.ndimage.gaussian_filter(self.im_od[..., 1], 1) > 0.33
        nuc_mask = od2_mask * od0_mask * gauss_nuc

        distance = scipy.ndimage.distance_transform_edt(nuc_mask)
        distance = scipy.ndimage.gaussian_filter(distance, sigma=1)
        coords = skimage.feature.peak_local_max(
            distance, footprint=np.ones((10, 10)), labels=nuc_mask
        )
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = scipy.ndimage.label(mask)
        labels_watershed = skimage.segmentation.watershed(
            -distance, markers, mask=nuc_mask, watershed_line=True
        )
        mask = labels_watershed.astype("bool")
        mask = skimage.morphology.remove_small_objects(mask, min_size=20)
        self.nuc_labels, self.num_cells = scipy.ndimage.label(mask)
        self.cell_area = np.count_nonzero(self.nuc_labels > 0)

    def get_tissue_mask(self):
        max_intensity_od = np.max(self.im_od, axis=2)
        self.im_tissue_mask = scipy.ndimage.gaussian_filter(max_intensity_od, 1) > 0.05
        self.tissue_area = np.count_nonzero(self.im_tissue_mask)

    def calculate_nuc_ratios(self):
        self.cells_per_tissue_area = self.num_cells / self.tissue_area
        self.cell_area_per_tissue_area = self.cell_area / self.tissue_area

    def add_save_clear_images(self, validation_viewer, prefix):
        full_save_path = os.path.join(self.savepath,
                                      f'{prefix}-{self.file_name}-subsample_{self.subsample_num:03}.png')
        validation_viewer.screenshot(full_save_path, scale=4, flash=False)
        while len(validation_viewer.layers) > 0:
            validation_viewer.layers.remove(validation_viewer.layers[0])

    def save_validation_images(self, validation_viewer):
        validation_viewer.add_image(self.im_rgb)
        self.add_save_clear_images(validation_viewer, 'rgb')
        if self.im_tissue_mask is not None:
            validation_viewer.add_image(self.im_rgb, opacity=0.5)
            validation_viewer.add_image(self.im_tissue_mask, colormap='green', blending='additive', opacity=0.5)
            self.add_save_clear_images(validation_viewer, 'tissue')
        if self.nuc_labels is not None:
            validation_viewer.add_image(self.im_rgb, opacity=0.5)
            validation_viewer.add_labels(self.nuc_labels, opacity=1)
            self.add_save_clear_images(validation_viewer, 'nuclei')


class HEStats:
    def __init__(self, subsample):
        self.sample_type = 'trichrome'
        self.filename = subsample.file_name,
        self.subsample_idx = subsample.subsample_num,
        self.nuc_num = subsample.num_cells,
        self.nuc_area = subsample.cell_area,
        self.tissue_area = subsample.tissue_area,
        self.nuc_num_per_tissue = subsample.cells_per_tissue_area,
        self.nuc_area_per_tissue = subsample.cell_area_per_tissue_area,


def run_he(slide, savepath, file_lock, feature_viewer=None):
    full_slide = tifffile.imread(slide.filepath, series=0, level=0)
    subsample_holder = []
    for subsample_idx, bounding_box in enumerate(slide.bounding_boxes_upscaled):
        subsample = HE(slide, subsample_idx, bounding_box, savepath)
        subsample.get_full_rgb_im(full_slide)
        subsample_holder.append(subsample)
    del full_slide  # free up some memory once we get ROIs. especially useful for multiprocessing.
    for subsample in subsample_holder:
        subsample.get_deconv_od_im()
        subsample.get_nuclei()
        subsample.get_tissue_mask()
        subsample.calculate_nuc_ratios()
        stats_to_save = HEStats(subsample)
        utils.io.write_csv(os.path.join(savepath, 'he_stats.csv'), stats_to_save, file_lock)
        if feature_viewer is not None:
            subsample.save_validation_images(feature_viewer)
