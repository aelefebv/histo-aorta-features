import os

import numpy as np
import scipy
import skimage
from histo_tools import deconvolution
from tifffile import tifffile

from src import utils, stains


class Trichrome:
    # tc stain:
    # - [x] collagen density in aortic wall
    # - [x] aortic wall thickness
    # - [x] thickness of external collagen lining
    # - [x] perimeter of aorta interior
    # - [x] perimeter of aorta exterior
    def __init__(self, slide, subsample_idx, bounding_box, savepath):
        self.filepath = slide.filepath
        self.file_name = slide.filepath.split(os.sep)[-1]
        self.subsample_num = subsample_idx
        self.bounding_box = bounding_box
        self.savepath = savepath
        self.im_rgb = None
        self.im_od = None
        self.im_collagen_mask = None
        self.im_aorta_mask = None
        self.im_collagen_in_aortic_wall_mask = None
        self.im_collagen_outside_aortic_wall_mask = None
        self.im_aorta_thickness_map = None
        self.im_collagen_thickness_map = None
        self.im_aorta_exterior = None
        self.im_aorta_interior = None
        self.collagen_in_aortic_wall_fraction = None
        self.aorta_exterior_perimeter = None
        self.aorta_interior_perimeter = None
        self.thickness_wall_stats = None
        self.thickness_ext_stats = None

    def get_deconv_od_im(self):
        self.im_od, _, _ = deconvolution.run_full(self.im_rgb, np.max(self.im_rgb, axis=2) < 230)

    def get_full_rgb_im(self, full_slide):
        """Gets the full sample rgb image from the bounding boxes."""
        self.im_rgb = full_slide[
                       self.bounding_box[0][0]:self.bounding_box[0][1],
                       self.bounding_box[1][0]:self.bounding_box[1][1],
                       :]

    def get_collagen_mask(self):
        self.im_collagen_mask = scipy.ndimage.gaussian_filter(self.im_od[..., 1], 1) > 0.2

    def get_aorta_wall_mask(self):
        sum_od = np.sum(self.im_od[..., :3], axis=2)
        od_mask = scipy.ndimage.gaussian_filter(sum_od, 3) > 0.3
        od_mask = skimage.morphology.remove_small_holes(od_mask, 5000)
        od_mask = skimage.morphology.remove_small_objects(od_mask, 5000)
        tissue = self.im_od[..., 0]
        tissue_mask = scipy.ndimage.gaussian_filter(tissue, 2) > 0.1
        tissue_mask = scipy.ndimage.binary_closing(tissue_mask, iterations=5)
        tissue_mask = skimage.morphology.remove_small_holes(tissue_mask, 5000)
        tissue_mask = skimage.morphology.remove_small_objects(tissue_mask, 5000)
        tissue_mask *= od_mask
        self.im_aorta_mask = tissue_mask

    def get_collagen_in_aortic_wall(self):
        self.im_collagen_in_aortic_wall_mask = self.im_aorta_mask * self.im_collagen_mask
        self.im_collagen_outside_aortic_wall_mask = ~self.im_aorta_mask * self.im_collagen_mask
        collagen_in_aortic_wall_px = np.count_nonzero(self.im_aorta_mask * self.im_collagen_mask)
        aortic_wall_area_px = np.count_nonzero(self.im_aorta_mask)
        if aortic_wall_area_px > 0:
            self.collagen_in_aortic_wall_fraction = collagen_in_aortic_wall_px / aortic_wall_area_px

    def get_collagen_thickness_around_aortic_wall(self):
        prelim_mask = ~ self.im_aorta_mask * (self.im_od[..., 1] > 0.1)
        collagen_outside_wall = scipy.ndimage.gaussian_filter(prelim_mask.astype('float'), 30) > 0.1
        collagen_distance = scipy.ndimage.distance_transform_edt(collagen_outside_wall)
        collagen_skeleton = skimage.morphology.skeletonize(collagen_outside_wall)
        self.im_collagen_thickness_map = collagen_distance * collagen_skeleton

    def get_aorta_thickness(self):
        distance = scipy.ndimage.distance_transform_edt(self.im_aorta_mask)
        skeleton = skimage.morphology.skeletonize(
            scipy.ndimage.gaussian_filter(self.im_aorta_mask.astype('float'), 30) > 0.8
        )
        self.im_aorta_thickness_map = distance * skeleton

    def get_aortic_perimeter(self):
        aortic_labels, _ = scipy.ndimage.label(self.im_aorta_mask)
        objects = skimage.measure.regionprops(aortic_labels)
        max_region_area = 0
        largest_region = None
        for region in objects:
            if region.area > max_region_area:
                largest_region = region
                max_region_area = region.area
        if largest_region is None:
            return
        region_dist = scipy.ndimage.distance_transform_edt(aortic_labels == largest_region.label)
        borders = region_dist == 1
        border_labels, _ = scipy.ndimage.label(borders, structure=np.ones((3, 3)))
        border_regions = skimage.measure.regionprops(border_labels)
        region_areas = []
        for region in border_regions:
            region_areas.append(region.area)
        argsorted_regions = np.argsort(region_areas)
        outside_border = border_regions[argsorted_regions[-1]]
        outside_perimeter = outside_border.area
        self.aorta_exterior_perimeter = outside_perimeter
        self.im_aorta_exterior = np.zeros_like(self.im_aorta_mask)
        self.im_aorta_exterior[border_labels == outside_border.label] = 1
        self.im_aorta_interior = np.zeros_like(self.im_aorta_mask)
        if len(border_regions) > 1:
            inside_border = border_regions[argsorted_regions[-2]]
            inside_perimeter = inside_border.area
            if inside_perimeter > (outside_perimeter / 10):
                self.aorta_interior_perimeter = inside_perimeter
            self.im_aorta_interior[border_labels == inside_border.label] = 1

    def add_save_clear_images(self, validation_viewer, add_im, contrast_limits=(0, 255), prefix='rgb',
                              colormap='turbo', opacity=0.5):
        if add_im is None:
            return
        elif np.array_equal(add_im, self.im_rgb):
            validation_viewer.add_image(self.im_rgb)
        else:
            validation_viewer.add_image(self.im_rgb, opacity=0.5)
            validation_viewer.add_image(add_im, colormap=colormap, blending='additive',
                                        contrast_limits=contrast_limits, opacity=opacity)
        full_save_path = os.path.join(self.savepath,
                                      f'{prefix}-{self.file_name}-subsample_{self.subsample_num:03}.png')
        validation_viewer.screenshot(full_save_path, scale=4, flash=False)
        while len(validation_viewer.layers) > 0:
            validation_viewer.layers.remove(validation_viewer.layers[0])

    def save_validation_images(self, validation_viewer):
        self.add_save_clear_images(validation_viewer, self.im_rgb)
        self.add_save_clear_images(validation_viewer, self.im_collagen_in_aortic_wall_mask,
                                   contrast_limits=(0, 1), prefix='aorta_collagen', opacity=0.5, colormap='green')
        self.add_save_clear_images(validation_viewer, self.im_aorta_thickness_map,
                                   contrast_limits=(1, 300), prefix='aorta_thickness', opacity=1)
        self.add_save_clear_images(validation_viewer, self.im_collagen_thickness_map,
                                   contrast_limits=(1, 300), prefix='collagen_thickness', opacity=1)

    def get_array_stats(self):
        self.thickness_wall_stats = stains.general.StatsHolder(self.im_aorta_thickness_map)
        self.thickness_ext_stats = stains.general.StatsHolder(self.im_collagen_thickness_map)


class TrichromeStats:
    def __init__(self, subsample):
        self.sample_type = 'trichrome'
        self.filename = subsample.file_name,
        self.subsample_idx = subsample.subsample_num,
        self.collagen_density_in_wall = subsample.collagen_in_aortic_wall_fraction,
        self.interior_perimeter = subsample.aorta_interior_perimeter,
        self.exterior_perimeter = subsample.aorta_exterior_perimeter,
        self.thickness_wall_mean = subsample.thickness_wall_stats.mean,
        self.thickness_wall_sd = subsample.thickness_wall_stats.sd,
        self.thickness_wall_sem = subsample.thickness_wall_stats.sem,
        self.thickness_wall_median = subsample.thickness_wall_stats.median,
        self.thickness_wall_q25 = subsample.thickness_wall_stats.q25,
        self.thickness_wall_q75 = subsample.thickness_wall_stats.q75,
        self.thickness_wall_min = subsample.thickness_wall_stats.min,
        self.thickness_wall_max = subsample.thickness_wall_stats.max,
        self.thickness_wall_sum = subsample.thickness_wall_stats.sum,
        self.thickness_wall_cov = subsample.thickness_wall_stats.cov,
        self.thickness_wall_skew = subsample.thickness_wall_stats.skew,
        self.thickness_wall_geo_mean = subsample.thickness_wall_stats.geo_mean,
        self.thickness_wall_geo_std = subsample.thickness_wall_stats.geo_std,
        self.thickness_ext_mean = subsample.thickness_ext_stats.mean,
        self.thickness_ext_sd = subsample.thickness_ext_stats.sd,
        self.thickness_ext_sem = subsample.thickness_ext_stats.sem,
        self.thickness_ext_median = subsample.thickness_ext_stats.median,
        self.thickness_ext_q25 = subsample.thickness_ext_stats.q25,
        self.thickness_ext_q75 = subsample.thickness_ext_stats.q75,
        self.thickness_ext_min = subsample.thickness_ext_stats.min,
        self.thickness_ext_max = subsample.thickness_ext_stats.max,
        self.thickness_ext_sum = subsample.thickness_ext_stats.sum,
        self.thickness_ext_cov = subsample.thickness_ext_stats.cov,
        self.thickness_ext_skew = subsample.thickness_ext_stats.skew,
        self.thickness_ext_geo_mean = subsample.thickness_ext_stats.geo_mean,
        self.thickness_ext_geo_std = subsample.thickness_ext_stats.geo_std


def run_trichrome(slide, savepath, file_lock, feature_viewer=None):
    full_slide = tifffile.imread(slide.filepath, series=0, level=0)
    subsample_holder = []
    for subsample_idx, bounding_box in enumerate(slide.bounding_boxes_upscaled):
        subsample = Trichrome(slide, subsample_idx, bounding_box, savepath)
        subsample.get_full_rgb_im(full_slide)
        subsample_holder.append(subsample)
    del full_slide  # free up some memory once we get ROIs. especially useful for multiprocessing.
    for subsample in subsample_holder:
        subsample.get_deconv_od_im()
        subsample.get_collagen_mask()
        subsample.get_aorta_wall_mask()
        subsample.get_collagen_in_aortic_wall()
        subsample.get_collagen_thickness_around_aortic_wall()
        subsample.get_aorta_thickness()
        subsample.get_aortic_perimeter()
        subsample.get_array_stats()
        stats_to_save = TrichromeStats(subsample)
        utils.io.write_csv(os.path.join(savepath, 'trichrome_stats.csv'), stats_to_save, file_lock)
        if feature_viewer is not None:
            subsample.save_validation_images(feature_viewer)
