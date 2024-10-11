import os

import numpy as np
import scipy
import skimage
import tifffile
from histo_tools import deconvolution
from src import stains, utils


class Pixel:
    """a network is composed of pixels"""
    def __init__(self, coord):
        self.coord = coord
        self.connected_pixels = []
        self.type = 0
        self.tortuosity = np.nan

    def __eq__(self, other):
        # if it's a pixel, check if the coords are equal, otherwise, "other" should just be coordinates
        if isinstance(other, self.__class__):
            return np.array_equiv(self.coord, other.coord)
        return np.array_equiv(self.coord, other)

    def undulation_depth_first(self, total_distance, search_rad, current_dir_ends, visited_pixels):
        if total_distance >= search_rad:
            current_dir_ends.append(self)
            return
        for connection in self.connected_pixels:
            if connection in visited_pixels:
                continue
            visited_pixels.append(connection)
            dist = np.linalg.norm(connection.coord - self.coord)
            total_distance += dist
            connection.undulation_depth_first(total_distance, search_rad, current_dir_ends, visited_pixels)

    def calculate_tortuosity(self, search_rad):  # only get tortuosity for pixels with len connected pixels == 2
        visited_pixels = [self]
        end_points = []
        for connection in self.connected_pixels:
            visited_pixels.append(connection)
            current_dir_ends = []
            total_distance = 0
            dist = np.linalg.norm(connection.coord - self.coord)
            total_distance += dist
            connection.undulation_depth_first(total_distance, search_rad, current_dir_ends, visited_pixels)
            end_points.append(current_dir_ends)
        dirs_x = []
        dirs_y = []
        for dirs in end_points:
            dir_x = [endpoint.coord[0] for endpoint in dirs]
            dir_y = [endpoint.coord[1] for endpoint in dirs]
            dirs_x.append(dir_x)
            dirs_y.append(dir_y)
        tortuosity_matrix = np.sqrt(
            (np.array(dirs_x[0]) - np.array(dirs_x[1])[None, ...].T) ** 2 +
            (np.array(dirs_y[0]) - np.array(dirs_y[1])[None, ...].T) ** 2
        )
        if tortuosity_matrix.size > 0:
            min_tort = np.max([1.0, (search_rad * 2 / np.mean(tortuosity_matrix, axis=None))])
        else:
            min_tort = np.nan
        self.tortuosity = min_tort


class Network:
    def __init__(self, skeleton):
        self.im_skeleton = skeleton
        self.im_network = None  # save turbo [0,4] additive overlaid on rgb at lower opacity
        self.im_tortuosity = None
        self.skeleton_coords = np.array(np.where(self.im_skeleton)).T
        self.pixels = []
        self.total_segments = 0
        self.num_trees = 0
        self.num_branch_points = 0
        self.num_tip_points = 0
        self.complexity = None
        self.network_len = len(self.skeleton_coords)
        self.segment_to_network_ratio = 0  # a lower number here means more breaks

    def get_pixels(self):
        for coord in self.skeleton_coords:
            self.pixels.append(Pixel(coord))

    def get_connection_mask(self):
        num_pixels, num_dims = self.skeleton_coords.shape
        loc_mask = np.ones((num_pixels, num_pixels), dtype='bool')
        np.fill_diagonal(loc_mask, False)
        for dim in range(num_dims):
            loc_array = np.asarray([[pixel.coord[dim] for pixel in self.pixels]])
            loc_mask_temp = np.abs(loc_array - loc_array.transpose())
            loc_mask[loc_mask_temp > 1] = False
        match_a, match_b = np.where(loc_mask)
        for match in range(len(match_a)):
            pixel_a = int(match_a[match])
            pixel_b = int(match_b[match])
            self.pixels[pixel_a].connected_pixels.append(self.pixels[pixel_b])

    def get_connection_mask_2(self):
        coord_list = self.skeleton_coords.tolist()
        for px_idx, pixel in enumerate(self.pixels):
            x_start = pixel.coord[0] - 1
            x_end = pixel.coord[0] + 2
            y_start = pixel.coord[1] - 1
            y_end = pixel.coord[1] + 2
            check_coords = [
                [x, y] for x in range(x_start, x_end) for y in range(y_start, y_end)
            ]
            for coord in check_coords:
                if (coord in coord_list) and not np.array_equiv(pixel.coord, coord):
                    match = coord_list.index(coord)
                    pixel.connected_pixels.append(self.pixels[match])

    def set_type(self):
        for pixel in self.pixels:
            pixel.type = min(len(pixel.connected_pixels), 3)

    def clean_branch_points(self):
        for pixel in self.pixels:
            if pixel.type != 3:
                continue
            for connection in pixel.connected_pixels:
                if connection.type == 3:
                    pixel.type = 2

    def get_network_im(self):
        self.im_network = np.zeros_like(self.im_skeleton, dtype='uint8')
        for pixel in self.pixels:
            self.im_network[tuple(pixel.coord)] = pixel.type

    def get_network_stats(self):
        self.num_branch_points = 0
        self.num_tip_points = 0
        _, self.num_trees = scipy.ndimage.label(self.im_skeleton, structure=np.ones((3, 3)))
        for pixel in self.pixels:
            if pixel.type == 3:
                self.num_branch_points += 1
            if pixel.type == 1:
                self.num_tip_points += 1
        self.total_segments = self.num_branch_points + self.num_trees
        try:
            self.segment_to_network_ratio = self.total_segments / self.network_len
        except ZeroDivisionError:
            self.segment_to_network_ratio = np.nan
        self.complexity = self.total_segments + self.num_tip_points

    def get_tortuosity_im(self):
        self.im_tortuosity = np.zeros_like(self.im_skeleton, dtype='float')
        for pixel in self.pixels:
            if np.isnan(pixel.tortuosity):
                self.im_tortuosity[tuple(pixel.coord)] = 0
            else:
                self.im_tortuosity[tuple(pixel.coord)] = pixel.tortuosity

    def get_tortuosity(self, search_rad=16):
        """waviness of lamina"""
        for px_idx, pixel in enumerate(self.pixels):
            if len(pixel.connected_pixels) == 2:
                pixel.calculate_tortuosity(search_rad)
        self.get_tortuosity_im()

    def build_network(self):
        self.get_pixels()
        try:  # way faster, but can make a very large array
            self.get_connection_mask()
        except MemoryError:  # slower, but won't overload
            self.get_connection_mask_2()
        self.set_type()
        self.clean_branch_points()
        self.get_network_im()
        self.get_network_stats()
        self.get_tortuosity(16)
        self.get_tortuosity_im()


class VVG:
    # - [x] thickness of elastic laminae
    # - [x] average number of layers of lamina
    # - [x] continuity vs breaks of elastic lamina
    # - [x] waviness of lamina
    def __init__(self, slide, subsample_num, bounding_box, savepath):
        self.filepath = slide.filepath
        self.file_name = slide.filepath.split(os.sep)[-1]
        self.subsample_num = subsample_num
        self.bounding_box = bounding_box
        self.savepath = savepath
        self.network = None
        self.im_rgb = None
        self.im_od = None
        self.elastin_layers = None
        self.im_tissue_mask = None
        self.im_thickness_map = None
        self.im_skeleton = None
        self.im_elastin_mask = None
        self.tortuosity_stats = None
        self.thickness_stats = None

    def get_full_rgb_im(self, full_slide):
        """Gets the full sample rgb image from the bounding boxes."""
        self.im_rgb = full_slide[
                       self.bounding_box[0][0]:self.bounding_box[0][1],
                       self.bounding_box[1][0]:self.bounding_box[1][1],
                       :]

    def get_od_deconvolved(self):
        """Gets the OD deconvolved image and avoids any badly stained regions."""
        rgb_mask = np.max(self.im_rgb, axis=2) < 230
        sum_rgb = np.sum(self.im_rgb[..., :3], axis=2)
        zero_two = (self.im_rgb[..., 0].astype('float') + self.im_rgb[..., 2].astype('float')) / sum_rgb
        one_two = (self.im_rgb[..., 1].astype('float') + self.im_rgb[..., 2].astype('float')) / sum_rgb
        zero_two_mask = scipy.ndimage.gaussian_filter(zero_two, 1) > 0.67
        one_two_mask = scipy.ndimage.gaussian_filter(one_two, 1) < 0.75
        final_bg_mask = rgb_mask * one_two_mask * zero_two_mask
        self.im_od, _, _ = deconvolution.run_full(self.im_rgb, final_bg_mask)

    def get_elastin_mask(self):
        mask = self.im_od[..., 1] > 0.7
        self.im_elastin_mask = skimage.morphology.remove_small_objects(mask, 100)

    def get_skeleton(self):
        self.im_skeleton = skimage.morphology.skeletonize(self.im_elastin_mask)
        self.network = Network(self.im_skeleton)

    def get_elastin_thickness(self):
        """Thickness is measured by the distance transform's value at the skeleton pixels."""
        im_distance = scipy.ndimage.distance_transform_edt(self.im_elastin_mask)
        self.im_thickness_map = im_distance * self.im_skeleton

    def get_tissue_mask(self):
        max_proj = np.max(self.im_od[..., 0:2], axis=2)
        mask = scipy.ndimage.gaussian_filter(max_proj, 1) > 0.05
        mask = skimage.morphology.remove_small_objects(mask, 10000)
        mask = scipy.ndimage.uniform_filter(mask.astype('float'), size=30) > 0.1
        self.im_tissue_mask = mask

    def get_num_elastin_layers(self):
        """
        get number of layers in the elastic lamina for each object in the frame.
        goes through each row, finds pieces of tissue, then counts number of lamina pieces in that tissue piece
        does the same for col, then gets the minimum of median of row and col.
        """
        labels_in_frame, _ = scipy.ndimage.label(self.im_tissue_mask)
        objects_in_frame = skimage.measure.regionprops(labels_in_frame, self.im_elastin_mask)
        for obj in objects_in_frame:
            num_rows, num_cols = obj.image_intensity.shape

            col_list = []
            for col in range(0, num_cols):
                chunks, num_chunks = scipy.ndimage.label(obj.image[:, col])
                for chunk_num in range(1, num_chunks + 1):
                    int_slice = obj.image_intensity[:, col]
                    _, num_slices = scipy.ndimage.label(int_slice[chunks == chunk_num])
                    col_list.append(num_slices)

            row_list = []
            for row in range(0, num_rows):
                chunks, num_chunks = scipy.ndimage.label(obj.image[row, :])
                for chunk_num in range(1, num_chunks + 1):
                    int_slice = obj.image_intensity[row, :]
                    _, num_slices = scipy.ndimage.label(int_slice[chunks == chunk_num])
                    row_list.append(num_slices)

            num_layers = min(np.median(row_list), np.median(col_list))

            if self.elastin_layers is None:
                self.elastin_layers = []
            if num_layers != 0:
                self.elastin_layers.append(num_layers)

        if not self.elastin_layers:  # if no layers were found anywhere, set it to 0
            self.elastin_layers = [0]

    def get_elastin_network(self):
        self.network.build_network()

    def add_save_clear_images(self, validation_viewer, add_im, contrast_limits=(0, 255), prefix='rgb'):
        if add_im is None:
            return
        elif np.array_equal(add_im, self.im_rgb):
            validation_viewer.add_image(self.im_rgb)
        else:
            validation_viewer.add_image(self.im_rgb, opacity=0.5)
            validation_viewer.add_image(add_im, colormap='turbo', blending='additive',
                                        contrast_limits=contrast_limits)
        full_save_path = os.path.join(self.savepath,
                                      f'{prefix}-{self.file_name}-subsample_{self.subsample_num:03}.png')
        validation_viewer.screenshot(full_save_path, scale=4, flash=False)
        while len(validation_viewer.layers) > 0:
            validation_viewer.layers.remove(validation_viewer.layers[0])

    def save_validation_images(self, validation_viewer):
        self.add_save_clear_images(validation_viewer, self.im_rgb)
        self.add_save_clear_images(validation_viewer, self.im_thickness_map,
                                   contrast_limits=(0, 10), prefix='elastin_thickness')
        self.add_save_clear_images(validation_viewer, self.network.im_tortuosity,
                                   contrast_limits=(1, 3), prefix='elastin_tortuosity')
        self.add_save_clear_images(validation_viewer, self.network.im_network,
                                   contrast_limits=(0, 4), prefix='elastin_network')

    def get_array_stats(self):
        self.thickness_stats = stains.general.StatsHolder(self.im_thickness_map)
        self.tortuosity_stats = stains.general.StatsHolder(self.network.im_tortuosity)


class VVGStats:
    def __init__(self, subsample):
        self.sample_type = 'vvg',
        self.filename = subsample.file_name,
        self.subsample_idx = subsample.subsample_num,
        self.num_layers = subsample.elastin_layers[0],
        self.complexity = subsample.network.complexity,
        self.segment_network_ratio = subsample.network.segment_to_network_ratio,
        self.thickness_mean = subsample.thickness_stats.mean,
        self.thickness_sd = subsample.thickness_stats.sd,
        self.thickness_sem = subsample.thickness_stats.sem,
        self.thickness_median = subsample.thickness_stats.median,
        self.thickness_q25 = subsample.thickness_stats.q25,
        self.thickness_q75 = subsample.thickness_stats.q75,
        self.thickness_min = subsample.thickness_stats.min,
        self.thickness_max = subsample.thickness_stats.max,
        self.thickness_sum = subsample.thickness_stats.sum,
        self.thickness_cov = subsample.thickness_stats.cov,
        self.thickness_skew = subsample.thickness_stats.skew,
        self.thickness_geo_mean = subsample.thickness_stats.geo_mean,
        self.thickness_geo_std = subsample.thickness_stats.geo_std,
        self.tortuosity_mean = subsample.tortuosity_stats.mean,
        self.tortuosity_sd = subsample.tortuosity_stats.sd,
        self.tortuosity_sem = subsample.tortuosity_stats.sem,
        self.tortuosity_median = subsample.tortuosity_stats.median,
        self.tortuosity_q25 = subsample.tortuosity_stats.q25,
        self.tortuosity_q75 = subsample.tortuosity_stats.q75,
        self.tortuosity_min = subsample.tortuosity_stats.min,
        self.tortuosity_max = subsample.tortuosity_stats.max,
        self.tortuosity_sum = subsample.tortuosity_stats.sum,
        self.tortuosity_cov = subsample.tortuosity_stats.cov,
        self.tortuosity_skew = subsample.tortuosity_stats.skew,
        self.tortuosity_geo_mean = subsample.tortuosity_stats.geo_mean,
        self.tortuosity_geo_std = subsample.tortuosity_stats.geo_std,


def run_vvg(slide, savepath, file_lock, feature_viewer=None):
    full_slide = tifffile.imread(slide.filepath, series=0, level=0)
    subsample_holder = []
    for subsample_idx, bounding_box in enumerate(slide.bounding_boxes_upscaled):
        subsample = VVG(slide, subsample_idx, bounding_box, savepath)
        subsample.get_full_rgb_im(full_slide)
        subsample_holder.append(subsample)
    del full_slide  # free up some memory once we get ROIs. especially useful for multiprocessing.
    for subsample in subsample_holder:
        subsample.get_od_deconvolved()
        subsample.get_elastin_mask()
        subsample.get_skeleton()
        subsample.get_elastin_thickness()
        subsample.get_tissue_mask()
        subsample.get_num_elastin_layers()
        subsample.get_elastin_network()
        subsample.get_array_stats()
        stats_to_save = VVGStats(subsample)
        utils.io.write_csv(os.path.join(savepath, 'vvg_stats.csv'), stats_to_save, file_lock)
        if feature_viewer is not None:
            subsample.save_validation_images(feature_viewer)
