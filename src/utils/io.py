import csv
import glob
import os
import os.path
from datetime import datetime

import numpy as np


def get_histo_filelist(directory):
    """Checks the directory for and gets all files that end in .ndpi and .svs."""
    file_list = [os.path.join(directory, f) for f in os.listdir(directory)]
    file_list = [f for f in file_list if f.endswith('.ndpi') or f.endswith('.svs')]
    return file_list


def create_output_directory(directory):
    """Given a top directory, create and return the location of the output directory with a datetime."""
    dt = datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = os.path.join(directory, 'output', dt)
    os.makedirs(save_path)
    return save_path


def save_subsample_validation_image(slide, subsample_viewer, save_path):
    """Given a slide object, """
    napari_bboxes = []
    label_num = []
    for label_idx, bounding_box in enumerate(slide.bounding_boxes):
        label_num.append(label_idx)
        bbox_rect = np.array(
            [[bounding_box[0][0], bounding_box[1][0]],
             [bounding_box[0][1], bounding_box[1][0]],
             [bounding_box[0][1], bounding_box[1][1]],
             [bounding_box[0][0], bounding_box[1][1]]]
        )
        napari_bboxes.append(bbox_rect)
    text_parameters = {
        'string': '{label}',
        'size': 20,
        'color': 'green',
        'anchor': 'upper_left',
        'translation': [-3, 0]
    }
    properties = {'label': label_num}
    subsample_viewer.add_image(slide.im_rgb_downsample)
    subsample_viewer.add_labels(slide.roi_labels)
    subsample_viewer.add_shapes(napari_bboxes, face_color='transparent',
                                edge_color='green',
                                name='bounding box',
                                properties=properties,
                                text=text_parameters,
                                )
    full_save_path = os.path.join(save_path, f'validation-{slide.file_name}.png')
    subsample_viewer.screenshot(full_save_path, scale=4, flash=False)
    while len(subsample_viewer.layers) > 0:
        subsample_viewer.layers.remove(subsample_viewer.layers[0])


def write_csv(csv_path, stats_to_save, file_lock):
    # get the list of attributes
    attributes = vars(stats_to_save)
    attr_names = list(attributes.keys())
    attr_values = []
    for value in list(attributes.values()):
        if isinstance(value, (list, tuple)):
            attr_values.append(value[0])
        else:
            attr_values.append(value)

    # Check if the file exists
    if not os.path.exists(csv_path):
        # Acquire the lock
        with file_lock:
            # Open the file for writing
            with open(csv_path, 'w', newline='') as csvfile:
                # Create a csv.writer object
                writer = csv.writer(csvfile, delimiter=',')
                # Write the header row
                writer.writerow(attr_names)

    # Acquire the lock
    with file_lock:
        # Open the file for appending
        with open(csv_path, 'a', newline='') as csvfile:
            # Create a csv.writer object
            writer = csv.writer(csvfile, delimiter=',')
            # Write each object to a row
            writer.writerow(attr_values)
