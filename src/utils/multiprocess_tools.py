import multiprocessing

import warnings
import napari

from src.stains import vvg, trichrome, he, rois
from src.utils.io import get_histo_filelist, create_output_directory, save_subsample_validation_image

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

save_roi_validation = False
save_feature_validation = False


class FileInfo:
    def __init__(self, filepath, savepath, filetype):
        self.filepath = filepath
        self.savepath = savepath
        self.filetype = filetype


def run_analysis(fileinfo, lock):
    print(fileinfo.filepath)
    if fileinfo.filetype not in ['VVG', 'TC', 'HE']:
        print(f'[WARNING] No run sequence found for filetype "{fileinfo.filetype}", returning.')
        return

    subsample_viewer = None
    if save_roi_validation:
        subsample_viewer = napari.Viewer()
    feature_viewer = None
    if save_feature_validation:
        feature_viewer = napari.Viewer()

    # get a downsample slide and save a validation of its ROIs
    slide = rois.get_downsample_rois(fileinfo.filepath)
    if subsample_viewer is not None:
        save_subsample_validation_image(slide, subsample_viewer, fileinfo.savepath)

    if fileinfo.filetype == 'VVG':
        vvg.run_vvg(slide, fileinfo.savepath, lock, feature_viewer)
    elif fileinfo.filetype == 'TC':
        trichrome.run_trichrome(slide, fileinfo.savepath, lock, feature_viewer)
    elif fileinfo.filetype == 'HE':
        he.run_he(slide, fileinfo.savepath, lock, feature_viewer)

    if subsample_viewer is not None:
        subsample_viewer.close()
    if feature_viewer is not None:
        feature_viewer.close()
    return None


# stain agnostic:
def prep_analysis(directory, filetype=None, multiproc=0):
    filelist = get_histo_filelist(directory)  # get list of ndpi or svs files
    # filelist = filelist[:2]  # for testing
    savepath = create_output_directory(directory)  # create output directory in top dir/output/dt
    all_files = []
    for filepath in filelist:
        all_files.append(FileInfo(filepath, savepath, filetype))

    if multiproc:
        if save_roi_validation or save_feature_validation:
            num_workers = min(multiprocessing.cpu_count(), 20)
        else:
            num_workers = min(multiprocessing.cpu_count(), 50)
        with multiprocessing.Manager() as manager:
            lock = manager.Lock()
            file_and_lock = [(file_info, lock) for file_info in all_files]
            pool = multiprocessing.Pool(num_workers)
            pool.starmap(run_analysis, file_and_lock)
            pool.close()
            pool.join()
    else:
        lock = multiprocessing.Lock()
        for file_info in all_files:
            run_analysis(file_info, lock)

    return savepath