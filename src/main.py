from src.utils.multiprocess_tools import prep_analysis


def main(file_dir, run_type, multiproc=1):
    prep_analysis(file_dir, filetype=run_type, multiproc=multiproc)


if __name__ == "__main__":
    filelist_type = [
        (r"C:\Users\austin\test_files\P22080A phys TC", 'TC'),
        (r"C:\Users\austin\test_files\P22080B phys HE", 'HE'),
        (r"C:\Users\austin\test_files\P22080A phys HE2", 'HE'),
        (r"C:\Users\austin\test_files\P22080A VVG", 'VVG')
    ]
    for file_dir, run_type in filelist_type:
        main(file_dir, filetype=run_type, multiproc=1)

    # prep_analysis(r"C:\Users\austin\test_files\P22080B phys HE", filetype='HE', multiproc=1)
