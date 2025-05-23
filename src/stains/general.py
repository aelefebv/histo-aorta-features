import numpy as np
import scipy


class StatsHolder:
    def __init__(self, stat_array):
        self.mean = None
        self.sd = None
        self.sem = None
        self.median = None
        self.q25 = None
        self.q75 = None
        self.min = None
        self.max = None
        self.sum = None
        self.cov = None
        self.skew = None
        self.geo_mean = None
        self.geo_std = None
        if (stat_array is not None) and (len(stat_array) > 0):
            try:
                stat_array = stat_array[stat_array > 0]
                self.get_stats(stat_array)
            except ValueError:
                return

    def get_stats(self, stat_array):
        self.mean = np.nanmean(stat_array)
        self.sd = np.nanstd(stat_array)
        self.sem = self.sd / np.sqrt(len(stat_array))
        self.median = np.nanmedian(stat_array)
        self.q25 = np.nanquantile(stat_array, 0.25)
        self.q75 = np.nanquantile(stat_array, 0.75)
        self.min = np.nanmin(stat_array)
        self.max = np.nanmax(stat_array)
        self.sum = np.nansum(stat_array)
        self.cov = self.sd / self.mean * 100
        self.skew = scipy.stats.skew(stat_array)
        self.geo_mean = scipy.stats.gmean(stat_array)
        self.geo_std = scipy.stats.gstd(stat_array)
