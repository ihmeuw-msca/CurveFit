import pandas as pd
import copy
from general.diagnostics.peak_detectors import PieceWiseLinearPeakDetector
from curvefit.core.utils import split_by_group

MINIMUM_NUM_POINTS = 6

class PeakDetector:

    def __init__(self, df, col_ln_gaussian_pdf_obs, col_group, col_t, peaked_groups, not_peaked_groups):
        self.df = df   
        self.col_ln_gaussian_pdf_obs = col_ln_gaussian_pdf_obs 
        self.col_group = col_group 
        self.col_t = col_t 
        self.peaked_groups = peaked_groups
        self.not_peaked_groups = not_peaked_groups

    def get_peak_detector(self):
        self.df_by_group = split_by_group(self.df, self.col_group)
        ln_gaussian_pdf_obs = []
        times = []
        peaked = []
        self.groups = []
        for grp, df in self.df_by_group.items():
            if grp in self.peaked_groups or grp in self.not_peaked_groups:
                ln_gaussian_pdf_obs.append(df[self.col_ln_gaussian_pdf_obs].to_numpy())
                times.append(df[self.col_t].to_numpy())
                if grp in self.peaked_groups:
                    peaked.append(1)
                else:
                    peaked.append(0)
                self.groups.append(grp)
        
        self.peak_detector = PieceWiseLinearPeakDetector(ln_gaussian_pdf_obs, self.groups, times, peaked)
        self.peak_detector.train_peak_classifier() 

    def predict_peaked(self):
        grp_to_pred = {}
        for grp, df in self.df_by_group.items():
            if grp not in self.groups:
                obs = df[self.col_ln_gaussian_pdf_obs].to_numpy()
                if len(obs) < MINIMUM_NUM_POINTS:
                    grp_to_pred[grp] = 'TBD'
                else:
                    self.peak_detector.has_peaked(obs, grp, df[self.col_t].to_numpy())
        for grp, pred in self.peak_detector.predicted.items():
            grp_to_pred[grp] = pred == 1
        return pd.DataFrame.from_dict(grp_to_pred, orient='index', columns=['peaked'])



    


    

