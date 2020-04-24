# -*- coding: utf -*-
"""
    Peak Detector
"""
import numpy as np
import pandas as pd
from scipy.optimize import bisect
import matplotlib.pyplot as plt
import curvefit.core.utils as utils
from mrtool import MRData
from mrtool import MRBRT
from mrtool import LinearCovModel


class PeakDetector:
    def __init__(self, df, col_location, col_days, col_death_rate):
        df = utils.process_input(df,
                                 col_location,
                                 col_days,
                                 col_death_rate,
                                 asddr_lower_threshold=0.0)
        self.df = df
        self.col_location = 'location'
        self.col_days = 'days'
        self.col_ascdr = 'ascdr'
        self.data = utils.split_by_group(df, self.col_location)
        self.locations = np.sort(list(self.data.keys()))

        self.peaked_locations = None
        self.fit_result = None
        self.num_peaked_locations = None

    def pick_peaked_locations(self):
        pass

    def plot_peaked_locations(self):
        pass

    def save_result(self):
        pass


class PolyPeakDetector(PeakDetector):
    def __init__(self, *args):
        super().__init__(*args)
        self.peak_days = None

    def pick_peaked_locations(self,
                              potential_peaked_locations=None,
                              tol_num_obs=25,
                              tol_after_peak=0):
        if potential_peaked_locations is None:
            potential_peaked_locations = self.locations
        df = pd.concat([self.data[location]
                        for location in potential_peaked_locations])
        peaked_locations, poly_fit = utils.create_potential_peaked_groups(
            df, self.col_location, self.col_days, self.col_ascdr,
            tol_num_obs=tol_num_obs,
            tol_after_peak=tol_after_peak,
            return_poly_fit=True
        )

        self.peaked_locations = peaked_locations
        self.fit_result = poly_fit

        self.peak_days = {}
        for location in self.peaked_locations:
            c = self.fit_result[location]
            self.peak_days.update({
                location: -0.5*c[1]/c[0]
            })

        self.num_peaked_locations = len(self.peaked_locations)

    def plot_peaked_locations(self, locations=None):
        assert self.peaked_locations is not None
        assert self.fit_result is not None
        if locations is None:
            locations = self.peaked_locations

        fig, ax = plt.subplots(len(locations), 2,
                               figsize=(8*2, 4*len(locations)))
        ax = ax.reshape(len(locations), 2)

        for i, location in enumerate(locations):
            df_location = self.data[location]

            t = np.linspace(df_location['days'].min(),
                            df_location['days'].max(),
                            100) 
            log_y = np.polyval(self.fit_result[location], t)
            y = np.exp(log_y)
            
            t_peak = self.peak_days[location]
            log_y_peak = np.polyval(self.fit_result[location], [t_peak])[0]
            y_peak = np.exp(log_y_peak)

            ax[i, 0].scatter(df_location['days'],
                             df_location['asddr'],
                             edgecolors='k', color='#ADD8E6')
            ax[i, 0].plot(t, y, c='#4682B4')
            ax[i, 0].scatter(t_peak, y_peak, edgecolors='k', color='r')
            ax[i, 0].set_ylim(0.0, df_location['asddr'].max()*1.2)
            ax[i, 0].plot([t_peak, t_peak], [0.0, y_peak], 'r--')
            ax[i, 0].set_title(location + ': daily death rate')

            ax[i, 1].scatter(df_location['days'], df_location['ln asddr'], edgecolors='k', color='#ADD8E6')
            ax[i, 1].plot(t, log_y, c='#4682B4')
            ax[i, 1].scatter(t_peak, log_y_peak, edgecolors='k', color='r')
            ax[i, 1].plot([t_peak, t_peak], [df_location['ln asddr'].min(), log_y_peak], 'r--')
            ax[i, 1].set_title(location + ': log daily death rate')

    def save_result(self, file_name):
        df_output = pd.DataFrame({
            'location': self.peaked_locations,
            'peak days': [self.peak_days[location]
                          for location in self.peaked_locations]
        })
        df_output.to_csv(file_name)


class SplinePeakDetector(PeakDetector):
    def __init__(self, *args):
        super().__init__(*args)

        for location in self.data.keys():
            self.data[location]['obs_se'] = 1.0

        self.peak_durations = None
        self.fit_result = {}

    def fit_spline(self, location):
        df_location = self.data[location]
        mr_data = MRData(
                df=df_location,
                col_obs='ln asddr',
                col_obs_se='obs_se',
                col_covs=['days'],
                col_study_id='location',
                add_intercept=True
            )
            
        mr_cov_model = LinearCovModel(
            alt_cov=['days'],
            use_re=False,
            use_spline=True,
            spline_knots=np.linspace(0.0, 1.0, 5),
            spline_degree=2,
            spline_knots_type='domain',
            prior_spline_convexity='concave',
            prior_spline_num_constraint_points=40,
            name='days'
        )
        
        mr_model = MRBRT(
            data=mr_data,
            cov_models=[
                LinearCovModel('intercept',
                                use_re=True,
                                prior_gamma_uniform=np.array([0.0, 0.0])),
                mr_cov_model
            ]
        )

        mr_model.fit_model(inner_print_level=5, inner_max_iter=100)

        spline = mr_cov_model.create_spline(mr_data)
        spline_coef = mr_model.beta_soln.copy()
        spline_coef[1:] += spline_coef[0]
        
        self.fit_result[location] = (spline, spline_coef)

    def compute_peak_and_duration(self, location,
                                  tol_der=0.1,
                                  tol_after_peak=0):
        spline = self.fit_result[location][0]
        y0, y1, logy0, logy1 = self.y_fun(
            np.array([spline.knots[0], spline.knots[-1]]),
            location
        )
        if y1[0]*y1[1] > 0.0:
            return None

        # compute the peak
        peak = self.der_root(location)

        # compute the peak_start and peak_end
        peak_start = self.der_root(location,
                                   offset=tol_der*logy1[0],
                                   lb=spline.knots[0], ub=peak, log_space=True)
        peak_end = self.der_root(location,
                                 offset=tol_der*logy1[-1],
                                 lb=peak, ub=spline.knots[-1], log_space=True)

        self.peak_durations[location] = [
            peak_start,
            peak,
            peak_end,
            peak_end - peak_start
        ]

        if spline.knots[-1] - peak_end >= tol_after_peak:
            self.peaked_locations.append(location)

    def y_fun(self, t, location):
        spline, spline_coef = self.fit_result[location]
        m0 = spline.design_mat(t)
        m1 = spline.design_dmat(t, 1)
        logy0 = m0.dot(spline_coef)
        logy1 = m1.dot(spline_coef)
        y0 = np.exp(logy0)
        y1 = logy1*y0

        return y0, y1, logy0, logy1

    def der_root(self, location,
                 offset=0.0, lb=None, ub=None, log_space=False):
        df_location = self.data[location]
        spline, spline_coef = self.fit_result[location]
        lb = spline.knots[0] if lb is None else lb
        ub = spline.knots[-1] if ub is None else ub

        if log_space:
            fun = lambda t: self.y_fun(np.array([t]), location)[3][0] - offset
        else:
            fun = lambda t: self.y_fun(np.array([t]), location)[1][0] - offset
        result =  bisect(fun, lb, ub)
        return result

    def pick_peaked_locations(self,
                              potential_peaked_locations=None,
                              tol_num_obs=25,
                              tol_der=0.1,
                              tol_after_peak=0):
        if potential_peaked_locations is None:
            potential_peaked_locations = self.locations
        df = pd.concat([self.data[location]
                        for location in potential_peaked_locations])

        # fit splines to potential_peaked_locations
        for location in potential_peaked_locations:
            if self.data[location].shape[0] >= tol_num_obs:
                self.fit_spline(location)
            
        self.peaked_locations = []
        self.peak_durations = {}
        for i, location in enumerate(potential_peaked_locations):
            if self.data[location].shape[0] >= tol_num_obs:
                self.compute_peak_and_duration(location,
                                               tol_der=tol_der,
                                               tol_after_peak=tol_after_peak)

    def plot_peaked_locations(self,
                              locations=None,
                              handle=None,
                              return_handle=False,
                              color_line='#4682B4',
                              color_point='#ADD8E6',
                              color_indicator='r'):
        assert self.peaked_locations is not None
        assert self.fit_result is not None
        if locations is None:
            locations = self.peaked_locations

        if handle is None:
            fig, ax = plt.subplots(len(locations), 2,
                                   figsize=(8*2, 4*len(locations)))
        else:
            fig, ax = handle
        ax = ax.reshape(len(locations), 2)

        for i, location in enumerate(locations):
            df_location = self.data[location]
            spline = self.fit_result[location][0]

            peak_start = self.peak_durations[location][0]
            peak = self.peak_durations[location][1]
            peak_end = self.peak_durations[location][2]
            
            t = np.linspace(spline.knots[0], spline.knots[-1], 100)
            y0, y1, logy0, logy1 = self.y_fun(t, location)

            peak_y0, _, peak_logy0, _ = self.y_fun(np.array([peak]), location)
            peak_y0 = peak_y0[0]
            peak_logy0 = peak_logy0[0]
            
            ax[i, 0].scatter(df_location['days'], df_location['asddr'],
                             edgecolors='k',
                             color=color_point)
            ax[i, 0].plot(t, y0, c=color_line)
            ax[i, 0].set_ylim(0.0, df_location['asddr'].max()*1.2)
            ax[i, 0].axvline(peak_start, linestyle='--', color=color_indicator)
            ax[i, 0].axvline(peak_end, linestyle='--', color=color_indicator)
            ax[i, 0].scatter(peak, peak_y0, color='r')
            ax[i, 0].set_title(location + ': daily death rate')

            ax[i, 1].scatter(df_location['days'], df_location['ln asddr'],
                             edgecolors='k',
                             color=color_point)
            ax[i, 1].plot(t, logy0, c=color_line)
            ax[i, 1].axvline(peak_start, linestyle='--', color=color_indicator)
            ax[i, 1].axvline(peak_end, linestyle='--', color=color_indicator)
            ax[i, 1].scatter(peak, peak_logy0, color='r')
            ax[i, 1].set_title(location + ': log daily death rate')

        if return_handle:
            return fig, ax

    def save_result(self, file_name, return_df=True):
        df_output = pd.DataFrame({
            'location': self.peaked_locations,
            'peak start': [self.peak_durations[location][0]
                           for location in self.peaked_locations],
            'peak': [self.peak_durations[location][1]
                     for location in self.peaked_locations],
            'peak end': [self.peak_durations[location][2]
                         for location in self.peaked_locations],
            'peak durations': [self.peak_durations[location][3]
                               for location in self.peaked_locations]
        })
        df_output.to_csv(file_name)
        if return_df:
            return df_output
        else:
            return None
