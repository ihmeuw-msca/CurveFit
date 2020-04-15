import numpy as np
import pandas as pd
import collections
from curvefit.core.utils import split_by_group
from general.diagnostics.baselines import LinearRegressionBaseline

class ResultChecker:

    def __init__(self, df, col_obs, col_group, col_est=None, models_dict=None):
        if col_est is None and models_dict is None:
            raise RuntimeError('must have either a column of estimates or CurveModels to generate estimates')
        self.df = df
        self.col_obs = col_obs
        self.col_group = col_group
        self.col_est = col_est
        self.models_dict = models_dict

        self.df_by_group = split_by_group(self.df, self.col_group)
        self.obs_by_group = {}
        self.est_by_group = {}
        self.groups = []
        for grp, df in self.df_by_group.items():
            self.obs_by_group[grp] = df[self.col_obs].to_numpy()      
            if self.col_est is not None:
                self.est_by_group[grp] = df[self.col_est].to_numpy()
            else:
                model = self.models_dict[grp]
                self.est_by_group[grp] = model.fun(model.t, model.params)
            self.groups.append(grp)

    def check_result(self):
        raise NotImplementedError() 


class LogDgaussian_cdfRegressionChecker(ResultChecker):

    def __init__(self, df, col_ln_gaussian_pdf_obs, col_group, col_t, col_est=None, models_dict=None):
        super().__init__(df, col_ln_gaussian_pdf_obs, col_group, col_est, models_dict)
        self.col_t = col_t

    def check_result(self): 
        ln_gaussian_pdf_obs = []
        times = []
        estimates = []
        for grp in self.groups:
            ln_gaussian_pdf_obs.append(self.obs_by_group[grp])
            times.append(self.df_by_group[grp][self.col_t].to_numpy())
            estimates.append(self.est_by_group[grp])
            
        lr_ln_gaussian_pdf = LinearRegressionBaseline(ln_gaussian_pdf_obs, self.groups, times)
        lr_ln_gaussian_pdf.fit()
        metric_fun_ln_gaussian_pdf = lambda est, obs: np.sqrt(np.mean((est - obs)**2))
        ln_gaussian_pdf_rmses = lr_ln_gaussian_pdf.compare(estimates, self.groups, metric_fun_ln_gaussian_pdf)
        metric_fun_gaussian_pdf = lambda est, obs: np.sqrt(np.mean((np.exp(est) - np.exp(obs))**2))
        gaussian_pdf_rmses = lr_ln_gaussian_pdf.compare(estimates, self.groups, metric_fun_gaussian_pdf)

        rmses = collections.defaultdict(list)
        for k, v in ln_gaussian_pdf_rmses.items():
            rmses[k].extend(v)
        for k, v in gaussian_pdf_rmses.items():
            rmses[k].extend(v)
        result_df = pd.DataFrame.from_dict(
            rmses, 
            orient='index', 
            columns=[
                'baseline RMSE ln_gaussian_pdf', 
                'curr model RMSE ln_gaussian_pdf', 
                'baseline RMSE gaussian_pdf', 
                'curr model RMSE gaussian_pdf',
            ]
        )
        return result_df

    
