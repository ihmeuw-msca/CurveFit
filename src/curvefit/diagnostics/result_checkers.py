import numpy as np
import pandas as pd
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


class LogDerfRegressionChecker(ResultChecker):

    def __init__(self, df, col_log_derf_obs, col_group, col_t, col_est=None, models_dict=None):
        super().__init__(df, col_log_derf_obs, col_group, col_est, models_dict)
        self.col_t = col_t

    def check_result(self): 
        log_derf_obs = []
        times = []
        estimates = []
        for grp in self.groups:
            log_derf_obs.append(self.obs_by_group[grp])
            times.append(self.df_by_group[grp][self.col_t].to_numpy())
            estimates.append(self.est_by_group[grp])
            
        lr_log_derf = LinearRegressionBaseline(log_derf_obs, self.groups, times)
        lr_log_derf.fit()
        metric_fun = lambda est, obs: np.sqrt(np.mean((est - obs)**2))
        log_derf_rmses = lr_log_derf.compare(estimates, self.groups, metric_fun)
        
        result_df = pd.DataFrame.from_dict(log_derf_rmses, orient='index', columns=['baseline RMSE', 'curr model RMSE'])
        return result_df

    
