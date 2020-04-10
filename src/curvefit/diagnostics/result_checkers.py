import numpy as np
import pandas as pd
from curvefit.core.utils import split_by_group
from general.diagnostics.baselines import LinearRegressionBaseline

class ResultChecker:

    def __init__(self, df, col_obs, col_est, col_group):
        self.df = df
        self.col_obs = col_obs
        self.col_est = col_est 
        self.col_group = col_group

    def check_result(self, groups=None):
        raise NotImplementedError() 


class LogDerfLinearRegressionChecker(ResultChecker):

    def __init__(self, df, col_log_derf_obs, col_est, col_group, col_t):
        super().__init__(df, col_log_derf_obs, col_est, col_group)
        self.col_t = col_t

    def check_result(self, groups=None): 
        if groups is not None:
            df_filtered = self.df[self.df[self.col_group].isin(groups)]
        else:
            df_filtered = self.df
        df_by_group = split_by_group(df_filtered, self.col_group)
        log_derf_rmses = {}

        log_derf_obs = []
        times = []
        estimates = []
        grps = []
        for grp, df in df_by_group.items():
            log_derf_obs.append(df[self.col_obs].to_numpy())
            times.append(df[self.col_t].to_numpy())
            estimates.append(df[self.col_est])
            grps.append(grp)
            
        lr_log_derf = LinearRegressionBaseline(log_derf_obs, grps, times)
        lr_log_derf.fit()
        metric_fun = lambda est, obs: np.sqrt(np.mean((est - obs)**2))
        log_derf_rmses = lr_log_derf.compare([estimates], [grps], metric_fun)
        
        result_df = pd.from_dict(log_derf_rmses, orient='index', columns=['baseline', 'curr model'])
        return result_df

    
