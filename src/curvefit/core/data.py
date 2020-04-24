from dataclasses import dataclass
from typing import List
from curvefit.core.utils import data_translator

@dataclass
class DataSpecs:
    col_t: str
    col_obs: str 
    col_covs: List[str]
    col_group: str 
    col_obs_se: str = None
    obs_space: callable
    obs_se_fun: callable = None



class Data:
    """
    {begin_markdown Data}

    {spell_markdown func covs init bool}

    # `curvefit.core.data.Data`
    ## Manages all data to be used for curve fitting

    The data class contains all information about the complete dataset, list of groups, column names.
    Creates observation standard error based on `obs_se_func`.
    All sorting will happen immediately in the `Data` class `init` and we won’t have to sort again.

    NOTE: will prioritize `obs_se_func` over `col_obs_se`.

    ## Syntax
    ```python
    d = Data(df, col_t, col_obs, obs_space, col_covs, col_group)
    ```

    ## Arguments

    - `df (pandas.DataFrame)`: all data
    - `col_t (str)`: column with time
    - `col_obs (str)`: column with observation values
    - `obs_space (callable)`: function from functions module that specifies what space the `col_obs`
        column represents (e.g. ln_gaussian_cdf)
    - `col_covs (List[str])`: covariate columns
    - `col_group (str)`: group column
    - `col_obs_se (optional, str)`: column with observation standard error specified.
    - `obs_se_func (optional, callable)`: observation standard error function to create from time

    ## Attributes

    - `self.groups (List[str])`: list of groups, sorted in order
    - `self.num_obs (int)`: dictionary of numbers of observations per group (key)
    - `self.times (Dict[str: np.array]))`: dictionary of times per group (key)

    ## Methods

    ### `get_df`
    Returns a copy of the data frame, or a pointer to the data frame. If you plan on modifying the data frame,
    use `copy=True`. If a group is passed, then a group-specific data frame will be passed.

    - `group (optional, str)`: optional group name
    - `copy (bool)`: return a copy or not

    ### `get_translated_observations`
    Returns the observations for a group in a particular space from `curvefit.core.functions`

    - `group (str)`: which group to return observations for
    - `space (callable)`: which space to translate observations to

    {end_markdown Data}
    """
    def __init__(self, df, data_specs):

        self.df = df.copy()
        self.data_specs = data_specs
        self.col_t = data_specs.col_t
        self.col_obs = data_specs.col_obs
        self.col_covs = data_specs.col_covs
        self.col_group = data_specs.col_group
        self.obs_space = data_specs.obs_space
        self.obs_se_func = data_specs.obs_se_func

        self.df.sort_values([self.col_group, self.col_t], inplace=True)

        if self.obs_se_func is not None:
            self.col_obs_se = 'obs_se'
            self.df[self.col_obs_se] = self.df[self.col_t].apply(self.obs_se_func)
        else:
            self.col_obs_se = data_specs.col_obs_se

        self.groups = self.df[self.col_group].unique()

    def get_df(self, group=None, copy=False, return_specs=False):
        if group is not None:
            df = self.df.loc[self.df[self.col_group] == group]
        else:
            df = self.df

        if not return_specs:
            if copy:
                return df.copy()
            else:
                return df
        else:
            return df, self.data_specs

    def get_translated_observations(self, group, space):
        values = self.get_df(group=group)[self.col_obs].values
        return data_translator(
            data=values,
            input_space=self.obs_space,
            output_space=space
        )
