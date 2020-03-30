import numpy as np


def get_residual_std(matrix, n, window):
    """
    Gets the residual standard deviation from a model
    residual matrix predicting n time points out from its last data point
    and using a maximum time window to average.

    NOTE: masks nans in case there was a missing observation.

    Args:
        matrix: (np.ndarray) square 2 dimensional array with
            each time points observation across the columns and
            each model fit with the last time point across rows
        n: (int) number of time points predicting out (n=0) would be
            the entry for the model where n^{th} observation was the
            last observation in the model
        window: (int) number of time + n points to use from the most
            recent one

    Returns:
        (float)
    """
    assert len(matrix.shape) == 2
    assert matrix.shape[0] == matrix.shape[1]
    assert type(window) == int
    assert type(n) == int

    return np.nanstd(
        np.diag(matrix, k=n)[-window:]
    )


def get_full_data(grp_df, col_t, col_obs, prediction_times):
    """
    Fill out an observation vector for all one increment time
    points, filling in missing values with nan.

    Args:
        grp_df: (pd.DataFrame) data frame with time and observation
            column
        col_t: (str) the time column
        col_obs: (str) the observation column
        prediction_times: (np.array) the time vector for which predictions are needed

    Returns:
        (np.array) observations filled out to the max time point
        in the loc_df
    """
    assert type(col_t) == str
    assert type(col_obs) == str
    assert col_t in grp_df.columns
    assert col_obs in grp_df.columns

    data_dict = grp_df[[col_t, col_obs]].set_index(col_t).to_dict(orient='index')
    data_dict = {k: v[col_obs] for k, v in data_dict.items()}
    full_data = np.array([data_dict[x] if x in data_dict else np.nan for x in prediction_times])
    return full_data


def pv_for_single_group(grp_df, col_t, col_obs, col_obs_compare, model_generator, predict_space):
    """
    Gets forward out of sample predictive validity for a model based on the function
    fit_model that takes arguments df and times and returns predictions at times.

    Args:
        grp_df: (pd.DataFrame)
        col_t: (str) column indicating the time
        col_obs: (str) column indicating the observation
        col_obs_compare: (str) column indicating the observation that represents the space we will
            be calculating predictive validity in (can be different from col_obs, but your fit_model
            function for predict should match that)
        model_generator: object of class model_generator.ModelGenerator
        predict_space: a function from curvefit.model that gives the prediction space to calculate PV in
            (needs to be the same as col_obs_compare)

    Returns:
        times: (np.array) of prediction times
        predictions: matrix of predictions of dimension times x times
        residuals: matrix of residuals of dimension times x times
    """
    assert type(col_t) == str
    assert type(col_obs) == str
    assert col_t in grp_df.columns
    assert col_obs in grp_df.columns
    assert callable(model_generator.fit)
    assert callable(model_generator.predict)

    available_times = np.unique(grp_df[col_t].values)
    difference = np.diff(available_times)
    assert all(difference == 1)
    cumulative_differences = np.cumsum(difference)

    compare_observations = grp_df[col_obs_compare].values

    predictions = []
    models = {}
    for i in available_times:
        print(f"Fitting model for end time {i}", end='\r')

        model = model_generator.generate()
        df_i = grp_df.loc[grp_df[col_t] <= i].copy()

        model.fit(df=df_i)
        predictions.append(model.predict(times=available_times, predict_space=predict_space))
        models[i] = model

    import pdb; pdb.set_trace()

    all_preds = np.vstack([predictions])
    residuals = all_preds - compare_observations

    return all_preds, residuals, models


def pv_for_group_collection(df, col_group, col_t, col_obs, col_obs_compare, model_generator, predict_space):
    """
    Gets a dictionary of predictive validity for all groups in the data frame.
    Args:
        df: (pd.DataFrame)
        col_group: (str) grouping column string
        col_t: (str) column indicating the time
        col_obs: (str) column indicating the observation for fitting
        col_obs_compare: (str) column indicating the observation that represents the space we will
            be calculating predictive validity in (can be different from col_obs, but your fit_model
            function for predict should match that)
        model_generator: object of class model_generator.ModelGenerator
        predict_space: a function from curvefit.model that gives the prediction space to calculate PV in
            (needs to be the same as col_obs_compare)

    Returns:
        dictionaries for prediction times, predictions, residuals, and models
    """
    assert type(col_group) == str
    assert col_group in df.columns
    data = df.copy()

    groups = sorted(data[col_group].unique())

    prediction_times = {}
    prediction_results = {}
    residual_results = {}
    model_results = {}

    for grp in groups:
        print(f"Getting PV for group {grp}")
        grp_df = data.loc[data[col_group] == grp].copy()
        times, preds, resid, mods = pv_for_single_group(
            grp_df=grp_df,
            col_t=col_t,
            col_obs=col_obs,
            col_obs_compare=col_obs_compare,
            model_generator=model_generator,
            predict_space=predict_space
        )
        prediction_times[grp] = times
        prediction_results[grp] = preds
        residual_results[grp] = resid
        model_results[grp] = mods

    return prediction_times, prediction_results, residual_results, model_results
