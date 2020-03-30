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


def pv_for_single_group(data, col_t, col_obs, col_grp, col_obs_compare,
                        model_generator, predict_space, predict_group):
    """
    Gets forward out of sample predictive validity for a model based on the function
    fit_model that takes arguments df and times and returns predictions at times.

    Args:
        data: (pd.DataFrame)
        col_t: (str) column indicating the time
        col_obs: (str) column indicating the observation
        col_grp: (str) column indicating the group membership
        col_obs_compare: (str) column indicating the observation that represents the space we will
            be calculating predictive validity in (can be different from col_obs, but your fit_model
            function for predict should match that)
        model_generator: object of class model_generator.ModelGenerator
        predict_space: a function from curvefit.model that gives the prediction space to calculate PV in
            (needs to be the same as col_obs_compare)
        predict_group: name of group to predict for

    Returns:
        dictionary of
            times: (np.array) of available prediction times
            predictions: matrix of predictions of dimension times x times
            models: list of models at each prediction time
            r_matrix: residual matrix transformed so that
                r_matrix[:,0]: how far out predicting
                r_matrix[:,1]: how many data points to do prediction
                r_matrix[:,2]: the corresponding residual
    """
    assert type(col_t) == str
    assert type(col_obs) == str
    assert col_t in data.columns
    assert col_obs in data.columns
    assert callable(model_generator.fit)
    assert callable(model_generator.predict)

    all_df = data.copy()
    grp_df = data.loc[data[col_grp] == predict_group].copy()
    available_times = np.unique(grp_df[col_t].values)

    # get the differences between the available times
    # these need to all be integers. the cumulative differences
    # tells us how big of a step we took to the next data point
    difference = np.diff(available_times)
    assert np.isclose(np.array([round(x) for x in difference]), difference, atol=1e-14).all()
    difference = np.array([int(round(x)) for x in difference])

    # which observations are we comparing the predictions to? and how much data do we have?
    compare_observations = grp_df[col_obs_compare].values
    amount_data = np.array(range(len(compare_observations))) + 1

    preds = []
    models = {}
    for i in available_times:
        print(f"Fitting model for end time {i}", end='\r')

        model = model_generator.generate()
        # remove the rows for this group that are greater than the available times
        remove_rows = (all_df[col_t] > i) & (all_df[col_grp] == predict_group)
        df = all_df[~remove_rows].copy()

        # fit the model on the rest of the data and predict for this particular group
        model.fit(df=df)
        preds.append(model.predict(
            times=available_times,
            predict_space=predict_space,
            predict_group=predict_group
        ))
        models[i] = model

    predictions = np.vstack([preds])
    residuals = predictions - compare_observations

    far_out = np.array([])
    num_data = np.array([])
    robs = np.array([])

    diagonals = np.array(range(residuals.shape[0]))[1:]

    # get the diagonal of the residual matrix and figure out
    # how many data points out we were predicting (convolve)
    # plus the amount of data that we had to do the prediction
    for i in diagonals:
        diagonal = np.diag(residuals, k=i)
        obs = len(diagonal)
        out = np.convolve(difference, np.ones(i, dtype=int), mode='valid')

        far_out = np.append(far_out, out[-obs:])
        num_data = np.append(num_data, amount_data[:obs])
        robs = np.append(robs, diagonal)

    # return the results for the residual matrix as a (len(available_times), 3) shaped matrix
    r_matrix = np.vstack([far_out, num_data, robs]).T

    return {
        'times': available_times,
        'predictions': predictions,
        'models': models,
        'r_matrix': r_matrix
    }


def model_pv(df, col_group, col_t, col_obs, col_obs_compare, model_generator, predict_space):
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
    model_results = {}
    r_matrices = {}

    for grp in groups:
        print(f"Getting PV for group {grp}")
        results = pv_for_single_group(
            data=data,
            col_t=col_t,
            col_obs=col_obs,
            col_grp=col_group,
            col_obs_compare=col_obs_compare,
            model_generator=model_generator,
            predict_space=predict_space,
            predict_group=grp
        )
        prediction_times[grp] = results['times']
        prediction_results[grp] = results['predictions']
        model_results[grp] = results['models']
        r_matrices[grp] = results['r_matrix']

    return prediction_times, prediction_results, model_results, r_matrices
