import pandas as pd
import numpy as np

def calibration(samples, observation, observation_index=0, quantiles=[0.25, 0.5, 0.75, 0.95], resample="D", dates=None):
    """_summary_

    Args:
        samples (_type_): _description_
        observation (_type_): _description_
    """

    k, T    = observation.shape
    _, _, m = samples.shape # T: simulation length, k: number of observations, m: number of samples/ensembles

    calibration_df = []

    if dates is None:
        dates = pd.date_range(start="2020-01-01", periods=T, freq="D")

    for ki in range(k):

        cal_df                = pd.DataFrame(columns=["quantiles", "proportion_inside", "observation"])
        cal_df["quantiles"]   = quantiles
        cal_df["observation"] = ki
        cal_df                = cal_df.set_index("quantiles")

        for quant in cal_df.index.values:

            samples_ki                                  = np.take(samples, ki, axis=observation_index)

            df_ward                                     = pd.DataFrame(samples_ki)
            df_ward["date"]                             = dates
            df_ward                                     = df_ward.set_index("date").resample(resample).sum()
            df_resume                                   = df_ward.T.quantile(q=[0.5-quant/2, 0.5+quant/2]).T
            df_resume["obs"]                            = np.take(observation, ki, axis=observation_index)
            df_resume["calibration"]                    = df_resume.apply(lambda x: x[0.5-quant/2] <= np.double(x.obs) <= x[0.5+quant/2], axis=1)
            cal_df.loc[quant]                           = df_resume["calibration"].sum()/len(df_resume)
        cal_df["observation"] = ki

        calibration_df.append(cal_df)

    calibration_df  = pd.concat(calibration_df).reset_index()
    return calibration_df