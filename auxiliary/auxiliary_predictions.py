import numpy as np
import pandas as pd
import pyreadstat as pread
import statsmodels.api as sm
import statsmodels.formula.api as smf
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from scipy.ndimage import gaussian_filter
from auxiliary.auxiliary_tables import *

variables_1 = ["realincome", "pctblack", "pcthighschl", "pcteligible"]
variables_2 = ["votingpop", "North", "South", "West"]
all_variables = [variables_1, variables_2]



def quartic25to75(dataset, varname):
    if varname not in dataset.columns:
        dataset = additional_column(dataset, varname)
    dataset.dembin = round(dataset["demvoteshare"], 2)
    dataset = dataset.dropna(subset=["state", "district", "dembin"])
    dataset = dataset.sort_values(by=["dembin"])

    dataset['x2'] = dataset.dembin ** 2
    dataset['x3'] = dataset.dembin ** 3
    dataset['x4'] = dataset.dembin ** 4
    meany100 = dataset.groupby("dembin")[varname].mean()

    dataset_25to50 = dataset.copy()
    dataset_25to50 = dataset_25to50[(dataset_25to50["demvoteshare"] < 0.50) & (dataset_25to50["demvoteshare"] >= 0.25)]
    result_25to50 = smf.ols(formula=f"{varname}~demvoteshare+x2+x3+x4", data=dataset_25to50).fit()
    ypred_25to50 = result_25to50.fittedvalues
    dataset_25to50["ypred"] = ypred_25to50

    lower = result_25to50.get_prediction().conf_int(alpha=0.05)[:, 0]
    upper = result_25to50.get_prediction().conf_int(alpha=0.05)[:, 1]
    dataset_25to50["lower"] = pd.DataFrame(lower, index=dataset_25to50.ypred.dropna().index)
    dataset_25to50["upper"] = pd.DataFrame(upper, index=dataset_25to50.ypred.dropna().index)

    dataset_50to75 = dataset.copy()
    dataset_50to75 = dataset_50to75[(dataset_50to75["demvoteshare"] > 0.50) & (dataset_50to75["demvoteshare"] <= 0.75)]
    result_50to75 = smf.ols(formula=f"{varname}~demvoteshare+x2+x3+x4", data=dataset_50to75).fit()
    ypred_50to75 = result_50to75.fittedvalues
    dataset_50to75["ypred"] = ypred_50to75

    lower = result_50to75.get_prediction().conf_int(alpha=0.05)[:, 0]
    upper = result_50to75.get_prediction().conf_int(alpha=0.05)[:, 1]
    dataset_50to75["lower"] = pd.DataFrame(lower, index=dataset_50to75.ypred.dropna().index)
    dataset_50to75["upper"] = pd.DataFrame(upper, index=dataset_50to75.ypred.dropna().index)

    df100_1 = dataset_25to50.groupby("dembin")[["ypred", "lower", "upper", "demvoteshare"]].mean()
    df100_2 = dataset_50to75.groupby("dembin")[["ypred", "lower", "upper", "demvoteshare"]].mean()
    df100 = df100_1.append(df100_2)
    df100[df100.index == 0.5] = np.nan
    df100["real_data"] = meany100.copy()

    return df100
