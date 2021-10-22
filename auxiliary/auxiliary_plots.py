import numpy as np
import pandas as pd
import pyreadstat as pread
import statsmodels.api as sm
import statsmodels.formula.api as smf
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from scipy.ndimage import gaussian_filter
from auxiliary.auxiliary_predictions import *


df = pread.read_dta("data/enricoall2.dta")[0]


def ada_score(dataset, lag=False):
    variable = None
    if lag:
        variable = "lagdemvoteshare"
        time = "t+1"
    else:
        variable = "demvoteshare"
        time = "t"
    dataset['d1'] = np.nan
    dataset.loc[dataset[variable] <= 0.5, "d1"] = 0
    dataset.loc[dataset[variable] > 0.5, "d1"] = 1

    dataset.dembin = round(dataset[variable], 2)
    dataset = dataset.dropna(subset=["state", "district", "dembin"])
    dataset = dataset.sort_values(by=["dembin"])

    dataset['x2'] = dataset.dembin ** 2
    dataset['x3'] = dataset.dembin ** 3
    dataset['x4'] = dataset.dembin ** 4

    dataset['dd1'] = np.nan
    dataset.loc[dataset.dembin <= 0.5, "dd1"] = 0
    dataset.loc[dataset.dembin > 0.5, "dd1"] = 1
    dataset["dembin_100"] = pd.cut(dataset.dembin, 100)

    result = smf.ols(formula="realada~dd1+x2+x3+x4", data=dataset).fit()
    ypred = result.fittedvalues
    dataset['ypred'] = ypred

    lower = result.get_prediction().conf_int(alpha=0.05)[:, 0]
    upper = result.get_prediction().conf_int(alpha=0.05)[:, 1]
    dataset["lower"] = pd.DataFrame(lower, index=dataset.ypred.dropna().index)
    dataset["upper"] = pd.DataFrame(upper, index=dataset.ypred.dropna().index)
    a1 = dataset[["ypred", "lower", "upper", "realada"]][dataset.dembin < 0.5]
    a2 = dataset[["ypred", "lower", "upper", "realada"]][dataset.dembin > 0.5]
    ypred_all = a1.append(a2)
    ypred_all["dembin_100"] = dataset["dembin_100"].copy()

    meany100 = ypred_all.groupby('dembin_100')[['ypred', "upper", "lower", "realada"]].mean()
    x_axis = np.linspace(0, 1, 100)

    plt.plot(x_axis, meany100.ypred, color="black")
    plt.plot(x_axis, meany100.upper, "--", color="black", linewidth=0.7)
    plt.plot(x_axis, meany100.lower, "--", color="black", linewidth=0.7)
    plt.scatter(x_axis, meany100.realada, color="gray")
    plt.ylim(0, 100)
    plt.xlim(0, 1)
    plt.axvline(0.5)
    plt.xlabel("Democratic Vote Share, time t")
    plt.ylabel(f"ADA Score, time {time}")
    plt.show()

    return


def dwscore(dataset, lag=False):
    variable = None
    if lag:
        variable = "lagdemvoteshare"
        time = "t+1"
    else:
        variable = "demvoteshare"
        time = "t"
    dataset['d1'] = np.nan
    dataset.loc[dataset[variable] <= 0.5, "d1"] = 0
    dataset.loc[dataset[variable] > 0.5, "d1"] = 1

    dataset.dembin = round(dataset[variable], 2)
    dataset = dataset.dropna(subset=["state", "district", "dembin"])
    dataset = dataset.sort_values(by=["dembin"])

    dataset['x2'] = dataset.dembin ** 2
    dataset['x3'] = dataset.dembin ** 3
    dataset['x4'] = dataset.dembin ** 4

    dataset['dd1'] = np.nan
    dataset.loc[dataset.dembin <= 0.5, "dd1"] = 0
    dataset.loc[dataset.dembin > 0.5, "dd1"] = 1
    dataset["dembin_100"] = pd.cut(dataset.dembin, 100)

    result = smf.ols(formula="dwnom1~dd1+x2+x3+x4", data=dataset).fit()
    ypred = result.fittedvalues
    dataset['ypred'] = ypred

    lower = result.get_prediction().conf_int(alpha=0.05)[:, 0]
    upper = result.get_prediction().conf_int(alpha=0.05)[:, 1]
    dataset["lower"] = pd.DataFrame(lower, index=dataset.ypred.dropna().index)
    dataset["upper"] = pd.DataFrame(upper, index=dataset.ypred.dropna().index)
    a1 = dataset[["ypred", "lower", "upper", "dwnom1"]][dataset.dembin < 0.5]
    a2 = dataset[["ypred", "lower", "upper", "dwnom1"]][dataset.dembin > 0.5]
    ypred_all = a1.append(a2)
    ypred_all["dembin_100"] = dataset["dembin_100"].copy()

    meany100 = ypred_all.groupby('dembin_100')[['ypred', "upper", "lower", "dwnom1"]].mean()
    x_axis = np.linspace(0, 1, 100)

    plt.plot(x_axis, meany100.ypred, color="black")
    plt.plot(x_axis, meany100.upper, "--", color="black", linewidth=0.7)
    plt.plot(x_axis, meany100.lower, "--", color="black", linewidth=0.7)
    plt.scatter(x_axis, meany100.dwnom1, color="gray")
    plt.ylim(-1, 0.5)
    plt.xlim(0, 1)
    plt.axvline(0.5)
    plt.xlabel("Democratic Vote Share, time t")
    plt.ylabel(f"Nominate Score, time {time}")
    plt.show()

    return

def Dlead(dataset, lag=False):
    variable = None
    if lag:
        variable = "lagdemvoteshare"
        time = "t+1"
    else:
        variable = "demvoteshare"
        time = "t"
    dataset['d1'] = np.nan
    dataset.loc[dataset[variable] <= 0.5, "d1"] = 0
    dataset.loc[dataset[variable] > 0.5, "d1"] = 1

    dataset.dembin = round(dataset[variable], 2)
    dataset = dataset.dropna(subset=["state", "district", "dembin"])
    dataset = dataset.sort_values(by=["dembin"])

    dataset['x2'] = dataset.dembin ** 2
    dataset['x3'] = dataset.dembin ** 3
    dataset['x4'] = dataset.dembin ** 4

    dataset['dd1'] = np.nan
    dataset.loc[dataset.dembin <= 0.5, "dd1"] = 0
    dataset.loc[dataset.dembin > 0.5, "dd1"] = 1
    dataset["dembin_100"] = pd.cut(dataset.dembin, 100)

    result = smf.ols(formula="eq_Dlead~dd1+x2+x3+x4", data=dataset).fit()
    ypred = result.fittedvalues
    dataset['ypred'] = ypred

    lower = result.get_prediction().conf_int(alpha=0.05)[:, 0]
    upper = result.get_prediction().conf_int(alpha=0.05)[:, 1]
    dataset["lower"] = pd.DataFrame(lower, index=dataset.ypred.dropna().index)
    dataset["upper"] = pd.DataFrame(upper, index=dataset.ypred.dropna().index)
    a1 = dataset[["ypred", "lower", "upper", "eq_Dlead"]][dataset.dembin < 0.5]
    a2 = dataset[["ypred", "lower", "upper", "eq_Dlead"]][dataset.dembin > 0.5]
    ypred_all = a1.append(a2)
    ypred_all["dembin_100"] = dataset["dembin_100"].copy()

    meany100 = ypred_all.groupby('dembin_100')[['ypred', "upper", "lower", "eq_Dlead"]].mean()
    x_axis = np.linspace(0, 1, 100)

    plt.plot(x_axis, meany100.ypred, color="black")
    plt.plot(x_axis, meany100.upper, "--", color="black", linewidth=0.7)
    plt.plot(x_axis, meany100.lower, "--", color="black", linewidth=0.7)
    plt.scatter(x_axis, meany100.eq_Dlead, color="gray")
    plt.ylim(0.4, 0.9)
    plt.xlim(0, 1)
    plt.axvline(0.5)
    plt.xlabel("Democratic Vote Share, time t")
    plt.ylabel(f"Percent Vote Equal to Democrat Leader, time {time}")
    plt.show()

    return



def probability_democrat(dataset):
    variable = "lagdemvoteshare"
    time = "t+1"

    prob = np.zeros(shape=101)
    for i in range(0, 101):
        a = dataset.democrat[(dataset[variable] >= (i / 100)) & (dataset[variable] < (i / 100 + 0.01))]
        try:
            prob[i] = len(a[a == 1]) / len(a)
        except:
            prob[i] = np.nan

    df100 = pd.DataFrame(prob, columns=["prob"])
    df100["x2"] = prob**2
    df100["x3"] = prob**3
    df100["x4"] = prob**4

    df100['dd1'] = np.nan
    df100.loc[df100.prob <= 0.5, "dd1"] = 0
    df100.loc[df100.prob > 0.5, "dd1"] = 1

    result = smf.ols(formula=f"prob~dd1+x2+x3+x4", data=df100).fit()
    ypred = result.fittedvalues
    df100['ypred'] = ypred
    lower = result.get_prediction().conf_int(alpha=0.05)[:, 0]
    upper = result.get_prediction().conf_int(alpha=0.05)[:, 1]
    df100["lower"] = pd.DataFrame(lower, index=df100.ypred.dropna().index)
    df100["upper"] = pd.DataFrame(upper, index=df100.ypred.dropna().index)
    a1 = df100[["ypred", "lower", "upper"]][df100.prob < 0.5]
    a2 = df100[["ypred", "lower", "upper"]][df100.prob > 0.5]
    df100[["ypred", "lower", "upper"]] = a1.append(a2)

    for i in df100.columns[-3:]:
        df100[i] = gaussian_filter(df100[i].values, sigma=1)
        df100[i][50] = np.nan

    x_axis = np.linspace(0, 1, 101)
    plt.plot(x_axis, df100.ypred, color="black")
    plt.plot(x_axis, df100.upper, "--", color="black", linewidth=0.7)
    plt.plot(x_axis, df100.lower, "--", color="black", linewidth=0.7)
    plt.scatter(x_axis, prob, color="gray")
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.axvline(0.5)
    plt.xlabel("Democratic Vote Share, time t")
    plt.ylabel(f"Probability of Democrat Wins, time {time}")
    plt.show()

    return

variables_1 = ["realincome", "pctblack", "pcthighschl", "pcteligible"]
variables_2 = ["votingpop", "North", "South", "West"]
all_variables = [variables_1, variables_2]

for i in all_variables:
    for var in i:
        globals()[f"df_{var}"] = quartic25to75(df, var)

def subplot2_2(variables:list):

    fig, axs = plt.subplots(2,2)
    fig.subplots_adjust(hspace=.5, wspace=.35)
    axs = axs.ravel()
    x_axis = globals()[f"df_{variables[0]}"]["demvoteshare"]

    for i in range(axs.size):
        data = globals()[f"df_{variables[i]}"].copy()
        axs[i].scatter(x_axis,data["real_data"],color="gray", s=10)
        axs[i].plot(x_axis, data["ypred"], color="black")
        axs[i].plot(x_axis, data["upper"], "--", color="black", linewidth=0.7)
        axs[i].plot(x_axis, data["lower"], "--", color="black", linewidth=0.7)
        ylims = (data[["ypred", "lower", "upper"]].min().min() - data[["ypred", "lower", "upper"]].std().max(),
                 data[["ypred", "lower", "upper"]].max().max() + data[["ypred", "lower", "upper"]].std().max())
        axs[i].set_ylim(ylims)
        ylabel = variables[i]
        if "real" in variables[i]:
            ylabel = variables[i].split("real")[1]
        elif "pct" in variables[i]:
            ylabel = variables[i].split("pct")[1]
        elif "pop" in variables[i]:
            ylabel = "Voting Pop"
        axs[i].set_ylabel(ylabel)
        axs[i].set_xlabel("Democratic Vote Share, time t")
        axs[i].set_xlim(0.25, 0.75)
        axs[i].axvline(0.5, color="black")
    fig.tight_layout()
    fig.show()
    return

def plot_hist(dataset, variable, bins):
    data = dataset[variable].dropna().values.copy()
    data[data == 1] = 0
    data = data[data > 0]
    plt.hist(data, bins=bins, density=True, label=variable, color="gray")
    plt.legend()
    plt.xlabel(variable)
    plt.ylabel("Density")
    plt.show()



def lagada_score(dataset):

    variable = "demvoteshare"
    dataset['d1'] = np.nan
    dataset.loc[dataset[variable] <= 0.5, "d1"] = 0
    dataset.loc[dataset[variable] > 0.5, "d1"] = 1

    dataset.dembin = round(dataset[variable], 2)
    dataset = dataset.dropna(subset=["state", "district", "dembin"])
    dataset = dataset.sort_values(by=["dembin"])

    dataset['x2'] = dataset.dembin ** 2
    dataset['x3'] = dataset.dembin ** 3
    dataset['x4'] = dataset.dembin ** 4

    dataset['dd1'] = np.nan
    dataset.loc[dataset.dembin <= 0.5, "dd1"] = 0
    dataset.loc[dataset.dembin > 0.5, "dd1"] = 1
    dataset["dembin_100"] = pd.cut(dataset.dembin, 100)

    result = smf.ols(formula="lagada_vs~dd1+x2+x3+x4", data=dataset).fit()
    ypred = result.fittedvalues
    dataset['ypred'] = ypred

    lower = result.get_prediction().conf_int(alpha=0.05)[:, 0]
    upper = result.get_prediction().conf_int(alpha=0.05)[:, 1]
    dataset["lower"] = pd.DataFrame(lower, index=dataset.ypred.dropna().index)
    dataset["upper"] = pd.DataFrame(upper, index=dataset.ypred.dropna().index)
    a1 = dataset[["ypred", "lower", "upper", "lagada_vs"]][dataset.dembin < 0.5]
    a2 = dataset[["ypred", "lower", "upper", "lagada_vs"]][dataset.dembin > 0.5]
    ypred_all = a1.append(a2)
    ypred_all["dembin_100"] = dataset["dembin_100"].copy()

    meany100 = ypred_all.groupby('dembin_100')[['ypred', "upper", "lower", "lagada_vs"]].mean()
    x_axis = np.linspace(0, 1, 100)

    plt.plot(x_axis, meany100.ypred, color="black")
    plt.plot(x_axis, meany100.upper, "--", color="black", linewidth=0.7)
    plt.plot(x_axis, meany100.lower, "--", color="black", linewidth=0.7)
    plt.scatter(x_axis, meany100.lagada_vs, color="gray")
    plt.ylim(0, 100)
    plt.xlim(0, 1)
    plt.axvline(0.5)
    plt.xlabel("Democratic Vote Share, time t")
    plt.ylabel(f"ADA Score, time t-1")
    plt.show()

    return



