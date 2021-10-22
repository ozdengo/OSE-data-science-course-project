import numpy as np
import pandas as pd
import pyreadstat as pread
import statsmodels.api as sm
import statsmodels.formula.api as smf
from itertools import combinations
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from scipy.ndimage import gaussian_filter
from stargazer.stargazer import Stargazer


def table_vscore(dataset, var, year=None, bandwidth=False):
    if year:
        year1,year2 = year
        dataset = dataset[(dataset.year >= year1) & dataset.year <= year2]
    dataset = dataset[(dataset.year!=1952)&(dataset.year!=1962)&(dataset.year!=1972)&(dataset.year!=1982)&(dataset.year!=1992)]
    if bandwidth:
        dataset = dataset[(dataset.lagdemvoteshare>.48) & (dataset.lagdemvoteshare<.52)]
    del dataset["democrat"]
    
    dataset["democrat"]=np.nan
    dataset.loc[dataset.demvoteshare>=.5, "democrat"] = 1
    dataset.loc[dataset.demvoteshare<.5, "democrat"] = 0
    
    dataset["lagdemocrat"]=np.nan
    dataset.loc[dataset.lagdemvoteshare>=.5, "lagdemocrat"] = 1
    dataset.loc[dataset.lagdemvoteshare<.5, "lagdemocrat"] = 0
    
    dataset["score"] = dataset[var]
    
    result1 = smf.ols(formula="score~lagdemocrat", data=dataset).fit()
    result2 = smf.ols(formula="score~democrat", data=dataset).fit()
    result3 = smf.ols(formula="democrat~lagdemocrat", data=dataset).fit()

    table = Stargazer([result1,result2,result3])
    table.custom_columns(['$\gamma$', '$π_1$', '($P^{D}_{t+1} - P^{R}_{t+1}$)'], [1, 1, 1])
    table.show_model_numbers(False)
    table.significant_digits(2)

    return table


variables_1 = ["realincome", "pctblack", "pcthighschl", "pcteligible"]
variables_2 = ["votingpop", "North", "South", "West"]
all_variables = [variables_1, variables_2]

def additional_column(dataset, varname):

    if varname == "pcteligible":
        dataset["pcteligible"] = dataset["votingpop"] / dataset["totpop"]
    elif varname in variables_2:
        if varname == "North":
            dataset[varname] = 0
            dataset.loc[dataset["state"] <= 37, varname] = 1
        elif varname == "South":
            dataset[varname] = 0
            dataset.loc[(dataset["state"]>=41)&(dataset["state"]<=56), varname] = 1
        elif varname == "West":
            dataset[varname] = 0
            dataset.loc[dataset["state"] >= 61, varname] = 1
    else:
        return print("Column name could not found")

    return dataset


def table_char(dataset):
    table_variables = ["realincome", "pcthighschl", "pcturban", "pctblack", "state", "demvoteshare", "democrat"]
    clear_df = dataset[table_variables]
    table_variables += ["North", "South", "West"]
    table_variables.remove("state")
    table_variables.remove("demvoteshare")
    table_variables.remove("democrat")
    clear_df["dembin"] = round(clear_df["demvoteshare"], 2)
    df_table = pd.DataFrame(np.array([[0, 0, 0, 0, 0, 0]] * len(table_variables)).astype(float),
                            columns=["all", "bw_25", "bw_10", "bw_5", "bw_2", "poly"], index=table_variables)
    df_table_err = pd.DataFrame(np.array([[0, 0, 0, 0, 0, 0]] * len(table_variables)).astype(float),
                                columns=["all", "bw_25", "bw_10", "bw_5", "bw_2", "poly"], index=table_variables)
    for varname in table_variables:
        if varname not in clear_df.columns:
            clear_df = additional_column(clear_df, varname)

        for index,i in enumerate((0.51,0.25,0.1,0.05,0.02,1)):
            if i != 1:
                data = clear_df[(clear_df["demvoteshare"]>0.5-i) & (clear_df["demvoteshare"]<0.5+i)]
                regress = smf.ols(formula=f"{varname}~democrat", data=data).fit()
                df_table._set_value(varname, df_table.columns[index], round(regress.params["democrat"],3))
                df_table_err._set_value(varname, df_table_err.columns[index], round(regress.bse["democrat"],3))
            else:
                data = clear_df.copy()
                data['x2'] = data.demvoteshare ** 2
                data['x3'] = data.demvoteshare ** 3
                data['x4'] = data.demvoteshare ** 4
                regress = smf.ols(formula=f"{varname}~demvoteshare+x2+x3+x4", data=data).fit()
                df_table._set_value(varname, df_table.columns[index], round(regress.params["demvoteshare"],3))
                df_table_err._set_value(varname, df_table_err.columns[index], round(regress.bse["demvoteshare"],3))

    return df_table, df_table_err



def table_multiyears(dataset_in, var='realada', years=list):
    col1, col2, col3 = "$ADA_{t+1}$", "$ADA_t$", "$DEM_{t+1}$"
    dataset_in = dataset_in[
        (dataset_in.year != 1952) & (dataset_in.year != 1962) & (dataset_in.year != 1972) & (
                    dataset_in.year != 1982) & (
                dataset_in.year != 1992)]
    dataset_in = dataset_in[(dataset_in.lagdemvoteshare > .48) & (dataset_in.lagdemvoteshare < .52)]

    type_list = ["", " "] * 4
    df_years = pd.DataFrame(np.zeros(shape=(8, 4)), columns=[" ", col1, col2, col3],
                            index=[str(x[0]) + "-" + str(x[1]) for x in years] * 2).sort_index()
    df_years[" "] = type_list
    param_list = ["lagdemocrat", "democrat", "lagdemocrat"]

    for y in years:
        year1, year2 = y
        dataset = dataset_in[(dataset_in.year >= year1) & (dataset_in.year <= year2)].copy()
        dataset["democrat"] = np.nan
        dataset.loc[dataset.demvoteshare >= .5, "democrat"] = 1
        dataset.loc[dataset.demvoteshare < .5, "democrat"] = 0

        dataset["lagdemocrat"] = np.nan
        dataset.loc[dataset.lagdemvoteshare >= .5, "lagdemocrat"] = 1
        dataset.loc[dataset.lagdemvoteshare < .5, "lagdemocrat"] = 0

        dataset["score"] = dataset[var].copy()
        result1 = smf.ols(formula="score~lagdemocrat", data=dataset).fit()
        result2 = smf.ols(formula="score~democrat", data=dataset).fit()
        result3 = smf.ols(formula="democrat~lagdemocrat", data=dataset).fit()
        result_all = [result1, result2, result3]
        index_year = str(year1) + "-" + str(year2)
        for i, col in enumerate(df_years.columns[1:]):
            df_years._set_value(index_year, col, (result_all[i].params[param_list[i]].round(2),
                                                  result_all[i].bse[param_list[i]].round(2)))

    df_years = df_years.groupby([df_years.index, " "]).mean()
    df_years = df_years.astype(str)
    x1 = df_years[[col1, col2, col3]][1:len(df_years):2].apply(
        lambda x: "(" + x + ")")
    x2 = x1.append(df_years[[col1, col2, col3]][0:len(df_years):2]).sort_index()

    return x2

def table_poly_and_bw(dataset, endog, exog, order: int, bandwidth: list):
    df_reg = dataset[["demvoteshare", "lagdemvoteshare", "realada"]].copy()
    share_exog = ["lagdemvoteshare" if "lag" in exog else "demvoteshare"][0]
    df_reg = df_reg.rename(columns={"realada": "score"})
    df_reg[exog] = np.nan
    df_reg.loc[df_reg[share_exog] >= .5, exog] = 1
    df_reg.loc[df_reg[share_exog] < .5, exog] = 0
    index_names = ["one", "two", "three", "four"]
    index_name = [index_names[order-1]]

    formula = f"{endog}~{exog}"
    if order > 1:
        for i in range(2, order + 1):
            df_reg[f"x{i}"] = df_reg[exog] ** i
            formula += f"+x{i}"

    df_table = pd.DataFrame(np.zeros(shape=(2, len(bandwidth))), columns=[str(col) for col in bandwidth],
                            index=index_name*2).sort_index()
    type_list = ["", " "]
    df_table.index = pd.MultiIndex.from_product([index_name, type_list])

    for bw in bandwidth:
        if bw:
            df_reg_bw = df_reg[(df_reg[share_exog] > 0.5 - bw) & (df_reg[share_exog] < 0.5 + bw)].copy()
        else:
            df_reg_bw = df_reg.copy()
        result = smf.ols(formula=formula, data=df_reg_bw).fit()
        df_table._set_value(index=(index_names[order - 1], ""), col=str(bw), value=result.params[exog])
        df_table._set_value(index=(index_names[order - 1], " "), col=str(bw), value=result.bse[exog])

    df_table = df_table.round(2).astype(str)
    x1 = df_table[df_table.columns][1:len(df_table):2].apply(lambda x: "(" + x + ")")
    x2 = x1.append(df_table[df_table.columns][0:len(df_table):2]).sort_index()

    return x2


