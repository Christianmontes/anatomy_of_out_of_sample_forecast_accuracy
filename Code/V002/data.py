import io
import os
import zipfile

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests


class Data:

    def get_nrtz(self):
        """
        NRTZ data
        """

        return None  # unused

        raw = pd.read_excel("../../Data/dataUpdateNRTZ.xlsx")

        df = raw.set_index("date")
        df.sort_index(inplace=True)

        return df

    def get_jwurgler(self):
        """
        http://people.stern.nyu.edu/jwurgler/
        """

        return None  # unused

        raw = pd.read_excel("http://people.stern.nyu.edu/jwurgler/data/"
                            "Investor_Sentiment_Data_20190327_POST.xlsx", sheet_name="DATA", na_values=".")

        raw = raw.set_index(pd.to_datetime(raw["yearmo"], format="%Y%m"))

        df = raw.iloc[:, 1:3]
        df.columns = ["jwurgler_osent", "jwurgler_sent"]

        df.index.name = "date"
        df.sort_index(inplace=True)

        return df

    def get_epu(self):
        """
        https://www.policyuncertainty.com/
        """

        return None  # unused

        raw = pd.read_csv("https://www.policyuncertainty.com/media/All_Daily_Policy_Data.csv")
        dpi = raw.set_index(pd.to_datetime(pd.to_datetime(raw[["year", "month", "day"]])))["daily_policy_index"]

        mpi = dpi.resample("MS").last().rename("epu_didx")  # resample to monthly (start) using last value

        raw = pd.read_excel("https://www.policyuncertainty.com/media/"
                            "Categorical_EPU_Data.xlsx", skipfooter=1,  na_values=0, engine='openpyxl').set_index("Date")

        df = raw.ffill().fillna(0)
        df.columns = ["epu_idx", "epu_monetary", "epu_fiscal", "epu_tax", "epu_govspend",
                      "epu_healthcare", "epu_security", "epu_entitlement", "epu_regulation",
                      "epu_finregulation", "epu_trade", "epu_debtcurrency"]

        raw = pd.read_excel("https://www.policyuncertainty.com/media/"
                            "US_Policy_Uncertainty_Data.xlsx", skipfooter=1, sheet_name="Main Index")
        raw["Day"] = 1
        raw = raw.set_index(pd.to_datetime(pd.to_datetime(raw[["Year", "Month", "Day"]])))

        news = raw["News_Based_Policy_Uncert_Index"].rename("epu_news")
        tci = raw["Three_Component_Index"].rename("epu_tci")

        df = pd.concat((df, mpi, news, tci), axis=1, join="inner")
        df.index.name = "date"
        df.sort_index(inplace=True)

        return df

    def get_sludvigson(self):
        """
        https://www.sydneyludvigson.com/macro-and-financial-uncertainty-indexes
        """

        return None  # unused

        zip = zipfile.ZipFile(io.BytesIO(requests.get("https://www.sydneyludvigson.com/s/"
                                                      "MacroFinanceUncertainty_202108Update.zip").content))

        raw = pd.read_csv(io.BytesIO(zip.read("FinancialUncertaintyToCirculate.csv")))
        fin = raw.set_index(pd.to_datetime(raw["Date"], format="%m/%Y")).iloc[:, 1:]
        fin.columns = ["sludvigson_fin_h1", "sludvigson_fin_h3", "sludvigson_fin_h12"]

        raw = pd.read_csv(io.BytesIO(zip.read("MacroUncertaintyToCirculate.csv")))
        macro = raw.set_index(fin.index).iloc[:, 1:]
        macro.columns = ["sludvigson_macro_h1", "sludvigson_macro_h3", "sludvigson_macro_h12"]

        raw = pd.read_csv(io.BytesIO(zip.read("RealUncertaintyToCirculate.csv")))
        real = raw.set_index(fin.index).iloc[:, 1:]
        real.columns = ["sludvigson_real_h1", "sludvigson_real_h3", "sludvigson_real_h12"]

        df = pd.concat((fin, macro, real), axis=1, join="inner")
        df.index.name = "date"
        df.sort_index(inplace=True)

        return df

    def get_soen(self):
        """
        http://structureofnews.com/
        """

        return None  # unused

        map = {"Profits": "soen_profit", "M&A": "soen_ma", "Savings & loans": "soen_savloan", "IPOs": "soen_ipo",
               "Record high": "soen_high", "Bond yields": "soen_yield", "Small business": "soen_sbusiness",
               "Short sales": "soen_short", "Nonperforming loans": "soen_nloan", "Economic growth": "soen_egrowth",
               "Credit ratings": "soen_crate", "Federal Reserve": "soen_fed", "Job cuts": "soen_job",
               "Problems": "soen_problem", "Financial crisis": "soen_crisis", "Recession": "soen_rec",
               "Earnings losses": "soen_earnings", "Product prices": "soen_prices", "Bankruptcy": "soen_bankr",
               "Major concerns": "soen_concern"}

        raw = pd.read_csv("http://structureofnews.com/data/download/Monthly_Topic_Attention_Theta.csv")
        df = raw[map.keys()].set_index(pd.to_datetime(raw["date"])).rename(columns=map)
        df.sort_index(inplace=True)

        return df

    def get_ism(self):
        """
        https://www.quandl.com/data/ISM-Institute-for-Supply-Management
        """

        return None  # unused

        api_key = os.environ.get("QUANDL_API_KEY", "")  # unused loader; key never shipped
        url = "https://www.quandl.com/api/v3/datasets/ISM/%s.csv?api_key=%s"

        keys = ["BUY_CAP_EXP", "BUY_MRO_SUPP", "BUY_PROD_MAT", "NONMAN_INVSENT", "NONMAN_PRICES", "NONMAN_IMPORTS",
                "NONMAN_BACKLOG", "NONMAN_INVENT", "NONMAN_EXPORTS", "NONMAN_DELIV", "NONMAN_NEWORD", "NONMAN_EMPL",
                "NONMAN_BUSACT", "MAN_IMPORTS", "MAN_INVENT", "MAN_EXPORTS", "MAN_PROD", "MAN_BACKLOG",
                "MAN_NEWORDERS", "MAN_EMPL", "MAN_PRICES", "MAN_CUSTINV", "MAN_DELIV", "NONMAN_NMI", "MAN_PMI"]

        # use average days for buying policy, diffusion index for sentiment, index for indices, pmi for composite index
        columns = ["Average Days", "Diffusion Index", "Index", "PMI"]

        series = []
        for key in keys:
            raw = pd.read_csv(url % (key, api_key))
            series.append(raw.set_index(
                pd.to_datetime(raw["Date"]))[[x for x in columns if x in raw.columns][0]].rename(key.lower()))

        df = pd.concat(series, axis=1)
        df.index.name = "date"
        df.sort_index(inplace=True)

        return df

    def get_soc(self):
        """
        http://www.sca.isr.umich.edu/tables.html
        """

        raw = pd.read_csv("http://www.sca.isr.umich.edu/files/tbmics.csv")
        ics = raw.set_index(raw.apply(lambda x: pd.to_datetime("%s/%i" % (x["Month"], x["YYYY"]), format="%B/%Y"),
                                      axis=1))["ICS_ALL"].rename("soc_ics")

        raw = pd.read_csv("http://www.sca.isr.umich.edu/files/tbmiccice.csv")
        raw = raw.set_index(raw.apply(lambda x: pd.to_datetime("%s/%i" % (x["Month"], x["YYYY"]), format="%B/%Y"), axis=1))
        ice, icc = raw["ICE"].rename("soc_ice"), raw["ICC"].rename("soc_icc")

        df = pd.concat((ics, ice, icc), axis=1)
        df = df.reindex(pd.date_range(df.index.min(), df.index.max(), freq="MS"))  # first years are quarterly
        df.index.name = "date"
        df.sort_index(inplace=True)

        return df

    def get_fred_md_raw(self):
        """
        https://research.stlouisfed.org/wp/more/2015-012
        """

        raw = pd.read_csv("https://files.stlouisfed.org/files/htdocs/fred-md/monthly/current.csv").iloc[1:]
        raw["sasdate"] = pd.to_datetime(raw["sasdate"])

        df = raw.dropna(subset=["sasdate"]).set_index("sasdate")
        df.columns = [x.lower() for x in df.columns]
        df.index.name = "date"
        df.sort_index(inplace=True)

        return df

    def get_fred_md(self):
        """
        Load the FRED-MD data and transform it according to the transformation codes.
        """

        raw = pd.read_csv("https://files.stlouisfed.org/files/htdocs/fred-md/monthly/current.csv")

        transform, raw = raw.iloc[0], raw.iloc[1:]
        transformers = {
            1: lambda x: x,
            2: lambda x: x.diff(1),
            3: lambda x: x.diff(2),
            4: lambda x: np.log(x),
            5: lambda x: np.log(x).diff(1),
            # 6: lambda x: np.log(x).diff(2),
            6: lambda x: np.log(x).diff(1),  # override, we dont want to take diff twice
            # 7: lambda x: x.pct_change().diff(1)
            7: lambda x: np.log(x).diff(1)  # override, we dont want to take diff twice
        }

        for column, key in transform.iloc[1:].iteritems():
            raw[column] = transformers[key](raw[column])

        raw["sasdate"] = pd.to_datetime(raw["sasdate"])

        df = raw.dropna(subset=["sasdate"]).set_index("sasdate")
        df.columns = [x.lower() for x in df.columns]
        df.index.name = "date"
        df.sort_index(inplace=True)

        return df

        #df = df[:"2019-12-31"]
        #df["Y"] = df["CLAIMSx"].shift(-1)
        #df_train, df_test = df[:"2010-01-01"], df["2010-02-01":]
        #return df_train, df_test

    def get_goyal_welch(self):
        """
        Load Goyal-Welch data and derive the 14 predictors.
        """

        return None  # unused

        raw = pd.read_excel("https://drive.google.com/u/0/uc?id=1ACbhdnIy0VbCWgsnXkjcddiV8HF4feWv&export=download")
        raw["date"] = pd.to_datetime(raw["yyyymm"], format="%Y%m")
        raw = raw.set_index("date").drop(["csp", "yyyymm"], axis=1)

        df = pd.DataFrame({
                "y": raw["CRSP_SPvw"],
                "dp": np.log(raw["D12"] / raw["Index"]),
                "dy": np.log(raw["D12"] / raw["Index"].shift(1)),
                "ep": np.log(raw["E12"] / raw["Index"]),
                "de": np.log(raw["D12"] / raw["E12"]),
                "svar": raw["svar"],
                "bm": raw["b/m"],
                "ntis": raw["ntis"],
                "tbl": raw["tbl"],
                "lty": raw["lty"],
                "ltr": raw["ltr"],
                "tms": raw["lty"] - raw["tbl"],
                "dfy": raw["BAA"] - raw["AAA"],
                "dfr": raw["corpr"] - raw["ltr"],
                "infl": raw["infl"].shift(1)
        })

        return df

        #df_train, df_test = df[:"2010-01-01"], df["2010-02-01":]
        #return df_train, df_test
