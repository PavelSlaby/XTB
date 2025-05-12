import pandas as pd
import numpy as np
import datetime
import time
import pathlib
import shutil
import warnings
import os

os.chdir(r'G:\STX v1\14 Risk\Svetozar\GOO Data\strategy code lorenzo')


strategies_trading_src= r"G:\STX v1\23 Trading\Trading Desk\35. Scripts\3. Utilities\Daily PnL\Strategies.xlsx"
main_folder = r'G:\STX v1\14 Risk\Svetozar\GOO Data\strategy code lorenzo'

strategies_pnl_dest = r"G:\STX v1\14 Risk\PNL new projet 2023\GOO Strategy"
strategies_pos_monitoring_dest = r"G:\STX v1\14 Risk\5.8 Risk Overviews\Position Monitoring"


price_history_start = pd.to_datetime('2017-12-31')
prices_history_end = pd.Timestamp.now()
dates_list = pd.DataFrame(index = pd.date_range(price_history_start, prices_history_end)).index


#To check the difference in P&L, we need to check the difference between risk_mid_eur of today minus the risk_mid_eur of yesterday where the realisation_date is blank.
#To get the P&L of the delivered positions we need to sum the transaction_value_eur where the realisation_date is equal to today.
class QueryLoader():
    def __init__(self):
        self.desks = ["goo"]
        self.books = ["Europe", "Benelux", "Switzerland", "Non AIB GoO", "Non Renewables", "REGO", "UK Fit", "Poland", "New Component", "Floating book (GOO)"]
        self.df_proxy_mapping = pd.read_excel("BookData.xlsx", sheet_name = "ProxyMapping")
        self.reporting_currency = "eur"
        self.business_entity_code = "992"

        print("Loading EOD...")
        self.df_eod = pd.read_csv("https://queryinterface.datahive.online/get_data?target_db=dwh&table=eod.positions_eod_split&dwh_current_flg=1&book=in('" + "','".join([desk.replace(" ", "%20") for desk in self.desks]) + "')&columns=position_id,matched_position_id,trade_id,type,proxy_name,proxy_category,volume,price_" + self.reporting_currency + ",production_from_year,production_from_month,labels,countries,trade_date,realisation_date,settled_at,mid_" + self.reporting_currency + "_trade_date,mid_" + self.reporting_currency + "_trade_date_reviewed,request_owner", sep = "\t")   
        self.df_eod[["realisation_date", "trade_date", "settled_at"]] = self.df_eod[["realisation_date", "trade_date", "settled_at"]].apply(pd.to_datetime)
        #Creating the new production year (as for UK products we have CPs instead of calendar year):
        self.df_eod["new_production_from_year"] = self.get_production_year(self.df_eod)
        self.df_eod["real_volume"] = self.df_eod.apply(lambda df: df["volume"] if df["type"] == "long" else -df["volume"], axis = 1)

        print("Loading Proxies...")
        self.df_proxies = pd.read_csv("https://queryinterface.datahive.online/get_data?target_db=dwh&table=reporting.v_proxy_price_history&dwh_business_entity_code=" + self.business_entity_code + "&proxy_category=in('" + "','".join([book.replace(" ", "%20") for book in self.books]) + "')&price_date=greater('2017-12-31')&columns=price_date,proxy_category,proxy_name,currency,mid,is_active", sep = "\t")
        self.df_proxies["price_date"] = pd.to_datetime(self.df_proxies["price_date"]).dt.floor("D")

        self.proxies_list = pd.DataFrame(self.df_proxies.proxy_name.unique()).iloc[:, 0]

        self.df_price_proxy = pd.MultiIndex.from_product([dates_list, self.proxies_list], names=["price_date", "proxy_name"]).to_frame(index=False)
        self.merged_prices = pd.merge(self.df_price_proxy, self.df_proxies , on=["price_date", "proxy_name"], how="left")

        self.merged_prices = self.merged_prices.sort_values(by=['proxy_name', 'price_date'], ascending=[True, True])
        self.merged_prices = self.merged_prices.fillna(method = 'ffill')

        self.df_proxies = self.merged_prices
       
        df_first_trade_date_of_proxy = self.df_eod.sort_values(by = "trade_date", ascending = False).groupby(by = ["proxy_name", "proxy_category"]).tail(1).reset_index(drop = True)
        df_first_date_of_proxy = self.df_proxies.sort_values(by = "price_date", ascending = False).groupby(by = ["proxy_name", "proxy_category"]).tail(1).reset_index(drop = True)

        #Here we check if the first trade of a certain proxy appears before the proxy was created.
        df = pd.merge(df_first_trade_date_of_proxy, df_first_date_of_proxy, on = "proxy_name", how = "left")
        df_missing_prices = pd.DataFrame(columns = ["proxy_name", "price_date"])
        #We loop through the proxies where the first trade date of a proxy is before the creation of the proxy (meaning the proxy was changed backwards) and we create a new row with the alternative proxy assigned, in a way of having a
        #price for each day until the first trade date. The alternative proxies are in the ProxyMapping sheet in the BookData file.
        for i, item in df.loc[(df["trade_date"] < df["price_date"])].iterrows():
            df_missing_prices = pd.concat([df_missing_prices, pd.DataFrame({"proxy_name" : item["proxy_name"], "price_date" : pd.date_range(start = item["trade_date"], end = item["price_date"])})])
        df = pd.merge(df_missing_prices.loc[df_missing_prices["price_date"] > datetime.datetime(2017, 12, 31)], self.df_proxy_mapping, on = "proxy_name", how = "left")
        df = pd.merge(df, self.df_proxies, left_on = ["alternative_proxy", "price_date"], right_on = ["proxy_name", "price_date"], how = "left", suffixes = ("", "_to_delete"))
        self.df_proxies = pd.concat([self.df_proxies, df[["price_date", "proxy_category", "proxy_name", "currency", "mid", "is_active"]].dropna(subset = ["mid"])]).reset_index(drop = True)
 
       
       
        print("Loading FX...")
        self.df_fx = pd.read_csv("https://queryinterface.datahive.online/get_data?target_db=dwh&table=public.dim_fx_rates&dwh_current_flg=1&dwh_business_entity_code=" + self.business_entity_code + "&base_currency=in('" + self.reporting_currency.upper() + "')&quote_currency=in('EUR','SEK','GBP','PLN','CHF','USD','NOK')&rate_date=greater('2017-12-31')&columns=rate_date,base_currency,quote_currency,rate", sep = "\t")
        self.df_fx["rate_date"] = pd.to_datetime(self.df_fx["rate_date"]).dt.floor("D")
        self.df_proxies = pd.merge(self.df_proxies, self.df_fx, left_on = ["price_date", "currency"], right_on = ["rate_date", "quote_currency"], how = "left").drop_duplicates()
        self.df_proxies["mid_" + self.reporting_currency] = self.df_proxies["mid"] * self.df_proxies["rate"]

        # print("Loading Contributions...")
        # #This is the same file as the EOD but has the rows split by contribution. So if a trade has sales trader 1 with 70% contribution and sales traded 2 with 30% contribution, the trade will be split in two rows, while in the EOD it
        # #would be in just one row.
        # self.df_contributions = pd.read_csv("https://queryinterface.datahive.online/get_data?target_db=dwh&table=eod.position_contributions_eod&book=in('" + "','".join([desk.replace(" ", "%20") for desk in self.desks]) + "')&columns=broker,broker_percentage,position_id,trade_id,type,proxy_category,trade_date,realisation_date,volume,price_" + self.reporting_currency + ",mid_" + self.reporting_currency + "_trade_date,production_from_year,production_from_month,marktomarket_mid_value_" + self.reporting_currency, sep = "\t")
        # self.df_contributions["trade_date"] = self.df_contributions["trade_date"].apply(pd.to_datetime)
        # self.df_contributions["real_volume"] = self.df_contributions.apply(lambda df: df["volume"] if df["type"] == "long" else -df["volume"], axis = 1)

        # print("Loading Bid Offer data...")
        # self.df_bid_offer = pd.read_csv("https://queryinterface.datahive.online/get_data?target_db=dwh&table=reporting.v_bid_offer_price&book=('" + "','".join([desk.replace(" ", "%20") for desk in self.desks]) + "')", sep = "\t")

        print("Loading Dataset On Full Offtakes...")
        self.df_fot = pd.read_csv("https://queryinterface.datahive.online/get_data?target_db=nl-trading&table=forecast_vs_actual&book=in('" + "','".join([desk.replace(" ", "%20") for desk in self.desks]) + "')", sep = "\t")

        print("All queries loaded.")


## Pavel: the following function does not seem to be used anywhere
    # def get_daily_eod(self, pos_date):
    #     df_eod = pd.read_csv("https://queryinterface.datahive.online/get_data?target_db=dwh&table=eod.positions_eod_history_" + str(pos_date.year) + "{:02d}".format(pos_date.month) + "{:02d}".format(pos_date.day) + "&dwh_current_flg=1&book=in('" + "','".join(self.desks) + "')&columns=book,position_status,position_id,position_match_id,matched_position_id,type,volume,production_from_year,production_from_month,proxy_category,request_owner,realisation_date,price_" + self.reporting_currency + ",price,currency,trade_date,marktomarket_mid_value_" + self.reporting_currency + ",proxy_name,selection_type,risk_mid_" + self.reporting_currency + ",original_proxy_risk_mid_" + self.reporting_currency + ",transaction_value_" + self.reporting_currency, sep = "\t")
    #     df_eod[["realisation_date", "trade_date"]] = df_eod[["realisation_date", "trade_date"]].apply(pd.to_datetime)
    #     df_eod["new_production_from_year"] = self.get_production_year(df_eod)
    #     df_eod["real_marktomarket_mid_value_" + self.reporting_currency] = df_eod.apply(lambda df: df["marktomarket_mid_value_" + self.reporting_currency] if df["type"] == "long" else -df["marktomarket_mid_value_" + self.reporting_currency], axis = 1)
    #     df_eod["real_volume"] = df_eod.apply(lambda df: df["volume"] if df["type"] == "long" else -df["volume"], axis = 1)
    #     return df_eod

    def get_production_year(self, df):
        uk_books = ["REGO", "UK Fit"]
        pjm_books = ["PJM_C1", "PJM_C2", "PJM_SREC"]
        ge_books = ["GE", "Non_GE"]
        futures_books = ["CA_LCFS", "RGGI", "WCI_CCA", "WCI_WCA"]
        df["new_production_from_year"] = df.apply(lambda df: ("CP" + str(df["production_from_year"] - 1)[2:4] if df["production_from_month"] >= 4 else "CP" + str(df["production_from_year"] - 2)[2:4]) if df["proxy_category"] in uk_books else
                                             (df["production_from_year"] + 1 if df["production_from_month"] >= 6 else df["production_from_month"]) if df["proxy_category"] in pjm_books else
                                             (df["production_from_year"] + 1 if df["production_from_month"] >= 7 else df["production_from_month"]) if df["proxy_category"] in ge_books else
                                             (2000 + int(df["proxy_name"][-2:])) if df["proxy_category"] in futures_books else
                                             df["production_from_year"], axis = 1)
        return df["new_production_from_year"]


class Strategies():
    def __init__(self, query_loader):
        self.query_loader = query_loader
        self.start_date = max(pd.read_excel("StrategiesData.xlsx")["date"]) + datetime.timedelta(days = 1) #datetime.datetime(2022, 4, 1)#
        self.end_date = datetime.datetime.today()
        self.reporting_currency = query_loader.reporting_currency
        self.df_strategies = pd.read_excel("Strategies.xlsx")
        self.df_strategies["date"] = pd.to_datetime(self.df_strategies["date"])        
        self.df_proxies = self.query_loader.df_proxies
        self.df_hist_pnl = pd.read_excel("StrategiesData.xlsx", sheet_name = "Strategies")

        self.labels = {"TUV_SUD_EE" : {"strategy" : "LabelSpread", "substrategy" : "TuvSud-AIB"},
                       "bramiljoval": {"strategy" : "LabelSpread", "substrategy" : "BMV-AIB"}}
        
        self.proxies = {"AT Hydro" : {"strategy" : "CountrySpread", "substrategy" : "Austria-AIB"},
                        "DE Hydro" : {"strategy" : "CountrySpread", "substrategy" : "Germany-AIB"},
                        "DE Wind" : {"strategy" : "CountrySpread", "substrategy" : "Germany-AIB"},
                        "French monthly" : {"strategy" : "CountrySpread", "substrategy" : "France-AIB"},
                        "Spanish Domestic" : {"strategy" : "CountrySpread", "substrategy" : "SpanishDomestic-AIB"}}

    def get_strategy_details(self, proxy, subkey):
        for key in self.proxies.keys():
            if key in proxy:
                return self.proxies[key][subkey]
        return float(nan)

    def generate_strategies_trades(self, start_date):
        #This function generates the trades (for country spread) in the past. Each time there is a trade with a proxy included in self.proxies, a correspondent trade with opposite sign at the Mid of Europe Hydro is created, setting up
        #the spread in the past. We should periodically run this function to add the new country spread trades that happen.
        df = self.query_loader.df_eod
        df = df.loc[(df["proxy_category"] == "Europe") & (df["trade_date"] >= start_date)]
        df["liquid_proxy"] = df["production_from_year"].apply(lambda df: "Europe Hydro " + str(df))
        df = pd.merge(df, self.df_proxies, how = "left", left_on = ["liquid_proxy", "trade_date"], right_on = ["proxy_name", "price_date"], suffixes=('_eod', '_proxies'))
        df = df.dropna(subset = ["price_" + self.reporting_currency, "mid_" + self.reporting_currency])

        df = df.loc[df["proxy_name_eod"].str.contains("|".join(list(self.proxies.keys())), na = False)]
        df["strategy"] = df["proxy_name_eod"].apply(lambda df: self.get_strategy_details(df, "strategy"))
        df["substrategy"] = df["proxy_name_eod"].apply(lambda df: self.get_strategy_details(df, "substrategy"))

        df_premium = df[["type", "proxy_category_eod", "production_from_year", "trade_date", "strategy", "substrategy", "volume", "proxy_name_eod", "price_" + self.reporting_currency]]
        df_standard = df[["type", "proxy_category_eod", "production_from_year", "trade_date", "strategy", "substrategy", "volume", "liquid_proxy", "mid_" + self.reporting_currency]]
        df_standard["type"] = df_standard["type"].apply(lambda df: "short" if df == "long" else "long")
        df_standard.columns = df_premium.columns

        #https://stackoverflow.com/questions/26205922/calculate-weighted-average-using-a-pandas-dataframe
        #https://stackoverflow.com/questions/14529838/apply-multiple-functions-to-multiple-groupby-columns
        df_premium = df_premium.groupby(by = ["type", "proxy_category_eod", "production_from_year", "strategy", "substrategy", "trade_date", "proxy_name_eod"]).apply(lambda df: pd.Series({"volume" : df["volume"].sum(), "wap" : np.average(df["price_" + self.reporting_currency], weights = df["volume"])}, index = ["volume", "wap"])).reset_index()
        df_standard = df_standard.groupby(by = ["type", "proxy_category_eod", "production_from_year", "strategy", "substrategy", "trade_date", "proxy_name_eod"]).apply(lambda df: pd.Series({"volume" : df["volume"].sum(), "wap" : np.average(df["price_" + self.reporting_currency], weights = df["volume"])}, index = ["volume", "wap"])).reset_index()
        df = pd.concat([df_premium, df_standard]).reset_index(drop = True)
        df["real_volume"] = df.apply(lambda df: -df["volume"] if df["type"] == "short" else df["volume"], axis = 1)
        df = df.sort_values(by = ["trade_date", "volume"])
        df = df[["trade_date", "proxy_category_eod", "strategy", "substrategy", "proxy_name_eod", "production_from_year", "real_volume", "wap"]]
        return df

    def get_premiums_of_strategies_with_no_proxy(self):
        #This function calculates the premium / discount we bought the products in self.proxies and self.label, compared to the Hydro price. This also works for products for which we don't have the proxy (example, the Tuv Sud positions
        #when we didn't have the Tuv Sud proxy).
        df = self.query_loader.df_eod
        df = df.loc[df["proxy_category"] == "Europe"]
        df["liquid_proxy"] = df["production_from_year"].apply(lambda df: "Europe Hydro " + str(int(df)))
        df = pd.merge(df, self.df_proxies, how = "left", left_on = ["liquid_proxy", "trade_date"], right_on = ["proxy_name", "price_date"], suffixes=('_eod', '_proxies'))
        df["premium"] = df["price_" + self.reporting_currency] - df["mid_" + self.reporting_currency]
        df = df.dropna(subset = "premium")

        df_eod_proxies = df.loc[df["proxy_name_eod"].str.contains("|".join(list(self.proxies.keys())), na = False)]
        df_eod_proxies["premium_name"] = df_eod_proxies["proxy_name_eod"]

        df_eod_labels = df.loc[(df["labels"].str.contains("|".join(list(self.labels.keys())), na = False)) & (~df["labels"].str.contains("TUV_SUD_EE_", na = False)) & (~df["countries"].str.contains("IS", na = False))]
        df_eod_labels["premium_name"] = df_eod_labels.apply(lambda df: "".join([label for label in list(self.labels.keys()) if label in df["labels"]]) + " " + str(df["production_from_year"]), axis = 1)

        df_eod_tuv_sud_is = df.loc[(df["labels"].str.contains("TUV_SUD_EE", na = False)) & (~df["labels"].str.contains("TUV_SUD_EE_", na = False)) & (df["countries"].str.contains("IS", na = False))]
        df_eod_tuv_sud_is["premium_name"] = df_eod_tuv_sud_is.apply(lambda df: "TUV_SUD_EE IS " + str(df["production_from_year"]), axis = 1)

        df = pd.concat([df_eod_proxies, df_eod_labels, df_eod_tuv_sud_is])
        return df[["position_id", "type", "trade_date", "volume", "premium_name", "premium"]]    

    def get_strategies_eod(self, pos_date):
        df = self.df_strategies.loc[self.df_strategies["date"] <= pos_date]
        df["current_date"] = pos_date
        df = pd.merge(df, self.df_proxies[["price_date", "proxy_name", "mid_" + self.reporting_currency]], how = "left", left_on = ["current_date", "proxy"], right_on = ["price_date", "proxy_name"])
        df["risk_mid_" + self.reporting_currency] = df["position"] * (df["mid_" + self.reporting_currency] - df["price"])
        return df

    def create_historical_pnl(self):
        #This works similarly to the Directional P&L seen before (PnL Class). I create a EOD of the strategies (with get_strategies_eod function), then for each strategy I calculate the difference between risk_mid_eur_t and risk_mid_eur_t_minus_1
        df_final = pd.DataFrame(columns = ["book", "strategy", "substrategy", "year", "proxy", "position_t", "pnl", "type_of_pnl", "date"])
        for n in range(int((self.end_date - self.start_date).days)):
            pos_date_t = self.start_date + datetime.timedelta(days = n)
            pos_date_t_minus_1 = self.start_date + datetime.timedelta(days = n - 1)
            print("Calculating Strategies P&L On " + str(pos_date_t))
            df_t = self.get_strategies_eod(pos_date_t)
            df_t_minus_1 = self.get_strategies_eod(pos_date_t_minus_1)

            #Calculating P&L from new trades on the strategies (Market Making P&L).
            df_new_trades = df_t.loc[(df_t["date"] == pos_date_t), ["book", "strategy", "substrategy", "proxy", "year", "risk_mid_" + self.reporting_currency]].groupby(by = ["book", "strategy", "substrategy", "proxy", "year"]).sum().round().reset_index()
            df_new_trades["type_of_pnl"] = "pnl_new_trades"
            df_new_trades["date"] = pos_date_t
            df_new_trades["position_t"] = 0
            df_new_trades = df_new_trades.rename(columns = {"risk_mid_" + self.reporting_currency : "pnl"})

            #Calculating Unrealized P&L from the Strategies (as difference between risk_mid_eur on t and t_minus_1).
            df_t = df_t.loc[(df_t["date"] != pos_date_t), ["book", "strategy", "substrategy", "proxy", "year", "risk_mid_" + self.reporting_currency, "position"]].groupby(by = ["book", "strategy", "substrategy", "proxy", "year"]).sum().round().reset_index()
            df_t_minus_1 = df_t_minus_1[["book", "strategy", "substrategy", "proxy", "year", "risk_mid_" + self.reporting_currency, "position"]].groupby(by = ["book", "strategy", "substrategy", "proxy", "year"]).sum().round().reset_index()

            df_old_trades = pd.merge(df_t, df_t_minus_1, how = "left", on = ["book", "strategy", "substrategy", "proxy", "year"], suffixes = ("_t", "_t_minus_1")).fillna(0)
            df_old_trades["pnl"] = df_old_trades["risk_mid_" + self.reporting_currency + "_t"] - df_old_trades["risk_mid_" + self.reporting_currency + "_t_minus_1"]
            df_old_trades["type_of_pnl"] = "pnl_unrealized"
            df_old_trades["date"] = pos_date_t

            df_final = pd.concat([df_final, pd.concat([df_new_trades[["book", "strategy", "substrategy", "year", "proxy", "position_t", "pnl", "type_of_pnl", "date"]], df_old_trades[["book", "strategy", "substrategy", "year", "proxy", "position_t", "pnl", "type_of_pnl", "date"]]]).reset_index(drop = True)]).reset_index(drop = True)

        df_final = df_final.loc[df_final[["pnl", "position_t"]].sum(axis = 1) != 0] #Remove rows where P&L and Position is 0
        df_final = df_final[["date", "book", "strategy", "substrategy", "year", "proxy", "type_of_pnl", "pnl", "position_t"]]
        df_final = df_final.rename(columns = {"position_t" : "position"})
        df_final = pd.concat([self.df_hist_pnl, df_final]).reset_index(drop = True)
        return df_final


def main():
    
    
    shutil.copy(strategies_trading_src, main_folder + "\Strategies.xlsx")
  

    
    
    warnings.filterwarnings("ignore")

    query_loader = QueryLoader()

    s = Strategies(query_loader)
    df_strategies = s.create_historical_pnl()
    # df_strategies_with_no_proxy = s.get_premiums_of_strategies_with_no_proxy()#pd.read_excel("StrategiesData.xlsx", sheet_name = "StrategiesAveragePremiums")#
    #s.generate_strategies_trades(datetime.datetime(2019, 4, 1)).to_excel("Strategy_Trades.xlsx")
    writer = pd.ExcelWriter("StrategiesData.xlsx")
    df_strategies.to_excel(writer, sheet_name = "Strategies", index = False)
    # df_strategies_with_no_proxy.to_excel(writer, sheet_name = "StrategiesAveragePremiums", index = False)
    writer.close()
    
    
    shutil.copy(r"StrategiesData.xlsx", strategies_pnl_dest + "\StrategiesData.xlsx")
    shutil.copy(r"StrategiesData.xlsx", strategies_pos_monitoring_dest + "\StrategiesData.xlsx")

if __name__ == "__main__":
    main()
