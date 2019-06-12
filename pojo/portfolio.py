import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA as arima
from scipy.stats import gamma,norm
from collections import defaultdict
import statsmodels.api as sm

from numpy import linalg
def equal(df,key,val):
    return df[df[key]==val]
def notequal(df,key,val):
    return df[df[key]!=val]
def greater(df,key,val):
    return df[df[key]>val]
def smaller(df,key,val):
    return df[df[key]<val]
def isin(df,key,val):
    return df[df[key].isin(val)]
pd.DataFrame.equal = equal
pd.DataFrame.notequal = notequal
pd.DataFrame.greater = greater
pd.DataFrame.smaller = smaller
pd.DataFrame.isin = isin


class BlackLittermanBayesian():
    def __init__(self, df, test_start_date,valid_company):
        self.df = df[df['NAME'].isin(valid_company)]
        self.q = pd.DataFrame()
        self.ft = pd.DataFrame()
        self.D = defaultdict(list)
        self.test_start_date = test_start_date
        self.theta_set = defaultdict(list)
        self.var_set = defaultdict(list)
        self.omega = defaultdict(list)
        self.date_index = {v: k for k, v in dict(enumerate(df['T'].unique())).items()}

    def feature_seperator(self, input_):
        X = input_[['BETA', 'MOMENTUM', 'SIZE', 'VOL']].values
        r = input_[['R']].values
        return X, r

    # get factor returns for all days in backtest period
    def set_ft(self):
        ft_ = []
        dates_index = []
        for dates in self.df['T'].unique():
            cur_df = self.df[self.df['T'] == dates]
            cur_X, cur_r = self.feature_seperator(cur_df)
            ft_.append(np.array(np.linalg.inv(cur_X.T @ cur_X) @ cur_X.T @ cur_r.flatten()))
            dates_index.append(dates)
        self.ft = pd.DataFrame(ft_, columns=['BETA', 'MOMENTUM', 'SIZE', 'VOL'])
        self.ft['DATE'] = dates_index

    # set residuals for each stock for all days
    def set_D(self):
        for name in self.df['NAME'].unique():
            cur_company = self.df.equal('NAME', name)
            X, r = self.feature_seperator(cur_company)
            model = sm.OLS(r, X)
            results = model.fit()
            var = pd.DataFrame({'residue': results.resid}).ewm(halflife=50).var().fillna(0)
            self.D[name] = var.values.flatten()

    # important! q_pred is our input for portfolio construction
    # here we use ARIMA(1,0,0) (best fitted) to predic next day risk premia factors change
    def set_q_pred(self):
        forecast_df = pd.DataFrame()
        forecast_df['BETA'] = self.df.groupby('T')['BETA'].mean()
        forecast_df['MOMENTUM'] = self.df.groupby('T')['MOMENTUM'].mean()
        forecast_df['SIZE'] = self.df.groupby('T')['SIZE'].mean()
        forecast_df['VOL'] = self.df.groupby('T')['VOL'].mean()

        train = forecast_df[forecast_df.index <= self.test_start_date]
        test = forecast_df[forecast_df.index > self.test_start_date]
        q_frame = {'BETA': [], 'MOMENTUM': [], 'SIZE': [], 'VOL': [], 'DATE': forecast_df.index[1:]}

        for factors in ['BETA', 'MOMENTUM', 'SIZE', 'VOL']:
            model = arima(train[factors].values, (1, 0, 0))
            model_fit = model.fit()
            q_frame[factors] = list(model_fit.predict())
            for i in range(1, len(test)):
                model = arima(list(train[factors].values) + list(test[factors][:i].values), (1, 0, 0))
                model_fit = model.fit()
                q_frame[factors].append(model_fit.forecast()[0][0])
        self.q = pd.DataFrame(q_frame)

    # normal-gamma fitter for factor returns, tho actually following a power law like distribution
    def estimator(self, input_):
        input_ = input_.values if len(input_.T) == 1 else input_.iloc[:, 0] * input_.iloc[:, 1].values
        mu = 0;
        k = 0;
        alpha = 1 / 2;
        beta = np.inf
        # posterior def
        mu1 = np.mean(input_)
        k1 = len(input_)

        alpha1 = -1 / 2 + k1 / 2
        beta1 = 1 / (0.5 * np.var(input_))
        spread1 = np.sqrt(1 / (k1 * alpha1 * beta1))
        rv = gamma.rvs(alpha1, mu1, beta1, k1)
        var = np.mean(1 / rv)
        theta = np.mean(norm.rvs(mu1, 1 / np.sqrt(k1 * rv)))
        return theta, var

    # set residual terms
    def set_sv_omega(self):
        count = 2
        for dates in self.df['T'].unique():
            columns_ = list(self.ft.columns)
            columns_.remove('DATE')
            for factors in columns_:
                theta, var = self.estimator(self.ft[[factors]][:count])
                self.theta_set[dates].append(theta)
                self.var_set[dates].append(var)
                self.omega[dates].append(np.sqrt(var))
            for fi in range(0, len(columns_) - 1):
                for fj in range(fi + 1, len(columns_)):
                    perm = [columns_[fi], columns_[fj]]
                    theta, var = self.estimator(self.ft[perm][:count])
                    self.var_set[dates].append(theta - self.theta_set[dates][fi] * self.theta_set[dates][fj])
            count += 1

    def get_Dt(self, name_list, date):
        dt_ = []
        for i in name_list:
            dt_.append(self.D[i][self.date_index[date]])
        return np.eye(len(name_list)) * dt_

    def get_sig(self, date):
        name_list = self.df.equal('T', date)['NAME']
        r_t = self.df.equal('T', date)['R']
        X_t = self.df.equal('T', date)[['BETA', 'MOMENTUM', 'SIZE', 'VOL']].as_matrix()
        F_t = self.ft.equal('DATE', date)[['BETA', 'MOMENTUM', 'SIZE', 'VOL']].values.reshape(-1, 1) * np.eye(4)
        D_t = self.get_Dt(name_list, date)
        return X_t, r_t, X_t @ F_t @ X_t.T + D_t, name_list

    def convert_var_mat(self, v):
        mat = np.eye(4) * v[:4]
        for i in range(0, 3):
            for j in range(i + 1, 4):
                mat[i][j] = v[3 + i + j] if i < 2 else v[3 + i + j + 1]
                mat[j][i] = v[3 + i + j] if i < 2 else v[3 + i + j + 1]
        return mat

    def get_risk_premia(self, date):
        v = self.convert_var_mat(self.var_set[date])
        mu = np.array(self.theta_set[date]).reshape(-1, 1)
        o = self.omega[date] * np.eye(4)
        q = self.q.equal('DATE', date)[['BETA', 'MOMENTUM', 'SIZE', 'VOL']].values.reshape(-1, 1)
        x_t, r_t, sig_t, name_list = self.get_sig(date)
        fac_ = x_t.T @ linalg.inv(sig_t) @ x_t
        var_ = linalg.inv(v) @ mu
        theta_ = linalg.inv(o) @ q

        return linalg.inv(linalg.inv(v) + linalg.inv(o) + fac_) @ (var_ + theta_), x_t, r_t, sig_t, name_list

    def get_optimal_holding(self, date):
        risk_prem, x_t, r_t, sig_t, name_list = self.get_risk_premia(date)
        h_blb = -0.2 * linalg.inv(sig_t) @ x_t @ risk_prem * 1000
        return h_blb, r_t, name_list

    def get_benchmark_holding(self, date):
        mu_f = np.mean(self.ft.smaller('DATE', date)).values.reshape(-1, 1)
        x_t, rt, sig_t, name_list = self.get_sig(date)
        h_b = -20 * linalg.inv(sig_t) @ x_t @ mu_f * 1000
        return h_b, rt, name_list
