"""
A package for fitting a SARIMA on non-Gaussian data
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from statsmodels.tsa.stattools import adfuller
from myfuncs.data_handling import *
from myfuncs.distributions import *
import numba
import statsmodels.stats.diagnostic
import statsmodels.tsa as stats
import random
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import warnings



@numba.jit(numba.float64[:](numba.float64[:],numba.float64[:], numba.bool_, numba.int64, numba.int64[:], numba.bool_), nopython=True)
def nl_ar_filter(params, endog, constant, pmax, p, return_signal):
    #params = [ar_params*, k, intercept(if intercept in model)]
    ar_params = params[:-1] #this should be determined dynamically

    k = params[0]
    intercept = params[-1] if constant else 0
    N = len(endog)

    ar_param_vec = np.atleast_2d(np.zeros(pmax)) if len(p) > 0 else np.atleast_2d(np.empty(0))
    if len(p) > 0:
        for i, lag in enumerate(p):
            ar_param_vec[0, lag - 1] = ar_params[i]

    signal = np.zeros(N) * np.nan
    innovations = np.zeros(N)
    for i in range(pmax - 1, N - 1):
        endog_lags = np.ascontiguousarray(np.flipud(np.atleast_2d(endog[i - (pmax - 1):i + 1]).T))  # p-last observations
        ar_param_vec = -2/(1+np.exp(-k*endog_lags**2)) +1 #gamma formula
        ar_part = ar_param_vec.dot(endog_lags)
        signal[i + 1] = ar_part[0,0] + intercept
        innovations[i + 1] = endog[i + 1] - signal[i + 1]

    return signal if return_signal else innovations


class sarima_fs:
    def __init__(self, endog, mod_order, D, s=12, constant=True):
        self.p, self.d, self.q = mod_order
        self.k = len(self.p) + len(self.q) + (1 if constant else 0)
        self.mod_order = mod_order
        self.D = D
        self.constant = constant
        self.s = s
        self.data = endog
        diff_data = self.data.copy()

        if self.d > 0:
            diff_data = diff_data.diff(self.d)  # apply differencing if necessary
        if D > 0:
            diff_data = seas_diff(diff_data, s, D)  # apply seasonal differencing if necessary

        self.diff_data = diff_data[(self.d + D * s):]  # differenced series without NaNs
        self.stationarity = adfuller(self.diff_data.replace([np.inf, -np.inf], np.nan).dropna(), regression=(
            'c' if constant else 'nc'))  # check unit root of differenced series
        self.mod_p = self.mod_order[0]
        self.mod_q = self.mod_order[2]

    def fit(self, criterion='errors2', LASSO_lambda=None, bounds=(-1, 1), cbound=1000, ftol=1e-9, LASSO_weights=None, t_degr_freedom = 4):
        # fits a model based on a criterion, options: 'errors2' (squared errors), 'abserrors' (absolute errors)..
        # MNA may add h, so the model will be fitted by optimizing the criterion for horizon h and levels errors:
        class fit:
            def __init__(self, mdl, criterion, LASSO_lambda, bounds, cbound, ftol=ftol, LASSO_weights=LASSO_weights, t_degr_freedom=t_degr_freedom):
                global CONSTANT_BOUNDS
                lpb, upb = bounds[0], bounds[1]
                pmax = np.max(mdl.mod_p) if len(mdl.mod_p) > 0 else 0
                qmax = np.max(mdl.mod_q) if len(mdl.mod_q) > 0 else 0
                self.pmax = pmax
                self.qmax = qmax

                pqmax = np.max([pmax, qmax])
                self.pqmax = pqmax
                self.start_diff = mdl.d + mdl.D * mdl.s
                start = pmax + self.start_diff
                self.start = start
                self.d = mdl.d
                self.D = mdl.D
                self.data = mdl.data
                self.diff_data = mdl.diff_data
                self.s = mdl.s
                self.constant = mdl.constant
                self.n_units = len(self.data) - start
                self.n = len(self.diff_data[self.start_diff:])
                self.k = mdl.k

                # criterion functions
                if type(LASSO_weights) == 'NoneType':
                    LASSO_weights = np.ones(k)


                # re-type some of the attributes for efficiency
                pmax, qmax = np.int64(pmax), np.int64(qmax)
                diff_data_array = np.array(self.diff_data)[:]
                p, q = np.array(mdl.p).astype('int64'), np.array(mdl.q).astype('int64')


                # Define function that produces signal given params
                self.filter_func_inv = lambda params: arma_filter(params, diff_data_array, mdl.constant, pmax, qmax,
                                                                  p, q, True)

                # Define function that produces residuals (including initial values)
                self.error_func = lambda params: arma_filter(params, diff_data_array, mdl.constant, pmax, qmax,
                                                             p, q, False)

                if criterion == 'errors2':
                    q_func = lambda params: np.mean(self.error_func(params)[-self.n_units:] ** 2)  # arma filter on differenced series, squared errors

                elif criterion == 'student-t':
                    # maximizes student t likelihood 4 degr of freedom, can easily make dependent om degrees of freedom(first passed parameter)\
                    # and skewness parameter  (second passed parameter)
                    #the log likelihood function
                    q_func = lambda params: sum(-np.log(x) for x in map(student_t(nu=t_degr_freedom, mu=0, sigma=params[-1]).pdf, self.error_func(params[:-1])[-self.n_units:]))
                    # the nuisance parameter sigma is also estimated

                elif criterion == 'ZINB':
                    # score function that maps states to a score series\

                    # first define a link function:
                    s, self.states = log_trans(self.diff_data.values, c=1)
                    self.s = s

                    self.state_filter_func_inv = lambda params: arma_filter(params,
                                                                            np.log((self.diff_data.values + 1) / s),
                                                                            mdl.constant, pmax, qmax,
                                                                            p, q, True)

                    self.score_func = lambda params: map(lambda x, mu: ZINB().score(x, params[-2], params[-1], mu),
                                                         self.diff_data,
                                                         inv_log_trans(self.state_filter_func_inv(params[:-2]), s))

                    # work around, set score to zero by minimizing the square of the estimated score

                    q_func = lambda params: -np.mean(list(map(lambda x, mu: ZINB().log_L(x, params[-2], params[-1], mu), self.diff_data[-self.n_units:], inv_log_trans(self.state_filter_func_inv(params[:-2]), s)[-self.n_units:]  )))
                    # the nuisance parameters alpha, and pi are also estimated
                    print()


                elif criterion == 'errors2_LASSO':
                    q_func = lambda params: np.mean(self.error_func(params)[-self.n_units:] ** 2) + LASSO_lambda * np.sum(np.abs(
                        params[:(-1 if mdl.constant else None)].dot(
                            LASSO_weights.T)))  # arma filter on differenced series, squared errors

                elif criterion == 'abserrors':
                    q_func = lambda params: np.mean(np.abs(self.error_func(params)[-self.n_units:]))  # arma filter on differenced series, absolute errors

                # parameter estimation
                n_params = mdl.k + (1 if criterion == 'student-t' else 0) + (2 if criterion == 'ZINB' else 0)

                bounds = [(lpb, upb)] * n_params if not self.constant else [(lpb, upb)] * (mdl.k - 1) +[(-cbound,cbound)]

                if criterion == 'student-t':
                    bounds += [(1e-6,1e+6)] # bounds for sigma

                if criterion == 'ZINB':
                    bounds+= [(1e-3,1 - 1e-3),(1e-2,5)] # bounds for pi and alpha

                res = minimize(q_func, x0=np.ones(
                    [n_params])*.5,
                               method='L-BFGS-B', bounds=bounds, options={'ftol': ftol})

                if criterion == 'student-t':
                    param_ests = res.x[:-1]
                    self.error_var_ml = res.x[-1]**2

                elif criterion == 'ZINB':
                    param_ests = res.x[:-2]
                    self.pi_est = res.x[-2]
                    self.alpha_est = res.x[-1]

                else:
                    param_ests = res.x

                self.params = param_ests

                # Define function that produces signal given new data
                if criterion == 'ZINB':
                    self.filter_func = lambda data: inv_log_trans(arma_filter(self.params, np.log((data+1)/self.s), mdl.constant, pmax, qmax,
                                                            p, q, True), s)

                    self.signal = self.filter_func(self.diff_data.values)

                    #self.signal = inv_log_trans(self.state_filter_func_inv(self.params), self.s)


                else:
                    self.filter_func = lambda data: arma_filter(self.params, data, mdl.constant, pmax, qmax,
                                                                p, q, True)

                    self.signal = self.filter_func_inv(self.params)

                self.resids = self.error_func(self.params)
                self.error_var = np.var(self.resids[-self.n_units:]) # variance of the residuals


            def calc_AIC(self, obs_density = 'errors2'):
                '''
                :param obs_density: string or function that takes residual value as input
                :return:
                '''
                # Returns AIC based on some observation density
                if type(obs_density) == str:
                    # use some built in density
                    if obs_density == 'errors2':
                        obs_density = normal(0, np.sqrt(self.error_var)).pdf # function that takes residual as input
                        log_L = sum([np.log(x) for x in map(obs_density, self.resids[-self.n_units:])])
                        AIC_val = 2 * self.k - 2 * log_L

                    elif obs_density == 'student-t':
                        obs_density = student_t(nu=t_degr_freedom, mu=0, sigma=self.error_var_ml).pdf
                        log_L = sum([np.log(x) for x in map(obs_density, self.resids[-self.n_units:])])
                        AIC_val = 2 * (self.k + 1) - 2 * log_L

                    elif obs_density == 'ZINB':
                        log_L = sum(map(lambda x, mu: ZINB().log_L(x, self.pi_est, self.alpha_est, mu), self.diff_data[-self.n_units:], self.signal[-self.n_units:]))
                        AIC_val = 2 * (self.k + 2) - 2 * log_L



                return AIC_val

            def plot_errors_dist(self):
                sigma = np.sqrt(self.error_var)
                mu = 0
                count, bins, ignored = plt.hist(self.resids[-self.n_units:], 30, density=True)
                plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *
                         np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)),
                         linewidth=2, color='r')
                plt.show()


            def resid_autocorr(self):
                # residual autocorrelation
                lb_stat, lb_p = statsmodels.stats.diagnostic.acorr_ljungbox(self.resids, lags=24, boxpierce=False,
                                                                            model_df=mdl.k, period=None, return_df=False)

                return lb_p[-1]<0.1 # True if auto correlation is significant at 24 lags and 10 percent



            def forecast_h(self, h, start=len(self.data), predict_diff=False):
                # predicts values h-steps ahead
                # start is the last observed value e.g. '2018-02-02' forecasts h periods after that date
                # the functions partitions the forecast in the forecast of the differenced data and that of
                # the levels data
                # if forecast of diff==True, the forecast of the differenced series is returned
                #if type(start) == int:
                    #start=start+1

                forecast = self.data.copy().iloc[:start]
                forecast_diff = self.diff_data.copy().iloc[:start-(self.d + self.D * self.s)]
                if self.pqmax == 0 and len(forecast_diff) == 0: #MNA dit is omslachtig
                    forecast_diff = self.diff_data.copy().iloc[:h]
                    if self.constant:
                        forecast_diff[:] = self.params[0]
                    else:
                        forecast_diff = 0


                for i in range(h):
                    forecast_diff = forecast_diff.append(pd.Series(np.nan, index=[forecast_diff.index[-1] + 1])) # create a NaN value to be filled
                    forecast_diff[-1] = \
                    self.filter_func(np.array(forecast_diff)[:])[-1] if self.pqmax > 0 else (self.params[-1] if self.constant else 0)
                    # fill it with the last value of the filter applied to the last few values

                for i in range(h):
                    forecast = forecast.append(pd.Series(np.nan, index=[forecast.index[-1] + 1]))  # create a NaN value to be filled with index = current index + 1 period
                    forecast[-1] = forecast_diff[
                        forecast.index[-1]]  # fill the value with the forecasted difference
                    forecast[-1] += (forecast[-(self.s+1)] if self.D else 0) + (forecast_diff[-2] if self.D * self.d else 0)\
                                    + (forecast[-2] if self.d * (1-self.D) else 0)  # if necessary, add (seasonal) lags

                return forecast

            def predict_in_sample_h(self, h, predict_diff=False):
                # the filter function allready gives us the in sample h=1 forecast of the diff_data
                if h==1 and (predict_diff or (self.d + self.D == 0)):
                    return pd.Series(data=self.signal[-self.n_units:], index=self.diff_data[-self.n_units:].index)

                # for other values of h produce the prediction iteratively
                prediction_h = self.data.copy()
                prediction_h[:] = np.nan
                start_forecast = self.start + h - 1  # first value that should/can be forecasted in sample
                for j in range(start_forecast, len(self.data)):
                    forecast = self.forecast_h(h, start=j+1 - h)
                    prediction = forecast[-1]  # fill the forecast h periods ahead
                    index = forecast.index[-1]
                    # index = forecast.index.values[-1]  # fill the forecast h periods ahead
                    prediction_h[index] = prediction
                return prediction_h

            def fit_error_dist(self):
                resids = self.resids[-self.n_units:]
                var_resids= np.var(resids)
                std_dev_resids = np.sqrt(var_resids)
                std_resids = self.resids/std_dev_resids
                self.scaled_error_dist = fit_skew(std_resids)
                self.draw_error = lambda: self.scaled_error_dist.ppf(np.random.uniform(0,1)) * std_dev_resids

            def simulate_h(self, h, start=len(self.data), predict_diff=False):
                # simulates values h-steps ahead
                # start is the last observed value e.g. '2018-02-02' forecasts h periods after that date
                # the functions partitions the forecast in the forecast of the differenced data and that of
                # the levels data
                # if forecast of diff==True, the forecast of the differenced series is returned
                # if type(start) == int:
                # start=start+1

                forecast = self.data.copy().iloc[:start]
                forecast_diff = self.diff_data.copy().iloc[:start - (self.d + self.D * self.s)]
                if self.pqmax == 0 and len(forecast_diff) == 0:
                    forecast_diff = self.diff_data.copy().iloc[:h]
                    if self.constant:
                        forecast_diff[:] = self.params[0]
                    else:
                        forecast_diff = 0

                else:
                    for i in range(h):
                        forecast_diff = forecast_diff.append(
                            pd.Series(np.nan, index=[forecast_diff.index[-1] + 1]))  # create a NaN value to be filled
                        forecast_diff[-1] = \
                            self.filter_func(np.array(forecast_diff)[:])[-1] if self.pqmax > 0 else (
                                self.params[-1] if self.constant else 0)
                        # fill it with the last value obtained from the filter applied to the last few values

                        # add a forecast error:
                        forecast_diff[-1] += self.draw_error()


                for i in range(h):
                    forecast = forecast.append(pd.Series(np.nan, index=[forecast.index[
                                                                            -1] + 1]))  # create a NaN value to be filled with index = current index + 1 period
                    forecast[-1] = forecast_diff[
                        forecast.index[-1]]  # fill the value with the forecasted difference
                    forecast[-1] += (forecast[-(self.s + 1)] if self.D else 0) + (
                        forecast_diff[-2] if self.D * self.d else 0) \
                                    + (forecast[-2] if self.d * (
                                1 - self.D) else 0)  # if necessary, add (seasonal) lags

                return forecast

            def simulate_in_sample_h(self, h):
                # produce in sample realisations based on paths of length h from the last observed value
                prediction_h = self.data.copy()
                prediction_h[:] = np.nan
                start_forecast = self.start + h - 1  # first value that should/can be forecasted in sample
                for j in range(start_forecast, len(self.data)):
                    forecast = self.simulate_h(h, start=j+1 - h)
                    prediction = forecast[-1]  # fill the forecast h periods ahead
                    index = forecast.index[-1]
                    # index = forecast.index.values[-1]  # fill the forecast h periods ahead
                    prediction_h[index] = prediction
                return prediction_h

            def R2_adj(self, R2_diff_h1=True, mean_assumption=True, h=1):
                # calculates and returns the adjusted R2 of the data
                # if R2_diff==True, the adjusted R2 of the fit on the differenced data is returned
                # if R2_diff==False, the adjusted R2 for the levels data is returned
                # if mean_assumption==False, R2 will be calculated without assuming that the data has mean zero if
                # there is no constant in the model

                if R2_diff_h1 == True:
                    resids = self.resids
                    data = self.diff_data
                    start = self.pqmax

                else:
                    start = - ( self.n_units - h + 1 )
                    data = self.data
                    resids = data - self.predict_in_sample_h(h)
                    resids = resids


                if self.constant or not mean_assumption:
                    self.TSS_adj = np.sum(
                        (data[start:] - np.mean(data[start:])) ** 2) / (
                                           len(data[start:]) - 1)

                else:
                    self.TSS_adj = np.sum(data[start:] ** 2) / (
                        len(data[start:]))

                self.RSS_adj = np.sum(resids[start:] ** 2) / (len(resids[start:]) - (len(self.params) + self.d + self.D)) # also apply degrees of correction for differencing

                R2_adj = 1 - self.RSS_adj / self.TSS_adj

                return R2_adj

        return fit(mdl=self, criterion=criterion, LASSO_lambda=LASSO_lambda, bounds=bounds, cbound=cbound , ftol=ftol,
                   LASSO_weights=LASSO_weights, t_degr_freedom=t_degr_freedom)