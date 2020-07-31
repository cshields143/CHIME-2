# -*- coding: utf-8 -*-
"""Package Unittests."""

import unittest
from ..bayesian.normal.models import SEIRModel
from ..bayesian.normal.fitting import fit_norm_to_prior_df
from ..bayesian.normal.util import one_minus_logistic_fcn
from io import StringIO
import pandas as pd
from pandas import date_range
from gvar import gvar
from gvar._gvarcore import GVar
from scipy.stats import expon
from datetime import timedelta
from lsqfit import empbayes_fit

class TestSEIR(unittest.TestCase):
    def test_construction(self):
        model = SEIRModel(fit_columns=['hospital_census', 'vent_census'], update_parameters=lambda _:_)
        self.assertTrue(model.compartments==['susceptible','infected','recovered','exposed'])
        self.assertTrue(model.model_parameters==['dates','initial_susceptible','initial_infected','inital_recovered','beta','gamma','alpha'])
        self.assertTrue(model.fit_columns==['hospital_census','vent_census'])
    def test_fit(self):
        parameters = {PARAMETER_MAP.get(k,k):v for k,v in fit_norm_to_prior_df(pd.read_csv(StringIO(eg_param))).items()}
        data = pd.read_csv(StringIO(eg_ts), parse_dates=['date']).dropna(how='all', axis=1).fillna(0).set_index('date').astype(int)[:-7]
        model = SEIRModel(fit_columns=['hospital_census', 'vent_census'], update_parameters=logistic_social_policy)
        pp = {key:val for key,val in parameters.items() if isinstance(val,GVar)}
        xx = {key:val for key,val in parameters.items() if key not in pp}
        xx['offset'] = int( expon.ppf(0.99, 1 / pp['incubation_days'].mean) )
        pp['logistic_x0'] += xx['offset']
        xx['day0'] = data.index.min()
        xx['dates'] = date_range( xx['day0'] - timedelta(xx['offset']), freq='D', periods=xx['offset'] ).union(data.index)
        for key in ['infected', 'recovered', 'icu', 'vent', 'hospital']:
            xx[f'initial_{key}'] = 0
        pp['initial_exposed'] = ( xx['n_hosp'] / xx['market_share'] / pp['hospital_probability'] )
        xx['initial_susceptible'] -= pp['initial_exposed'].mean
        fit_kwargs = lambda error_infos: dict(
            data=(xx, get_yy(data, hosp_rel=0, vent_rel=0, **({'hosp_min':10,'vent_min':1} if error_infos is None else error_infos))),
            prior=pp,
            fcn=model.fit_fcn
        )
        fit, xx['error_infos'] = empbayes_fit({'hosp_min':10,'vent_min':1}, fit_kwargs)
        self.assertTrue(False)

eg_param = '''param,base,distribution,p1,p2,description
n_hosp,1,constant,,,Number of hospitalized COVID-19 patients on day 1
hosp_prop,0.025,gamma,6.326832789,0.004168888,Prportion of infections requiring hospitalization
ICU_prop,0.45,beta,52.0593112,96.8674197,Proportion of hospitalizations admitted to ICU
vent_prop,0.66,beta,5.224029085,3.078885266,Proportion of ICU patients requiring ventilation
hosp_LOS,12,gamma,95.24265772,0.128844547,Hospital Length of Stay
ICU_LOS,9,gamma,122.6010815,0.104221931,ICU Length of Stay
vent_LOS,1.111111111,gamma,51.11447064,0.451781486,time on vent
mkt_share,0.1178,constant,,,Hospital Market Share (%)
region_pop,1200000,constant,,,Regional Population
incubation_days,5,gamma,9.514379271,0.513980244,Days from exposure to infectiousness
recovery_days,14,gamma,9.833457434,1.642265575,Days from infection to recovery
logistic_k,1,gamma,4.018953794,0.22738215,logistic growth rate
logistic_x0,14,gamma,6.407435434,2.859728136,logistic days from beginning of time series to middle of logistic
logistic_L,0.5,beta,2,3,logistic depth of social distancing
nu,2.5,gamma,93.9552169,0.02634306,Networked contact structure power-law exponent
beta,0.25,beta,5,10,SEIR beta parameter (force of infection)
hosp_capacity,,constant,,,Hospital Bed Capacity
vent_capacity,196,constant,,,Ventilator Capacity
beta_spline_dimension,5,constant,,,number of splines for beta
beta_spline_power,1,constant,,,polynomial of the truncated power spline
beta_spline_prior,0,normal,0,10,prior on spline terms. Variance of splines is the inverse of an L2 penalty.
b0,4,normal,-5,2.5,"This is the intercept on the mean of the logistic. It should be large and negative such that 1-logistic(b0+XB) is close to one when X is zero, because the (1-sd) is a coef on beta"
'''

eg_ts = '''date,hosp,vent
3/9/20,2,1
3/10/20,3,1
3/11/20,3,1
3/12/20,3,1
3/13/20,5,1
3/14/20,5,1
3/15/20,5,1
3/16/20,5,1
3/17/20,5,1
3/18/20,5,1
3/19/20,5,1
3/20/20,4,1
3/21/20,4,1
3/22/20,5,1
3/23/20,5,1
3/24/20,7,1
3/25/20,8,1
3/26/20,12,3
3/27/20,14,4
3/28/20,22,6
3/29/20,26,8
3/30/20,33,9
3/31/20,43,11
4/1/20,55,14
4/2/20,64,17
4/3/20,69,22
4/4/20,68,24
4/5/20,72,28
4/6/20,79,29
4/7/20,77,31
4/8/20,78,35
4/9/20,82,35
4/10/20,83,38
4/11/20,82,38
4/12/20,83,37
4/13/20,82,38
4/14/20,85,38
4/15/20,86,42
4/16/20,94,39
4/17/20,100,39
4/18/20,94,36
4/19/20,94,35
4/20/20,98,34
4/21/20,100,36
4/22/20,101,36
4/23/20,101,33
4/24/20,102,28
4/25/20,97,28
4/26/20,98,30
4/27/20,100,31
4/28/20,94,27
4/29/20,94,26
4/30/20,89,25
5/1/20,87,24
5/2/20,81,24
5/3/20,80,21
5/4/20,81,22
5/5/20,77,22
5/6/20,75,23
5/7/20,68,21
5/8/20,68,21
5/9/20,64,21
5/10/20,67,16
5/11/20,65,18
5/12/20,65,18
5/13/20,61,21
5/14/20,68,22
5/15/20,70,24
5/16/20,65,22
5/17/20,64,23
5/18/20,62,21
5/19/20,60,22
5/20/20,54,22
5/21/20,50,18
5/22/20,51,17
5/23/20,47,17
5/24/20,43,17
5/25/20,39,17
5/26/20,35,14
5/27/20,36,15
5/28/20,34,12
5/29/20,34,12
5/30/20,36,12
5/31/20,32,12
6/1/20,33,12
6/2/20,30,11
6/3/20,33,11
6/4/20,30,10
6/5/20,26,10
6/6/20,28,10
6/7/20,27,9
6/8/20,28,8
6/9/20,27,8
6/10/20,30,11
6/11/20,29,10
6/12/20,24,9
6/13/20,24,10
6/14/20,27,10
6/15/20,27,10
6/16/20,22,7
6/17/20,20,
6/18/20,15,
6/19/20,15,
6/20/20,15,
6/21/20,20,
6/22/20,20,
6/23/20,14,
6/24/20,13,
6/25/20,11,
6/26/20,12,
6/27/20,12,
6/28/20,12,
6/29/20,12,
6/30/20,12,
7/1/20,11,
7/2/20,11,
7/3/20,8,
7/4/20,8,
7/5/20,10,
7/6/20,9,
7/7/20,8,
7/8/20,9,
7/9/20,10,
7/10/20,8,
7/11/20,11,
7/12/20,13,
7/13/20,11,
7/14/20,10,
7/15/20,4,'''

PARAMETER_MAP = {
    "hosp_prop": "hospital_probability",
    "hosp_LOS": "hospital_length_of_stay",
    "ICU_prop": "icu_probability",
    "ICU_LOS": "icu_length_of_stay",
    "vent_prop": "vent_probability",
    "vent_LOS": "vent_length_of_stay",
    "region_pop": "initial_susceptible",
    "mkt_share": "market_share",
}

def logistic_social_policy(date, **kwargs):
    xx = (date - kwargs['dates'][0]).days
    ppars = kwargs.copy()
    ppars['beta'] = kwargs['beta'] * one_minus_logistic_fcn(xx, kwargs['logistic_L'], kwargs['logistic_k'], kwargs['logistic_x0'])
    return ppars

def get_yy(data, **err):
    return gvar([data['hosp'].values, data['vent'].values],
                [[data['hosp'].values * err['hosp_rel'] + err['hosp_min']],
                 [data['vent'].values * err['vent_rel'] + err['vent_min']]])

if __name__ == '__main__':
    unittest.main()
