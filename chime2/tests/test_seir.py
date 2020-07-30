# -*- coding: utf-8 -*-
"""Package Unittests."""
import unittest, os
from ..bayesian.normal.models import SEIRModel
from gvar._gvarcore import GVar
from gvar import gvar
from scipy.stats import expon
from pandas import date_range
import pandas as pd
from ..bayesian.normal.util import one_minus_logistic_fcn
from ..bayesian.normal.fitting import fit_norm_to_prior_df
from lsqfit import empbayes_fit
from datetime import timedelta

class TestSEIR(unittest.TestCase):
    def build_model(self):
        model = SEIRModel(
            fit_columns=['hospital_census', 'vent_census'],
            update_parameters=self.logistic_social_policy
        )
        return model
    def init_params(self):
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
        parameters = {PARAMETER_MAP.get(key,key):val for key,val in fit_norm_to_prior_df(pd.read_csv('chime2/data/HUP_parameters.csv')).items()}
        data = pd.read_csv('chime2/data/HUP_ts.csv', parse_dates=['date']).dropna(how='all', axis=1).fillna(0).set_index('date').astype(int)
        test_set = data[-7:]
        data = data[:-7]
        return parameters, data, test_set
    def prepare_params(self, parameters, data):
        pp = {key:val for key,val in parameters.items() if isinstance(val,GVar)}
        xx = {key:val for key,val in parameters.items() if key not in pp}
        xx['offset'] = int(
            expon.ppf(0.99, 1 / pp['incubation_days'].mean)
        )
        pp['logistic_x0'] += xx['offset']
        xx['day0'] = data.index.min()
        xx['dates'] = date_range(
            xx['day0'] - timedelta(xx['offset']), freq='D', periods=xx['offset']
        ).union(data.index)
        for key in ['infected', 'recovered', 'icu', 'vent', 'hospital']:
            xx[f'initial_{key}'] = 0
        pp['initial_exposed'] = (
            xx['n_hosp'] / xx['market_share'] / pp['hospital_probability']
        )
        xx['initial_susceptible'] -= pp['initial_exposed'].mean
        return xx, pp
    def logistic_social_policy(self, date, **kwargs):
        xx = (date - kwargs["dates"][0]).days
        ppars = kwargs.copy()
        ppars["beta"] = kwargs["beta"] * one_minus_logistic_fcn(
            xx, L=kwargs["logistic_L"], k=kwargs["logistic_k"], x0=kwargs["logistic_x0"],
        )
        return ppars
    def get_yy(self, data, **err):
        return gvar(
            [data['hosp'].values, data['vent'].values],
            [
                data['hosp'].values * err['hosp_rel'] + err['hosp_min'],
                data['vent'].values * err['vent_rel'] + err['vent_min']
            ]
        ).T
    def build_fit(self, data, xx, pp, m):
        fit_kwargs = lambda error_infos: dict(
            data=(xx, self.get_yy(data, hosp_rel=0, vent_rel=0, **({'hosp_min':10,'vent_min':1} if error_infos is None else error_infos))),
            prior=pp,
            fcn=m.fit_fcn
        )
        fit, xx['error_infos'] = empbayes_fit(
            {'hosp_min':10,'vent_min':1}, fit_kwargs
        )
        return fit
    def test_construction(self):
        m = self.build_model()
        self.assertTrue(m.compartments==['susceptible', 'infected', 'recovered', 'exposed'])
        self.assertTrue(m.model_parameters==['dates', 'initial_susceptible', 'initial_infected', 'inital_recovered', 'beta', 'gamma', 'alpha'])
        self.assertTrue(m.fit_columns==['hospital_census','vent_census'])
    def test_fit(self):
        m = self.build_model()
        parameters, data, test_set = self.init_params()
        xx, pp = self.prepare_params(parameters, data)
        f = self.build_fit(data, xx, pp, m)
        self.assertTrue(f)

if __name__ == '__main__':
    unittest.main()