# -*- coding: utf-8 -*-
"""Package Unittests."""

import unittest
from ..bayesian.normal.models import SEIRModel

class TestSEIR(unittest.TestCase):
    def test_construction(self):
        model = SEIRModel(fit_columns=['hospital_census', 'vent_census'], update_parameters=lambda _:_)
        self.assertTrue(model.compartments==['susceptible','infected','recovered','exposed'])
        self.assertTrue(model.model_parameters==['dates','initial_susceptible','initial_infected','inital_recovered','beta','gamma','alpha'])
        self.assertTrue(model.fit_columns==['hospital_census','vent_census'])

if __name__ == '__main__':
    unittest.main()
