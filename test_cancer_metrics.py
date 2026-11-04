import unittest
from cancer_metrics import harrell_c, ipcw_brier


class CancerMetricsTest(unittest.TestCase):
    def test_cindex_and_no_comparable_pairs(self):
        result=harrell_c([1,2,3],[1,1,0],[.9,.6,.1])
        self.assertEqual(result,{'value':1.,'comparable_pairs':3})
        self.assertIsNone(harrell_c([1,2],[0,0],[1,0])['value'])

    def test_brier_no_censoring_known_answer(self):
        # At horizon 2, first subject failed, second is event-free.
        got=ipcw_brier([1,3,4],[1,1,1],[1,3],[1,1],[.2,.8],2)
        self.assertAlmostEqual(got['value'],.04)
        self.assertEqual(got['g_horizon'],1)

    def test_brier_unsupported_horizon(self):
        got=ipcw_brier([1,2],[0,0],[1,3],[1,0],[.3,.7],3)
        self.assertIsNone(got['value'])

    def test_brier_training_censor_weights(self):
        got=ipcw_brier([1,3],[0,1],[.5,3,1],[1,1,0],[.2,.8,.9],2)
        self.assertAlmostEqual(got['g_horizon'],.5)
        self.assertAlmostEqual(got['value'],.04)

if __name__ == '__main__':
    unittest.main()
