import unittest
from eventforecastwm.performance_v2.prepare_reproduction import rewrite,producer_dependencies


class ReproductionPaths(unittest.TestCase):
    def test_path_boundaries_and_external_cache(self):
        value={'run':'/campaign/fit','args':['/campaign/fit/best.pt','/campaign/fitting/manifest.json',
            '/data/frozen/manifest.json'],'seed':20261014,'steps':24576}
        expected={'run':'/new/fit','args':['/new/fit/best.pt','/campaign/fitting/manifest.json',
            '/data/frozen/manifest.json'],'seed':20261014,'steps':24576}
        self.assertEqual(rewrite(value,{'/campaign/fit':'/new/fit'}),expected)
        self.assertEqual(value['run'],'/campaign/fit')

    def test_specific_config_mapping_precedes_parent(self):
        mapping={'/source':'/new/source','/source/config.json':'/new/configs/config.json'}
        self.assertEqual(rewrite(['/source/config.json','/source/module.py'],mapping),
                         ['/new/configs/config.json','/new/source/module.py'])

    def test_named_comparison_argument(self):
        self.assertEqual(rewrite('candidate=/campaign/fit/result.json',{'/campaign/fit':'/new/fit'}),
                         'candidate=/new/fit/result.json')

    def test_alias_introduces_dependency_without_self_dependency(self):
        producers={'/new/physical':'/new/physical/result.json','/new/calibration':'/new/calibration/result.json'}
        cfg={'run':'/new/calibration','checkpoint':'/new/physical/best.pt','output':'/new/calibration/model.pt',
             'args':['reference=/new/physical/result.json'],'manifest':'/data/frozen/manifest.json'}
        self.assertEqual(producer_dependencies(cfg,producers,'/new/calibration'),{'/new/physical/result.json'})


if __name__=='__main__':unittest.main()
