"""Static certification must inspect refreshed Bullet contacts at each pose."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
import magnum as mn
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'scripts'))
# Native geometry is covered by the isolated gate; unit tests fake Bullet only.
with patch.dict(sys.modules, {'habitat_sim':SimpleNamespace(physics=SimpleNamespace(VelocityControl=SimpleNamespace))}):
    from collector.static_habitat import HabitatStaticWorld

class ContactRefresh(unittest.TestCase):
    def world(self, bad_after=None):
        world=HabitatStaticWorld.__new__(HabitatStaticWorld)
        obj=SimpleNamespace(object_id=1,transformation=mn.Matrix4.identity_init())
        world.robot=SimpleNamespace(sim_obj=obj,base_pos=mn.Vector3(0,0,0))
        world.original=mn.Matrix4(obj.transformation)
        world.static_ids={0}
        count=[0]
        def detect():count[0]+=1
        def contacts():
            normal=.799787 if bad_after and count[0]>=bad_after else .999
            return [SimpleNamespace(object_id_a=1,object_id_b=0,
                contact_normal_on_b_in_ws=mn.Vector3(0,normal,0),
                position_on_a_in_ws=mn.Vector3(0,.035,0),
                position_on_b_in_ws=mn.Vector3(0,.029,0))]
        world.sim=SimpleNamespace(ac_freq_ratio=4,get_physics_time_step=lambda:.008,
            perform_discrete_collision_detection=detect,get_physics_contact_points=contacts)
        return world,count
    def test_later_contact_refresh_rejects_same_pose(self):
        world,count=self.world(bad_after=3)
        before=mn.Matrix4(world.robot.sim_obj.transformation)
        result=world.native_control_check(dict(executable_controls=[[0,0]],control_dt=.0486666667))
        self.assertFalse(result['static_feasible'])
        self.assertEqual(result['collision_pass'],2)
        self.assertEqual(result['failed_segment'],0)
        self.assertEqual(world.robot.sim_obj.transformation,before)
    def test_support_contacts_remain_valid_without_advancing_world(self):
        world,count=self.world()
        result=world.native_control_check(dict(executable_controls=[[0,0]],control_dt=.0486666667))
        self.assertTrue(result['static_feasible'])
        self.assertEqual(count[0],13)  # 12 refreshes + restored-pose detection
    def test_detection_exception_propagates_and_restores_pose(self):
        world,count=self.world()
        before=mn.Matrix4(world.robot.sim_obj.transformation)
        world.sim.get_physics_contact_points=Mock(side_effect=RuntimeError('detector failed'))
        with self.assertRaisesRegex(RuntimeError,'detector failed'):
            world.native_control_check(dict(executable_controls=[[0,0]],control_dt=.0486666667))
        self.assertEqual(world.robot.sim_obj.transformation,before)
