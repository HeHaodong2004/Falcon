import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from eventforecastwm.performance_v2.resource_policy import AdaptiveGuard,throughput_state,valid_samples,extraction_recovery,VERSION


def sample(timestamp,rate,**kwargs):
    return dict(timestamp=timestamp,heartbeat_timestamp=timestamp-1,collector_pid=42,state='running',
                queries=int(timestamp),recent_qph=rate,**kwargs)


class TrainingPriorityGuardTest(unittest.TestCase):
    def test_old_pause_recovers_at_2100_after_two_valid_windows(self):
        state=dict(samples=[sample(1000,2100)],legacy_resume_pending=True,throttle_level=1,stopped=True)
        throughput_state(state);self.assertTrue(state['legacy_resume_pending'])
        state['samples'].append(sample(1060,2100));throughput_state(state)
        self.assertFalse(state['legacy_resume_pending']);self.assertEqual(state['duty_cycle'],1.)

    def test_acceptable_rates_do_not_throttle_or_stop(self):
        state=dict(samples=[sample(1000+i*60,1200) for i in range(8)],stopped=False)
        throughput_state(state);self.assertEqual(state['throughput_status'],'acceptable')
        self.assertEqual(state['throttle_level'],0);self.assertFalse(state['stopped'])

    def test_sustained_low_rate_throttles_but_never_stops_all_research(self):
        state=dict(samples=[],stopped=False)
        for i in range(9):
            state['samples'].append(sample(1000+i*60,800));throughput_state(state)
            self.assertEqual(state['throttle_level'],min(3,(i+1)//3))
            self.assertFalse(state['stopped']);self.assertGreaterEqual(state['duty_cycle'],.6)
            self.assertFalse(state['defer_auxiliary'])
        state['samples'].append(sample(1600,1800));throughput_state(state)
        self.assertEqual(state['throttle_level'],3)
        state['samples'].append(sample(1660,1800));throughput_state(state)
        self.assertEqual(state['throttle_level'],0)

    def test_healthy_low_throughput_does_not_starve_cpu_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'guard.json'
            evidence=dict(policy_version=VERSION,stopped=False,samples=[dict(timestamp=1000)],
                          latest_valid_qph=800.,duty_cycle=.6,defer_auxiliary=True)
            path.write_text(json.dumps(evidence))
            with patch('eventforecastwm.performance_v2.resource_policy.time.time',return_value=1001):
                self.assertTrue(extraction_recovery(path)['throughput_is_soft_constraint'])
                evidence['stopped']=True;path.write_text(json.dumps(evidence))
                with self.assertRaisesRegex(RuntimeError,'Research stopped:'):extraction_recovery(path)

    def test_invalid_and_rapid_samples_do_not_trigger_throttling(self):
        history=[sample(1000,800),sample(1001,800),sample(1060,800,measurement_valid=False)]
        self.assertEqual(len(valid_samples(history)),1)
        state=throughput_state(dict(samples=history,stopped=False));self.assertEqual(state['throttle_level'],0)

    def actual_guard(self,directory,ram=100*1024**3,disk=100*1024**3,heartbeat=1650,oom=0):
        previous=dict(samples=[dict(timestamp=1000,heartbeat_timestamp=990,collector_pid=42,
            state='running',queries=1000,recent_qph=2000)],stopped=False,collector_pid=42,kernel_oom_baseline=0)
        (directory/'guard.json').write_text(json.dumps(previous))
        current=dict(timestamp=1660,heartbeat_timestamp=heartbeat,collector_pid=42,state='running',queries=1100,qph=9999)
        resources=dict(available_ram_bytes=ram,disk_free_bytes=disk,kernel_oom_kills=oom,load_average=[80,80,80])
        with patch('eventforecastwm.performance_v2.resource_policy.collector_sample',return_value=current),\
             patch('eventforecastwm.performance_v2.resource_policy.resource_sample',return_value=resources),\
             patch('eventforecastwm.performance_v2.resource_policy.time.time',return_value=1660),\
             patch('eventforecastwm.performance_v2.resource_policy.os.kill'):
            return AdaptiveGuard(directory)

    def test_rate_uses_counter_and_heartbeat_interval_not_lifetime_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            guard=self.actual_guard(Path(tmp));row=guard.state['samples'][-1]
            self.assertAlmostEqual(row['recent_qph'],100*3600/660)
            self.assertEqual(row['window_seconds'],660);self.assertTrue(row['measurement_valid'])
            self.assertFalse(guard.state['stopped'])

    def test_hard_safety_constraints_still_stop(self):
        cases=[dict(ram=47*1024**3),dict(disk=49*1024**3),dict(heartbeat=1000),dict(oom=1)]
        for options in cases:
            with self.subTest(options=options),tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(RuntimeError,'Research stopped:'):self.actual_guard(Path(tmp),**options)


if __name__=='__main__':unittest.main()
