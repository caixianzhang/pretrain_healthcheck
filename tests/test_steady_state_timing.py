from __future__ import annotations

import os
import unittest
from unittest import mock

from pretrain_healthcheck.torch_checks import DistEnv, _steady_state_timings


class FakeEvent:
    def __init__(self, api, enable_timing=True):
        self.api = api
        self.value = 0.0

    def record(self):
        self.value = self.api.clock_ms

    def elapsed_time(self, other):
        return other.value - self.value


class FakeNpu:
    def __init__(self):
        self.clock_ms = 0.0
        self.sync_count = 0

    def is_available(self):
        return True

    def synchronize(self):
        self.sync_count += 1

    def Event(self, enable_timing=True):
        return FakeEvent(self, enable_timing)


class FakeTorch:
    def __init__(self):
        self.npu = FakeNpu()


class FakeDist:
    def __init__(self):
        self.barrier_count = 0

    def barrier(self, **_kwargs):
        self.barrier_count += 1


class SteadyStateTimingTest(unittest.TestCase):
    def test_default_uses_one_measurement_batch(self) -> None:
        torch = FakeTorch()
        dist = FakeDist()
        env = DistEnv(0, 0, 1, 1, "npu:0", "host", "hccl", "hccl", "ascend", "hccl", "", "pod", "node")
        calls = 0

        def operation():
            nonlocal calls
            calls += 1
            torch.npu.clock_ms += 2.0

        with mock.patch.dict(os.environ, {}, clear=True):
            latencies, mode = _steady_state_timings(torch, dist, env, operation, iters=3)

        self.assertEqual(mode, "steady_state_device_event")
        self.assertEqual(calls, 3)
        self.assertEqual(dist.barrier_count, 1)
        self.assertEqual(torch.npu.sync_count, 2)
        self.assertEqual(latencies, [0.002])

    def test_continuous_iterations_use_one_sync_pair_per_batch(self) -> None:
        torch = FakeTorch()
        dist = FakeDist()
        env = DistEnv(0, 0, 1, 1, "npu:0", "host", "hccl", "hccl", "ascend", "hccl", "", "pod", "node")
        calls = 0

        def operation():
            nonlocal calls
            calls += 1
            torch.npu.clock_ms += 2.0

        latencies, mode = _steady_state_timings(torch, dist, env, operation, iters=3, measurement_batches=3)
        self.assertEqual(mode, "steady_state_device_event")
        self.assertEqual(calls, 9)
        self.assertEqual(dist.barrier_count, 3)
        self.assertEqual(torch.npu.sync_count, 6)
        self.assertEqual(latencies, [0.002, 0.002, 0.002])


if __name__ == "__main__":
    unittest.main()
