from __future__ import annotations

import unittest

from pretrain_healthcheck.torch_checks import (
    _CollectiveBandwidthWorkspace,
    _collective_bandwidth_once,
)


class RecordingDist:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def all_reduce(self, tensor, group=None) -> None:
        self.calls.append(("all_reduce", tensor, group))

    def broadcast(self, tensor, src, group=None) -> None:
        self.calls.append(("broadcast", tensor, src, group))

    def reduce_scatter(self, output, chunks, group=None) -> None:
        self.calls.append(("reduce_scatter", output, chunks, group))

    def all_gather(self, outputs, tensor, group=None) -> None:
        self.calls.append(("all_gather", outputs, tensor, group))

    def all_to_all_single(
        self,
        output,
        tensor,
        output_split_sizes=None,
        input_split_sizes=None,
        group=None,
    ) -> None:
        self.calls.append(
            (
                "all_to_all_single",
                output,
                tensor,
                output_split_sizes,
                input_split_sizes,
                group,
            )
        )


class CollectiveBandwidthBufferTests(unittest.TestCase):
    def test_timed_collectives_reuse_preallocated_buffers(self) -> None:
        group = object()
        cases = {
            "all_reduce": _CollectiveBandwidthWorkspace(input_tensor=object()),
            "broadcast": _CollectiveBandwidthWorkspace(input_tensor=object()),
            "reduce_scatter": _CollectiveBandwidthWorkspace(
                input_tensor=object(), output_tensor=object(), input_chunks=[object(), object()]
            ),
            "all_gather": _CollectiveBandwidthWorkspace(
                input_tensor=object(), output_tensors=[object(), object()]
            ),
            "all_to_all": _CollectiveBandwidthWorkspace(
                input_tensor=object(),
                output_tensor=object(),
                input_split_sizes=[2, 2],
                output_split_sizes=[2, 2],
            ),
            "all_to_allv": _CollectiveBandwidthWorkspace(
                input_tensor=object(),
                output_tensor=object(),
                input_split_sizes=[1, 3],
                output_split_sizes=[2, 2],
            ),
        }

        for op, workspace in cases.items():
            with self.subTest(op=op):
                dist = RecordingDist()
                _collective_bandwidth_once(dist, op, workspace, group=group)
                _collective_bandwidth_once(dist, op, workspace, group=group)

                self.assertEqual(len(dist.calls), 2)
                self.assertIs(dist.calls[0][1], dist.calls[1][1])
                if op in {"reduce_scatter", "all_gather", "all_to_all", "all_to_allv"}:
                    self.assertIs(dist.calls[0][2], dist.calls[1][2])


if __name__ == "__main__":
    unittest.main()
