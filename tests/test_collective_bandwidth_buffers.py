from __future__ import annotations

import unittest

from pretrain_healthcheck.torch_checks import (
    _CollectiveBandwidthWorkspace,
    _collective_bandwidth_once,
    _object_from_packet,
    _object_packet,
    _routing_counts,
    _routing_output_counts,
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

                if op == "all_to_all":
                    self.assertIsNone(dist.calls[0][3])
                    self.assertIsNone(dist.calls[0][4])
                elif op == "all_to_allv":
                    self.assertEqual(dist.calls[0][3], [2, 2])
                    self.assertEqual(dist.calls[0][4], [1, 3])


class SplitSizeExchangeTests(unittest.TestCase):
    def test_reconstructs_512_rank_receive_splits_without_collective(self) -> None:
        world_size = 512
        destination_rank = 317
        total_tokens = 1_073_741_824
        for pattern in ("uniform", "empty_expert", "hot_expert", "skewed", "random"):
            with self.subTest(pattern=pattern):
                expected = [
                    _routing_counts(pattern, world_size, total_tokens, source_rank, 20260706)[destination_rank]
                    for source_rank in range(world_size)
                ]
                received = _routing_output_counts(
                    pattern,
                    world_size,
                    total_tokens,
                    destination_rank,
                    20260706,
                )
                self.assertEqual(received, expected)

    def test_rejects_invalid_destination_rank(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside world size"):
            _routing_output_counts("uniform", 4, 1024, 4, 20260706)


class ObjectPacketTests(unittest.TestCase):
    def test_round_trip_preserves_nested_metadata(self) -> None:
        obj = {"rank": 7, "rows": [{"latency": 0.125}], "ok": True}
        self.assertEqual(_object_from_packet(_object_packet(obj)), obj)

    def test_rejects_truncated_packet(self) -> None:
        packet = _object_packet({"rank": 7})
        with self.assertRaisesRegex(ValueError, "truncated"):
            _object_from_packet(packet[:-1])


if __name__ == "__main__":
    unittest.main()
