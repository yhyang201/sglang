import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers import mm_utils, schedule_batch
from sglang.srt.managers.mm_utils import (
    _try_heuristic_split,
    get_new_expanded_mm_items,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)


def _make_proxy_with_reconstruct_result(tensor: torch.Tensor):
    proxy = mm_utils.CudaIpcTensorTransportProxy.__new__(
        mm_utils.CudaIpcTensorTransportProxy
    )
    proxy.reconstruct_on_target_device = Mock(return_value=tensor)
    return proxy


class TestMultimodalInputsFromDict(unittest.TestCase):
    def test_materialize_proxy(self):
        """Test that CudaIpcTensorTransportProxy features are reconstructed correctly."""
        feature_tensor = torch.tensor([[7.0], [8.0]], dtype=torch.float32)
        proxy_feature = _make_proxy_with_reconstruct_result(feature_tensor)
        mm_item = MultimodalDataItem(
            modality=Modality.IMAGE,
            offsets=[(0, 1), (1, 2)],
            feature=proxy_feature,
            model_specific_data={"image_grid_thw": [[1, 1, 1], [1, 1, 1]]},
        )

        with patch.object(
            schedule_batch.torch.cuda, "is_available", return_value=True
        ), patch.object(
            schedule_batch.torch.cuda, "current_device", return_value=0
        ), patch.object(
            schedule_batch.envs.SGLANG_MM_BUFFER_SIZE_MB, "get", return_value=0
        ):
            mm_inputs = MultimodalInputs.from_dict({"mm_items": [mm_item]})

        # With image_grid_thw matching 2 offsets, the item should be split into 2
        self.assertEqual(len(mm_inputs.mm_items), 2)
        self.assertTrue(torch.equal(mm_inputs.mm_items[0].feature, feature_tensor[0:1]))
        self.assertTrue(torch.equal(mm_inputs.mm_items[1].feature, feature_tensor[1:2]))
        proxy_feature.reconstruct_on_target_device.assert_called_once_with(0)


class TestGetNewExpandedMmItems(unittest.TestCase):
    def test_single_offset_item_unchanged(self):
        """Items with a single offset should pass through unchanged."""
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            offsets=[(0, 5)],
            feature=torch.randn(3, 4),
            model_specific_data={},
        )
        result = get_new_expanded_mm_items([item])
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], item)

    def test_no_offsets_item_unchanged(self):
        """Items with offsets=None should pass through unchanged."""
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            offsets=None,
            feature=torch.randn(3, 4),
            model_specific_data={},
        )
        result = get_new_expanded_mm_items([item])
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], item)

    def test_split_with_image_grid_thw(self):
        """Items with matching image_grid_thw should be split correctly."""
        # 2 images: first has 2 patches (1*1*2), second has 3 patches (1*1*3)
        feature = torch.randn(5, 4)  # total 5 patches
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            offsets=[(0, 1), (5, 7)],
            feature=feature,
            model_specific_data={"image_grid_thw": [[1, 1, 2], [1, 1, 3]]},
        )
        result = get_new_expanded_mm_items([item])
        self.assertEqual(len(result), 2)
        self.assertTrue(torch.equal(result[0].feature, feature[0:2]))
        self.assertTrue(torch.equal(result[1].feature, feature[2:5]))
        self.assertEqual(result[0].offsets, [(0, 1)])
        self.assertEqual(result[1].offsets, [(5, 7)])

    def test_heuristic_split_tensor_dim0(self):
        """Items without grid_thw but with dim(0)==num_offsets should be split by heuristic."""
        feature = torch.randn(3, 4, 4)  # 3 images stacked
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            offsets=[(0, 2), (5, 7), (10, 12)],
            feature=feature,
            model_specific_data={"image_sizes": [(100, 100), (200, 200), (300, 300)]},
        )
        result = get_new_expanded_mm_items([item])
        self.assertEqual(len(result), 3)
        for i in range(3):
            self.assertTrue(torch.equal(result[i].feature, feature[i : i + 1]))
            self.assertEqual(result[i].offsets, [item.offsets[i]])

    def test_heuristic_split_list_feature(self):
        """Items with list features where len==num_offsets should be split."""
        feat_list = [torch.randn(4, 4), torch.randn(6, 4)]
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            offsets=[(0, 3), (5, 10)],
            feature=feat_list,
            model_specific_data={"tgt_size": [(2, 2), (3, 2)]},
        )
        result = get_new_expanded_mm_items([item])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].feature, [feat_list[0]])
        self.assertEqual(result[1].feature, [feat_list[1]])


class TestTryHeuristicSplit(unittest.TestCase):
    def test_unsplittable_item_kept_bundled(self):
        """Items that cannot be split should be returned as-is."""
        # feature dim(0) doesn't match num_items
        feature = torch.randn(7, 4)  # 7 != 3
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            offsets=[(0, 2), (5, 7), (10, 12)],
            feature=feature,
            model_specific_data={},
        )
        result = _try_heuristic_split(item, 3)
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], item)


if __name__ == "__main__":
    unittest.main(verbosity=2)
