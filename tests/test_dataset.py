from pathlib import Path

import pytest

from pystamps.io.dataset import discover_dataset, infer_merged_stage, infer_patch_stage


DATASET = Path("inputs_and_outputs/InSAR_dataset_test")


pytestmark = [
    pytest.mark.skipif(
        not DATASET.exists(),
        reason="requires local parity dataset under inputs_and_outputs/InSAR_dataset_test",
    ),
    pytest.mark.dataset_parity,
]


def test_discover_dataset_has_patches() -> None:
    layout = discover_dataset(DATASET)

    assert layout.root.exists()
    assert len(layout.patches) >= 4


def test_stage_inference_on_reference_data() -> None:
    layout = discover_dataset(DATASET)

    for patch in layout.patches:
        assert infer_patch_stage(patch) >= 4

    assert infer_merged_stage(DATASET) >= 7
