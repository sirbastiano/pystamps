from __future__ import annotations

from pathlib import Path
import os

import numpy as np

from pystamps.notebooks import inspect_stage1_inputs
from pystamps.notebooks.plots import point_count, sample_points, select_points
from pystamps.notebooks.stage_execution import StageNotebookContext, _execution_env, stage3_indices, stage4_indices
from pystamps.config import RunConfig


def test_sample_points_transposes_two_by_n_coordinates() -> None:
    lonlat = np.array(
        [
            [11.0, 12.0, 13.0],
            [44.0, 45.0, 46.0],
        ],
        dtype=float,
    )
    values = np.array([1.0, 2.0, 3.0], dtype=float)

    sampled_points, sampled_values = sample_points(lonlat, values, limit=10)

    assert sampled_points.shape == (3, 2)
    np.testing.assert_allclose(sampled_points[:, 0], [11.0, 12.0, 13.0])
    np.testing.assert_allclose(sampled_points[:, 1], [44.0, 45.0, 46.0])
    np.testing.assert_allclose(sampled_values, values)


def test_point_count_handles_two_by_n_coordinates() -> None:
    lonlat = np.array(
        [
            [11.0, 12.0, 13.0, 14.0],
            [44.0, 45.0, 46.0, 47.0],
        ],
        dtype=float,
    )

    assert point_count(lonlat) == 4


def test_select_points_normalizes_before_indexing() -> None:
    lonlat = np.array(
        [
            [11.0, 12.0, 13.0, 14.0],
            [44.0, 45.0, 46.0, 47.0],
        ],
        dtype=float,
    )

    selected = select_points(lonlat, [1, 3])

    assert selected.shape == (2, 2)
    np.testing.assert_allclose(selected[:, 0], [12.0, 14.0])
    np.testing.assert_allclose(selected[:, 1], [45.0, 47.0])


def test_stage3_indices_respects_keep_mask() -> None:
    payload = {
        "ix": np.array([1, 2, 3, 4], dtype=int),
        "keep_ix": np.array([True, False, True, False], dtype=bool),
    }

    all_ix, kept_ix, rejected_ix = stage3_indices(payload)

    np.testing.assert_array_equal(all_ix, [0, 1, 2, 3])
    np.testing.assert_array_equal(kept_ix, [0, 2])
    np.testing.assert_array_equal(rejected_ix, [1, 3])


def test_stage4_indices_applies_both_weed_masks() -> None:
    select_payload = {
        "ix": np.array([1, 2, 3, 4], dtype=int),
        "keep_ix": np.array([True, True, True, False], dtype=bool),
    }
    weed_payload = {
        "ix_weed": np.array([True, False, True], dtype=bool),
        "ix_weed2": np.array([False, True], dtype=bool),
    }

    kept_after_stage3, final_ix = stage4_indices(select_payload, weed_payload)

    np.testing.assert_array_equal(kept_after_stage3, [0, 1, 2])
    np.testing.assert_array_equal(final_ix, [2])


def test_stage2_execution_env_respects_process_threads() -> None:
    context = StageNotebookContext(
        repo_root=Path('.'),
        stamps_root=Path('.'),
        scratch_parent=Path('.'),
        scratch_root=Path('.'),
        representative_patch='PATCH_1',
        config_path=None,
        replay_config_path=None,
        replay_stages=frozenset(),
        config=RunConfig(),
    )

    original = {key: os.environ.get(key) for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}
    try:
        for key in original:
            os.environ.pop(key, None)
        env = _execution_env(context, 2)
    finally:
        for key, value in original.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    assert 'OMP_NUM_THREADS' not in env
    assert 'OPENBLAS_NUM_THREADS' not in env
    assert 'MKL_NUM_THREADS' not in env


def test_non_stage2_execution_env_limits_threads() -> None:
    context = StageNotebookContext(
        repo_root=Path('.'),
        stamps_root=Path('.'),
        scratch_parent=Path('.'),
        scratch_root=Path('.'),
        representative_patch='PATCH_1',
        config_path=None,
        replay_config_path=None,
        replay_stages=frozenset(),
        config=RunConfig(),
    )

    env = _execution_env(context, 1)

    assert env['OMP_NUM_THREADS'] == '1'
    assert env['OPENBLAS_NUM_THREADS'] == '1'
    assert env['MKL_NUM_THREADS'] == '1'


def test_inspect_stage1_inputs_summarizes_raw_patch_inputs(tmp_path: Path) -> None:
    dataset = tmp_path / 'dataset'
    patch = dataset / 'PATCH_1'
    patch.mkdir(parents=True)
    (dataset / 'width.txt').write_text('20\n', encoding='utf-8')
    (dataset / 'len.txt').write_text('10\n', encoding='utf-8')

    np.savetxt(
        patch / 'pscands.1.ij',
        np.array([[1, 10, 20], [2, 11, 21]], dtype=float),
        fmt='%.0f',
    )
    np.savetxt(patch / 'pscands.1.da', np.array([0.2, 0.3], dtype=float))
    np.asarray([[11.0, 44.0], [12.0, 45.0]], dtype='>f4').tofile(patch / 'pscands.1.ll')
    np.asarray(
        [
            [1.0, 0.0, 0.5, 0.5],
            [0.0, 1.0, -0.5, 0.5],
            [1.0, 1.0, -1.0, 0.0],
        ],
        dtype='<f4',
    ).tofile(patch / 'pscands.1.ph')

    summary = inspect_stage1_inputs(dataset, patch_name='PATCH_1')

    assert summary['metadata_mode'] == 'missing'
    assert len(summary['overview_rows']) >= 4
    assert len(summary['consistency_rows']) >= 4
    assert summary['preparation_rows'][0]['step'] == 1
    assert any(row['mat_file'] == 'ps1.mat' for row in summary['mat_output_rows'])
    assert len(summary['preview_rows']) == 2
    assert summary['phase_preview']['angle'].shape == (2, 3)
    assert summary['phase_preview']['magnitude'].shape == (2, 3)
    assert any(row['file'] == 'pscands.1.ph' for row in summary['input_rows'])
    assert any('time axis' in warning for warning in summary['warnings'])
