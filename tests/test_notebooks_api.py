from __future__ import annotations

from pathlib import Path
import os

import numpy as np

from pystamps.input_contracts import describe_stage1_snap2stamps_flow
from pystamps.io.mat import write_mat
from pystamps.notebooks import (
    compare_fields,
    compute_field_statistics,
    inspect_stage1_inputs,
    load_velocity_diagnostics,
    match_diagnostic_points,
    plot_mode_from_step8,
)
from pystamps.notebooks import diagnostics as diagnostics_module
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


def test_describe_stage1_snap2stamps_flow_mentions_export_step() -> None:
    rows = describe_stage1_snap2stamps_flow()

    assert any("stamps export" in row["upstream_stage"].lower() for row in rows)
    assert any("pscands.1.ph" in row["why_stage1_cares"] for row in rows)


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


def test_plot_mode_from_step8_defaults_and_override() -> None:
    assert plot_mode_from_step8(True) == 'dos'
    assert plot_mode_from_step8(False) == 'do'
    assert plot_mode_from_step8(True, 'do') == 'do'


def test_load_velocity_diagnostics_falls_back_to_stage7_stability_proxy(tmp_path: Path) -> None:
    write_mat(
        tmp_path / 'ps2.mat',
        {
            'lonlat': np.array([[11.0, 44.0], [12.0, 45.0], [13.0, 46.0]], dtype=np.float64),
            'day': np.array([10.0, 20.0, 30.0], dtype=np.float64),
            'master_ix': np.asarray(2.0, dtype=np.float64),
        },
    )
    write_mat(tmp_path / 'pm2.mat', {'coh_ps': np.array([0.9, 0.8, 0.7], dtype=np.float64)})
    write_mat(
        tmp_path / 'mean_v.mat',
        {'m': np.array([[1.0, 2.0, 3.0], [0.1, 0.2, 0.3]], dtype=np.float32)},
    )
    write_mat(
        tmp_path / 'mv2.mat',
        {
            'mean_v': np.array([0.1, 0.2, 0.3], dtype=np.float32),
            'mean_v_std': np.zeros(3, dtype=np.float32),
        },
    )
    write_mat(tmp_path / 'scla2.mat', {'C_ps_uw': np.array([-2.0, 0.5, 1.5], dtype=np.float32)})

    diag = load_velocity_diagnostics(tmp_path, apply_step8=False)

    assert diag.plot_mode == 'do'
    assert diag.velocity_source == 'mv2.mean_v'
    assert diag.stability_source == 'abs(scla2.C_ps_uw) stability proxy'
    np.testing.assert_allclose(diag.time_axis_days, [-10.0, 0.0, 10.0])
    np.testing.assert_allclose(diag.stability, [2.0, 0.5, 1.5])
    np.testing.assert_allclose(diag.slope, [0.1, 0.2, 0.3])


def test_load_velocity_diagnostics_uses_mean_v_when_mv2_missing(tmp_path: Path) -> None:
    write_mat(
        tmp_path / 'ps2.mat',
        {
            'lonlat': np.array([[11.0, 44.0], [12.0, 45.0]], dtype=np.float64),
            'day': np.array([10.0, 20.0, 30.0], dtype=np.float64),
            'master_ix': np.asarray(2.0, dtype=np.float64),
        },
    )
    write_mat(
        tmp_path / 'mean_v.mat',
        {'m': np.array([[1.0, 2.0], [0.1, 0.2]], dtype=np.float32)},
    )
    write_mat(tmp_path / 'scla2.mat', {'C_ps_uw': np.array([-2.0, 0.5], dtype=np.float32)})

    diag = load_velocity_diagnostics(tmp_path, apply_step8=True)

    assert diag.velocity_source == 'mean_v.m slope fallback'
    assert diag.stability_source == 'abs(scla2.C_ps_uw) stability proxy'
    np.testing.assert_allclose(diag.velocity, [0.1, 0.2])
    np.testing.assert_allclose(diag.stability, [2.0, 0.5])


def test_load_velocity_diagnostics_skips_stage7_proxy_when_std_exists(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / 'mv2.mat').touch()
    payloads = {
        'ps2.mat': {
            'lonlat': np.array([[11.0, 44.0], [12.0, 45.0]], dtype=np.float64),
            'day': np.array([10.0, 20.0, 30.0], dtype=np.float64),
            'master_ix': np.asarray(2.0, dtype=np.float64),
        },
        'mean_v.mat': {'m': np.array([[1.0, 2.0], [0.1, 0.2]], dtype=np.float32)},
        'mv2.mat': {
            'mean_v': np.array([0.1, 0.2], dtype=np.float32),
            'mean_v_std': np.array([0.01, 0.02], dtype=np.float32),
        },
    }

    def fake_read_mat(path):
        name = Path(path).name
        if name == 'scla2.mat':
            raise AssertionError('scla2.mat should not be loaded when mean_v_std is populated')
        return payloads[name]

    monkeypatch.setattr(diagnostics_module, 'read_mat', fake_read_mat)

    diag = load_velocity_diagnostics(tmp_path, apply_step8=True)

    assert diag.stability_source == 'mv2.mean_v_std'
    np.testing.assert_allclose(diag.stability, [0.01, 0.02])


def test_match_diagnostic_points_uses_lonlat_alignment() -> None:
    run_diag = diagnostics_module.VelocityDiagnostics(
        root=Path('.'),
        plot_mode='dos',
        lonlat=np.array([[10.0, 40.0], [20.0, 50.0]], dtype=float),
        day=np.array([0.0, 12.0], dtype=float),
        master_ix=1,
        time_axis_days=np.array([0.0, 12.0], dtype=float),
        coherence=np.ones(2, dtype=float),
        velocity=np.array([1.0, 2.0], dtype=np.float32),
        velocity_source='mv2.mean_v',
        stability=np.array([0.1, 0.2], dtype=np.float32),
        stability_source='mv2.mean_v_std',
        intercept=np.zeros(2, dtype=np.float32),
        slope=np.array([1.0, 2.0], dtype=np.float32),
    )
    stamps_diag = diagnostics_module.VelocityDiagnostics(
        root=Path('.'),
        plot_mode='dos',
        lonlat=np.array([[99.0, 99.0], [20.0, 50.0], [10.0, 40.0]], dtype=float),
        day=np.array([0.0, 12.0], dtype=float),
        master_ix=1,
        time_axis_days=np.array([0.0, 12.0], dtype=float),
        coherence=np.ones(3, dtype=float),
        velocity=np.array([9.0, 2.5, 1.5], dtype=np.float32),
        velocity_source='mv2.mean_v',
        stability=np.array([0.9, 0.25, 0.15], dtype=np.float32),
        stability_source='mv2.mean_v_std',
        intercept=np.zeros(3, dtype=np.float32),
        slope=np.array([9.0, 2.5, 1.5], dtype=np.float32),
    )

    matched = match_diagnostic_points(run_diag, stamps_diag)

    np.testing.assert_array_equal(matched.run_indices, [0, 1])
    np.testing.assert_array_equal(matched.stamps_indices, [2, 1])

    report = diagnostics_module.build_velocity_report(run_diag, stamps_diag)
    comparison = {row['field']: row for row in report['comparison_rows']}
    assert comparison['v-dos']['count'] == 2.0
    assert np.isclose(comparison['v-dos']['bias'], -0.5)


def test_compute_field_statistics_and_compare_fields() -> None:
    stats = compute_field_statistics(
        np.array([1.0, 2.0, 3.0, 100.0], dtype=float),
        compute_percentiles=True,
        percentiles=(5.0, 95.0),
        outlier_filter='iqr',
    )
    comparison = compare_fields(
        np.array([1.0, 2.0, 3.0], dtype=float),
        np.array([1.5, 2.5, 3.5], dtype=float),
        metrics=('RMSE', 'MAE', 'bias'),
    )

    assert stats['count'] == 3.0
    assert stats['mean'] == 2.0
    assert np.isclose(comparison['bias'], -0.5)
    assert np.isclose(comparison['mae'], 0.5)
    assert np.isclose(comparison['rmse'], 0.5)
