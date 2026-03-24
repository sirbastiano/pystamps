from __future__ import annotations

from copy import deepcopy
from typing import Any


STAGE_INPUT_CONTRACTS: dict[int, dict[str, Any]] = {
    1: {
        "title": "Load and organize raw candidate inputs",
        "inputs": [
            {
                "logical_input": "candidate indices",
                "array_name": "ij",
                "shape_hint": "[n_candidates, 3]",
                "meaning": "candidate id, azimuth row, range column",
                "common_sources": "pscands.1.ij, npy, h5, pickle, in-memory ndarray",
            },
            {
                "logical_input": "complex candidate phase stack",
                "array_name": "ph",
                "shape_hint": "[n_candidates, n_ifg]",
                "meaning": "one complex phase value per candidate and interferogram",
                "common_sources": "pscands.1.ph, mat, npy, h5, pickle, in-memory ndarray",
            },
            {
                "logical_input": "candidate longitude/latitude",
                "array_name": "lonlat",
                "shape_hint": "[n_candidates, 2]",
                "meaning": "longitude and latitude for each candidate",
                "common_sources": "pscands.1.ll, npy, h5, pickle, in-memory ndarray",
            },
            {
                "logical_input": "patch width",
                "array_name": "width",
                "shape_hint": "scalar",
                "meaning": "range width of the patch raster",
                "common_sources": "width.txt, config, h5 attribute, in-memory scalar",
            },
            {
                "logical_input": "patch length",
                "array_name": "length",
                "shape_hint": "scalar",
                "meaning": "azimuth length of the patch raster",
                "common_sources": "len.txt, config, h5 attribute, in-memory scalar",
            },
            {
                "logical_input": "acquisition days",
                "array_name": "day",
                "shape_hint": "[n_ifg]",
                "meaning": "slave acquisition dates used to build the time axis",
                "common_sources": "day.1.in or values derived from diff0/rslc metadata",
            },
            {
                "logical_input": "master acquisition day",
                "array_name": "master_day",
                "shape_hint": "scalar",
                "meaning": "the master date inserted into the image timeline",
                "common_sources": "master_day.1.in or values derived from diff0/rslc metadata",
            },
            {
                "logical_input": "perpendicular baseline summary",
                "array_name": "bperp",
                "shape_hint": "[n_ifg]",
                "meaning": "one perpendicular baseline value per interferogram",
                "common_sources": "bperp.1.in or values derived from diff0/rslc metadata",
            },
            {
                "logical_input": "optional stability metric",
                "array_name": "D_A",
                "shape_hint": "[n_candidates]",
                "meaning": "candidate stability values for QC and side outputs",
                "common_sources": "pscands.1.da, npy, h5, pickle, in-memory ndarray",
            },
            {
                "logical_input": "optional height prior",
                "array_name": "hgt",
                "shape_hint": "[n_candidates]",
                "meaning": "per-candidate height values when available",
                "common_sources": "pscands.1.hgt, npy, h5, pickle, in-memory ndarray",
            },
        ],
    },
    2: {
        "title": "Estimate phase model and coherence per patch",
        "inputs": [
            {
                "logical_input": "stage-1 geometry payload",
                "array_name": "ps1.{ij, lonlat, xy, day, bperp}",
                "shape_hint": "mixed arrays",
                "meaning": "organized candidate geometry and timing metadata from Stage 1",
                "common_sources": "ps1.mat or equivalent in-memory structure",
            },
            {
                "logical_input": "stage-1 phase stack",
                "array_name": "ph1.ph",
                "shape_hint": "[n_candidates, n_image]",
                "meaning": "complex phase stack aligned to Stage-1 metadata",
                "common_sources": "ph1.mat or equivalent in-memory ndarray",
            },
            {
                "logical_input": "baseline matrix",
                "array_name": "bp1.bperp_mat",
                "shape_hint": "[n_candidates, n_ifg]",
                "meaning": "per-candidate perpendicular baselines",
                "common_sources": "bp1.mat or equivalent in-memory ndarray",
            },
        ],
    },
    3: {
        "title": "Select persistent scatterers",
        "inputs": [
            {
                "logical_input": "phase-model fit outputs",
                "array_name": "pm1.{K_ps, C_ps, coh_ps}",
                "shape_hint": "[n_candidates]",
                "meaning": "per-candidate fit parameters and coherence scores",
                "common_sources": "pm1.mat or equivalent in-memory arrays",
            },
            {
                "logical_input": "stage-1 candidate metadata",
                "array_name": "ps1.{ij, lonlat, xy}",
                "shape_hint": "mixed arrays",
                "meaning": "candidate geometry used to map selections back to patch space",
                "common_sources": "ps1.mat or equivalent in-memory structure",
            },
        ],
    },
    4: {
        "title": "Weed noisy or redundant candidates",
        "inputs": [
            {
                "logical_input": "stage-3 selection masks",
                "array_name": "select1.{ix, keep_ix}",
                "shape_hint": "[n_selected]",
                "meaning": "selected candidates and their keep/reject decisions",
                "common_sources": "select1.mat or equivalent in-memory arrays",
            },
            {
                "logical_input": "candidate geometry and QC metrics",
                "array_name": "ps1 / da1 / pm1 arrays",
                "shape_hint": "mixed arrays",
                "meaning": "geometry and quality terms used to weed the selected set",
                "common_sources": "stage-1 and stage-2 outputs or equivalent in-memory structures",
            },
        ],
    },
    5: {
        "title": "Merge patch outputs into one dataset view",
        "inputs": [
            {
                "logical_input": "patch-level retained candidates",
                "array_name": "ps2 / ph2 / bp2 per patch",
                "shape_hint": "mixed arrays",
                "meaning": "the patch outputs that survive Stage 4 and are ready to be promoted",
                "common_sources": "per-patch MAT files or equivalent in-memory structures",
            }
        ],
    },
    6: {
        "title": "Unwrap the merged phase products",
        "inputs": [
            {
                "logical_input": "merged phase stack",
                "array_name": "ph2 / ps2 / baseline and graph-support arrays",
                "shape_hint": "mixed arrays",
                "meaning": "merged candidate geometry and phase values prepared for unwrapping",
                "common_sources": "merged MAT files or equivalent in-memory structures",
            }
        ],
    },
    7: {
        "title": "Estimate slow trends and correction terms",
        "inputs": [
            {
                "logical_input": "unwrapped phase products",
                "array_name": "phuw2 / ps2 / auxiliary correction arrays",
                "shape_hint": "mixed arrays",
                "meaning": "unwrapped phase and merged geometry used for SCLA and velocity estimation",
                "common_sources": "Stage-6 outputs or equivalent in-memory structures",
            }
        ],
    },
    8: {
        "title": "Apply final space-time filtering",
        "inputs": [
            {
                "logical_input": "stage-7 correction products",
                "array_name": "scla2 / mean_v / ps2 / phuw2",
                "shape_hint": "mixed arrays",
                "meaning": "merged correction terms and unwrapped phase used for the final filtered products",
                "common_sources": "Stage-7 outputs or equivalent in-memory structures",
            }
        ],
    },
}


def parse_stage_spec(stage: str | int) -> list[int]:
    if isinstance(stage, int):
        if stage not in STAGE_INPUT_CONTRACTS:
            raise ValueError(f"Unsupported stage: {stage}")
        return [stage]

    stage_str = str(stage).strip().lower()
    if stage_str == "all":
        return sorted(STAGE_INPUT_CONTRACTS)

    stages: list[int] = []
    for token in stage_str.split(","):
        value = int(token.strip())
        if value not in STAGE_INPUT_CONTRACTS:
            raise ValueError(f"Unsupported stage: {value}")
        stages.append(value)
    return stages


def describe_stage_inputs(stage: str | int = "all") -> list[dict[str, Any]]:
    return [
        {"stage": stage_id, **deepcopy(STAGE_INPUT_CONTRACTS[stage_id])}
        for stage_id in parse_stage_spec(stage)
    ]
