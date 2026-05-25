use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeReadiness {
    Planned,
    Scaffolded,
    ParityCertified,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Deserialize, Serialize)]
pub struct StageImplementation {
    pub stage: u8,
    pub scope: &'static str,
    pub crate_name: &'static str,
    pub entrypoint: &'static str,
    pub readiness: NativeReadiness,
    pub details: &'static str,
}

pub fn native_stage_inventory() -> &'static [StageImplementation] {
    &INVENTORY
}

pub fn native_stage_is_parity_certified(stage: u8, scope: &str) -> bool {
    INVENTORY.iter().any(|implementation| {
        implementation.stage == stage
            && implementation.scope == scope
            && implementation.readiness == NativeReadiness::ParityCertified
    })
}

pub fn native_stage_details(stage: u8, scope: &str) -> &'static str {
    INVENTORY
        .iter()
        .find(|implementation| implementation.stage == stage && implementation.scope == scope)
        .map(|implementation| implementation.details)
        .unwrap_or("No native stage scaffold has been registered for this stage scope.")
}

const INVENTORY: [StageImplementation; 9] = [
    StageImplementation {
        stage: 1,
        scope: "patch",
        crate_name: "pystamps-core",
        entrypoint: "native_stage1::run_stage1_native",
        readiness: NativeReadiness::ParityCertified,
        details: "Stage 1 patch execution is native for canonical raw single-master inputs, reusable ps1 metadata, and SNAP metadata synthesis with parity coverage.",
    },
    StageImplementation {
        stage: 2,
        scope: "patch",
        crate_name: "pystamps-core",
        entrypoint: "native_stage2::run_stage2_native",
        readiness: NativeReadiness::ParityCertified,
        details: "Stage 2 patch coherence estimation is native for MAT artifact loading, CLAP grid preparation/checkpoint writes, iterative weighting, topofit-compatible output variables, and synthetic parity coverage.",
    },
    StageImplementation {
        stage: 3,
        scope: "patch",
        crate_name: "pystamps-core",
        entrypoint: "native_stage3::run_stage3_native",
        readiness: NativeReadiness::ParityCertified,
        details: "Stage 3 patch selection is native for PERCENT and density threshold selection with parity coverage.",
    },
    StageImplementation {
        stage: 4,
        scope: "patch",
        crate_name: "pystamps-core",
        entrypoint: "native_stage4::run_stage4_native",
        readiness: NativeReadiness::ParityCertified,
        details: "Stage 4 patch weeding is native for selected artifact loading, neighbor/height/duplicate masks, Rust graph construction, Rust edge-stat reduction, and weed1 parity coverage.",
    },
    StageImplementation {
        stage: 5,
        scope: "patch",
        crate_name: "pystamps-core",
        entrypoint: "native_stage5::run_stage5_patch_native",
        readiness: NativeReadiness::ParityCertified,
        details: "Stage 5 patch promotion is native for selected/weeded artifact promotion with parity coverage.",
    },
    StageImplementation {
        stage: 5,
        scope: "merged",
        crate_name: "pystamps-core",
        entrypoint: "native_stage5::run_stage5_merge_native",
        readiness: NativeReadiness::ParityCertified,
        details: "Stage 5 merged aggregation is native for promoted patch artifact loading, overlap handling, dataset artifact writes, and merged ifg standard deviation parity.",
    },
    StageImplementation {
        stage: 6,
        scope: "merged",
        crate_name: "pystamps-core",
        entrypoint: "native_stage6::run_stage6_native",
        readiness: NativeReadiness::ParityCertified,
        details: "Stage 6 merged unwrap is native for MAT artifact loading, Rust grid graph generation, graph-based phase unwrapping, and phuw2/uw_phaseuw/uw_grid/uw_interp writes with synthetic parity coverage.",
    },
    StageImplementation {
        stage: 7,
        scope: "merged",
        crate_name: "pystamps-core",
        entrypoint: "native_stage7::run_stage7_native",
        readiness: NativeReadiness::ParityCertified,
        details: "Stage 7 merged SCLA orchestration is native for MAT artifact loading, Rust SCLA kernel execution, and scla2/scla_smooth2 writes with parity coverage.",
    },
    StageImplementation {
        stage: 8,
        scope: "merged",
        crate_name: "pystamps-core",
        entrypoint: "native_stage8::run_stage8_native",
        readiness: NativeReadiness::ParityCertified,
        details: "Stage 8 merged filtering orchestration is native for merged unwrap/SCLA artifact loading, Rust edge-noise kernel execution, and mean_v/uw_space_time writes with parity coverage.",
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inventory_covers_all_stage_scopes() {
        let scopes: Vec<(u8, &str)> = native_stage_inventory()
            .iter()
            .map(|implementation| (implementation.stage, implementation.scope))
            .collect();

        assert_eq!(
            scopes,
            vec![
                (1, "patch"),
                (2, "patch"),
                (3, "patch"),
                (4, "patch"),
                (5, "patch"),
                (5, "merged"),
                (6, "merged"),
                (7, "merged"),
                (8, "merged"),
            ]
        );
    }

    #[test]
    fn stage1_patch_is_parity_certified() {
        assert!(native_stage_is_parity_certified(1, "patch"));
    }

    #[test]
    fn stage2_patch_is_parity_certified() {
        assert!(native_stage_is_parity_certified(2, "patch"));
    }

    #[test]
    fn stage3_patch_is_parity_certified() {
        assert!(native_stage_is_parity_certified(3, "patch"));
    }

    #[test]
    fn stage5_patch_is_parity_certified() {
        assert!(native_stage_is_parity_certified(5, "patch"));
        assert!(native_stage_is_parity_certified(5, "merged"));
    }

    #[test]
    fn stage6_merged_is_parity_certified() {
        assert!(native_stage_is_parity_certified(6, "merged"));
    }

    #[test]
    fn stage7_merged_is_parity_certified() {
        assert!(native_stage_is_parity_certified(7, "merged"));
    }

    #[test]
    fn stage8_merged_is_parity_certified() {
        assert!(native_stage_is_parity_certified(8, "merged"));
    }
}
