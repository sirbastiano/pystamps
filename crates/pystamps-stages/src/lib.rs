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
        readiness: NativeReadiness::Scaffolded,
        details: "Canonical raw single-master Stage 1 path is scaffolded; parity certification belongs to the Stage 1 port story.",
    },
    StageImplementation {
        stage: 2,
        scope: "patch",
        crate_name: "pystamps-stages",
        entrypoint: "planned_stage_port",
        readiness: NativeReadiness::Planned,
        details: "Stage 2 full native semantics are not implemented yet.",
    },
    StageImplementation {
        stage: 3,
        scope: "patch",
        crate_name: "pystamps-stages",
        entrypoint: "planned_stage_port",
        readiness: NativeReadiness::Planned,
        details: "Stage 3 full native semantics are not implemented yet.",
    },
    StageImplementation {
        stage: 4,
        scope: "patch",
        crate_name: "pystamps-stages",
        entrypoint: "planned_stage_port",
        readiness: NativeReadiness::Planned,
        details: "Stage 4 full native semantics are not implemented yet.",
    },
    StageImplementation {
        stage: 5,
        scope: "patch",
        crate_name: "pystamps-stages",
        entrypoint: "planned_stage_port",
        readiness: NativeReadiness::Planned,
        details: "Stage 5 patch promotion full native semantics are not implemented yet.",
    },
    StageImplementation {
        stage: 5,
        scope: "merged",
        crate_name: "pystamps-stages",
        entrypoint: "planned_stage_port",
        readiness: NativeReadiness::Planned,
        details: "Stage 5 merged aggregation full native semantics are not implemented yet.",
    },
    StageImplementation {
        stage: 6,
        scope: "merged",
        crate_name: "pystamps-stages",
        entrypoint: "planned_stage_port",
        readiness: NativeReadiness::Planned,
        details: "Stage 6 full native semantics are not implemented yet.",
    },
    StageImplementation {
        stage: 7,
        scope: "merged",
        crate_name: "pystamps-stages",
        entrypoint: "planned_stage_port",
        readiness: NativeReadiness::Planned,
        details: "Stage 7 full native semantics are not implemented yet.",
    },
    StageImplementation {
        stage: 8,
        scope: "merged",
        crate_name: "pystamps-stages",
        entrypoint: "planned_stage_port",
        readiness: NativeReadiness::Planned,
        details: "Stage 8 full native semantics are not implemented yet.",
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
    fn scaffolded_stage_is_not_parity_certified() {
        assert!(!native_stage_is_parity_certified(1, "patch"));
    }
}
