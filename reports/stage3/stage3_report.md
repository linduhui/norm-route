# Stage 3 Routing and Oracle Report

## Scope

This report summarizes Stage 3 routing-task export, evaluator-only Oracle
analysis, and complementarity analysis.

## Inputs

- `outputs/stage3/evaluator_only/routing_matrix_long.csv`
- `outputs/stage3/evaluator_only/routing_matrix_wide.csv`
- `outputs/stage3/quality/expert_quality_by_run.csv`

## Agent-Visible Outputs

- `outputs/stage3/agent_visible/agent_routing_tasks.jsonl`
- `outputs/stage3/agent_visible/expert_cards.json`

These files must not contain `label`, `mask_path`, `defect_type`,
`anomaly_type`, or `oracle_best_expert`.

## Evaluator-Only Outputs

- `outputs/stage3/oracle/oracle_summary.csv`
- `outputs/stage3/oracle/oracle_selection_counts.csv`
- `outputs/stage3/oracle/complementarity_summary.csv`

Oracle artifacts must be marked `evaluator_only=True`.

## Results Summary

Fill in after running Stage 3:

- Best single expert:
- Run-level Oracle:
- Sample-level Oracle:
- Main complementarity finding:

## Validation Checklist

- Evaluator-only routing matrix files exist.
- Agent-visible files contain no label or mask leakage.
- Oracle files are marked evaluator-only.
- PatchCore, WinCLIP, and AnomalyDINO sample IDs are aligned.
- `pytest` passes after changes.
