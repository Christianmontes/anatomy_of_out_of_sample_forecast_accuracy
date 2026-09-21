"""Build the referee figure set for the three DGPs.

This script does not re-run simulations. It reads the canonical scenario
outputs under ``outputs/sim1_master_out`` and writes an organized figure tree
under ``outputs/sim1_master_out/referee_outputs``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
SIMULATION_ROOT = SCRIPT_DIR.parents[1]

REVISION_REFEREE_SCRIPT = SIMULATION_ROOT / "scripts" / "revision" / "postprocess_sim1_referee.py"
REFEREE_OUTPUT_ROOT = SIMULATION_ROOT / "outputs" / "sim1_master_out" / "referee_outputs"
DEFAULT_OUTPUT_ROOT = REFEREE_OUTPUT_ROOT / "three_dgp_figures"

MODEL_ORDER = ["ols", "enet", "rf"]


@dataclass(frozen=True)
class RefereeDGP:
    order: int
    label: str
    source_dir: Path
    note: str

    @property
    def scenario(self) -> str:
        return self.source_dir.name

    @property
    def output_folder(self) -> str:
        return f"{self.order:02d}_{self.label}__{self.scenario}"


def _load_referee_module():
    spec = importlib.util.spec_from_file_location("sim1_referee_postprocess", REVISION_REFEREE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load referee postprocessor: {REVISION_REFEREE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _default_dgps(include_appendix_break: bool) -> List[RefereeDGP]:
    dgps = [
        RefereeDGP(
            order=1,
            label="dgp1_structural_break",
            source_dir=SIMULATION_ROOT / "outputs" / "sim1_master_out" / "break_b08_expanding",
            note="Headline structural-break DGP.",
        ),
        RefereeDGP(
            order=2,
            label="dgp2_persistent_noise",
            source_dir=SIMULATION_ROOT / "outputs" / "sim1_master_out" / "persistent_rho099_rolling20",
            note="Persistent-noise / spurious-predictor DGP.",
        ),
        RefereeDGP(
            order=3,
            label="dgp3_proxy_break",
            source_dir=SIMULATION_ROOT / "outputs" / "sim1_master_out" / "proxy_break_reverse_noisyx1_expanding",
            note="Proxy-break DGP: stale proxy reverses after the break.",
        ),
    ]
    if include_appendix_break:
        dgps.insert(
            1,
            RefereeDGP(
                order=99,
                label="appendix_structural_break_strong",
                source_dir=SIMULATION_ROOT / "outputs" / "sim1_master_out" / "break_b15_expanding",
                note="Appendix strength variant of the structural-break DGP.",
            ),
        )
    return dgps


def _ensure_clean_output_root(output_root: Path, clean: bool) -> None:
    output_root = output_root.resolve()
    referee_root = REFEREE_OUTPUT_ROOT.resolve()
    if clean and output_root.exists():
        if referee_root not in output_root.parents:
            raise RuntimeError(f"Refusing to clean outside the referee output root: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def _validate_sources(dgps: Iterable[RefereeDGP]) -> None:
    missing = []
    for dgp in dgps:
        if not (dgp.source_dir / "grand_summary.csv").exists():
            missing.append(str(dgp.source_dir))
    if missing:
        joined = "\n  - ".join(missing)
        raise FileNotFoundError(f"Missing DGP output folders with grand_summary.csv:\n  - {joined}")


def _write_readme(output_root: Path, dgps: Sequence[RefereeDGP], models: Sequence[str]) -> None:
    rows = [
        "# Referee Figures: Three DGPs",
        "",
        "This figure set is generated from the final canonical CSV outputs under",
        "`Simulation/outputs/sim1_master_out/`. It does not re-run simulations.",
        "",
        "## Folder Structure",
        "",
        "Each DGP folder contains:",
        "",
        "- `combined/`: multi-page PDFs with all selected models.",
        "- `by_model/<model>/`: one-page PDFs for a single model.",
        "- `referee_model_summary.csv`: one row per model for that DGP.",
        "",
        "Top-level files:",
        "",
        "- `referee_model_summary_all_dgps.csv`: combined summary across DGPs.",
        "- `manifest.json`: source folders and model list used for this figure set.",
        "",
        "## DGPs",
        "",
    ]
    for dgp in dgps:
        rows.append(f"- `{dgp.output_folder}`: {dgp.note}")
        rows.append(f"  Source: `{dgp.source_dir}`")
    rows.extend(
        [
            "",
            "## Models",
            "",
            ", ".join(f"`{m}`" for m in models),
            "",
        ]
    )
    (output_root / "README.md").write_text("\n".join(rows), encoding="utf-8")


def _make_one_dgp_figure_set(referee, dgp: RefereeDGP, output_root: Path, models: Sequence[str]) -> pd.DataFrame:
    source_dir = dgp.source_dir.resolve()
    scenario = dgp.scenario
    out_dir = output_root / dgp.output_folder
    combined_dir = out_dir / "combined"
    by_model_dir = out_dir / "by_model"
    combined_dir.mkdir(parents=True, exist_ok=True)
    by_model_dir.mkdir(parents=True, exist_ok=True)

    grand = referee._read_grand_summary(str(source_dir))
    grand["scenario"] = scenario

    feature_df = referee._build_long_feature_df(str(source_dir), grand)
    long_metrics = referee._to_long_metric_view(feature_df) if not feature_df.empty else pd.DataFrame()

    summary = referee.make_referee_model_summary_table(grand)
    summary.insert(0, "dgp_label", dgp.label)
    summary.insert(1, "source_dir", str(source_dir))
    summary.to_csv(out_dir / "referee_model_summary.csv", index=False)

    skip_for_combined = [m for m in MODEL_ORDER if m not in models]
    if not long_metrics.empty:
        referee.make_importance_vs_gpbsv_pdf(
            long_metrics,
            scenario=scenario,
            out_pdf_path=str(combined_dir / "fig_importance_vs_gpbsv_all_models.pdf"),
            skip_models=skip_for_combined,
        )
        referee.make_importance_vs_gpbsv_avg_pdf(
            long_metrics,
            scenario=scenario,
            out_pdf_path=str(combined_dir / "fig_importance_vs_gpbsv_avg_all_models.pdf"),
            skip_models=skip_for_combined,
        )

        for model in models:
            model_dir = by_model_dir / model
            model_dir.mkdir(parents=True, exist_ok=True)
            skip_models = [m for m in MODEL_ORDER if m != model]
            referee.make_importance_vs_gpbsv_pdf(
                long_metrics,
                scenario=scenario,
                out_pdf_path=str(model_dir / f"fig_importance_vs_gpbsv_{model}.pdf"),
                skip_models=skip_models,
            )
            referee.make_importance_vs_gpbsv_avg_pdf(
                long_metrics,
                scenario=scenario,
                out_pdf_path=str(model_dir / f"fig_importance_vs_gpbsv_avg_{model}.pdf"),
                skip_models=skip_models,
            )

    referee.make_mas_quadrant_plot(
        grand,
        scenario=scenario,
        out_path=str(combined_dir / "fig_mas_quadrant_all_models.pdf"),
        skip_models=skip_for_combined,
    )
    referee.make_mas_distribution_plot(
        grand,
        scenario=scenario,
        out_path=str(combined_dir / "fig_mas_distribution_all_models.pdf"),
        skip_models=skip_for_combined,
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--models", nargs="+", choices=MODEL_ORDER, default=MODEL_ORDER)
    parser.add_argument("--include-appendix-break", action="store_true")
    parser.add_argument("--clean", action="store_true", help="Delete the output root before regenerating it.")
    args = parser.parse_args()

    dgps = _default_dgps(include_appendix_break=bool(args.include_appendix_break))
    models = [m for m in MODEL_ORDER if m in args.models]
    output_root = args.output_root.resolve()

    _validate_sources(dgps)
    _ensure_clean_output_root(output_root, clean=bool(args.clean))

    referee = _load_referee_module()
    summaries = [_make_one_dgp_figure_set(referee, dgp, output_root, models) for dgp in dgps]
    all_summary = pd.concat(summaries, ignore_index=True)
    all_summary.to_csv(output_root / "referee_model_summary_all_dgps.csv", index=False)

    manifest = {
        "output_root": str(output_root),
        "models": models,
        "dgps": [
            {
                **asdict(dgp),
                "scenario": dgp.scenario,
                "output_folder": dgp.output_folder,
                "source_dir": str(dgp.source_dir.resolve()),
            }
            for dgp in dgps
        ],
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_readme(output_root, dgps, models)

    print(f"Referee figure set written to: {output_root}")
    for dgp in dgps:
        print(f"  - {output_root / dgp.output_folder}")


if __name__ == "__main__":
    main()
