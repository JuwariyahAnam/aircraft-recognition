"""Run the whole project from the command line (the notebook does the same, step by step).

Real data on a GPU (Colab):
    python scripts/run_all.py --root /content/drive/MyDrive/aircraft --download

Quick smoke test on fake data (CPU, a few minutes):
    python scripts/make_synthetic_data.py --root ./smoke
    python scripts/run_all.py --root ./smoke --fast
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aircraft_recog import experiments as ex          # noqa: E402
from aircraft_recog import steps                      # noqa: E402
from aircraft_recog.config import Config, fast_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./workdir")
    ap.add_argument("--fast", action="store_true", help="tiny settings for a smoke test")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--skip-extras", action="store_true")
    ap.add_argument("--from-step", type=int, default=3)
    a = ap.parse_args()
    cfg = (fast_config(a.root) if a.fast else Config(root=Path(a.root))).make_dirs()
    cfg.save()
    if a.download:
        steps.step1_download(cfg)
    s = a.from_step
    if s <= 3: steps.step3_build_dataset(cfg, denoise=not a.fast)
    if s <= 5: print(steps.step5_segmentation(cfg)["test"])
    if s <= 6: print(steps.step6_baselines(cfg).to_string())
    if s <= 6: steps.step6b_fine_grained(cfg)
    if s <= 7: steps.step7_optimize(cfg)
    if s <= 8: print(steps.step8_semi_supervised(cfg))
    if s <= 9: steps.step9_explain(cfg, lime_n=1 if a.fast else 3)
    if not a.skip_extras:
        print(steps.extra_svm_baseline(cfg).to_string())
        print(steps.extra_synthesis(cfg, (8, 16, 32) if a.fast else (30, 75, 300, 750)).to_string())
        print(steps.extra_ablation(cfg).to_string())
    print("\n================ RESULTS TABLE ================")
    print(ex.results_table(cfg).to_string())
    print("\nAll figures and tables are in", cfg.results)


if __name__ == "__main__":
    main()
