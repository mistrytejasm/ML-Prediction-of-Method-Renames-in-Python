One folder per experiment run: results/runs/<YYYY-MM-DD>_<prefix>_<run id>/, written once and never
overwritten (see src/evaluation/run_store.py). Stage 8 runs contain run_config.json, provenance.json,
fold_metrics.csv, oof_predictions.csv, summary.json and warnings.json; failed runs end in __FAILED;
.tmp_* folders are interrupted writes, never results. How to run and read them:
docs/stage8_execution_guide.md. Earlier results stay in results/legacy_experiments/.
