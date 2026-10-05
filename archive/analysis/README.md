# Archived diagnosis and plotting utilities

These scripts are kept for reproducing earlier analyses. Run them from the
`quadlocofault` repository root. Outputs and source data remain under `logs/`.

- `diagnose_faults.py` captures first-episode fault traces using Isaac Lab.
- `analyze_fault_diagnosis.py` analyzes those traces offline.
- `summarize_fault_study.py` summarizes the September 27 diagnostic study.
- `analyze_actor_reflection.py` measures checkpoint reflection errors offline.
- `plot_flat_timeseries.py` plots saved flat-ground recovery traces.
- `record_v83_complete_fault_videos.sh` and `record_v83_complete_fault_videos_15s_marker.sh` reproduce the v8.3 joint-by-joint videos.

For example:

```bash
python archive/analysis/analyze_fault_diagnosis.py --help
python archive/analysis/plot_flat_timeseries.py --help
```
