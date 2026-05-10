# Code Review: 3_QC_Segmentation.py

**Status: ON HOLD — no changes made yet**

## Error

```
KeyError: ['MRR_analysis_id_mm']
File "pages/3_QC_Segmentation.py", line 543, in <module>
    st.session_state.df_outliers.dropna(subset=[asys_id_col], inplace=True)
```

## Root Cause

`asys_id_col` is constructed dynamically from the sidebar tool selection:

```python
prefix      = "ra" if segmentation_tool == "recon-all-clinical" else "mm"
asys_id_col = f"MRR_analysis_id_{prefix}"
st.session_state.df_outliers.dropna(subset=[asys_id_col], inplace=True)
```

`pandas.DataFrame.dropna(subset=[...])` raises `KeyError` if any column in `subset` is absent from the DataFrame. If the uploaded CSV does not contain `MRR_analysis_id_mm` (minimorph) or `MRR_analysis_id_ra` (recon-all-clinical), the crash is immediate and uncaught.

## Contributing Factors

1. **No pre-flight column validation (primary — line ~543)**: No guard checks whether `asys_id_col` exists in `df_outliers.columns` before calling `dropna`.
2. **No validation at upload time (line ~470)**: CSV is accepted unconditionally; required columns are never verified at load time. Error only surfaces deep inside the prepare block.
3. **Silent skip for multi-project CSVs (line ~479)**: The entire download/prepare block is nested under `if len(project_labels) == 1`. Multi-project CSVs silently skip the block; `data_prepared` stays `False`, causing an infinite rerun with no user feedback.
4. **Tool/CSV mismatch not surfaced**: No cross-check that the selected tool has a corresponding non-empty ID column in the CSV.

## Resolution Plan

| # | Location | Fix |
|---|----------|-----|
| 1 | Line ~543 | Check `asys_id_col in df_outliers.columns`; if absent `st.error(...)` + `st.stop()` |
| 2 | Line ~470 | After `pd.read_csv`, validate `subject`, `session`, `project` present; warn if neither ID column exists |
| 3 | Line ~479 | Add `else: st.error(...) + st.stop()` for `len(project_labels) != 1` case |
| 4 | Line ~545 | After `dropna`, if DataFrame is now empty warn user before entering download loop |
