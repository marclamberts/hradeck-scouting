# FCHK Scouting Streamlit App

Local scouting workbench for the FCHK Model V3 Excel outputs.

## Run

```bash
python3 -m pip install -r requirements.txt
streamlit run app.py
```

The app opens with the FCHK Model V3 output folder:

```text
/Users/user/Downloads/01_Football_Analytics/Data_and_Models/RModel/outputs
```

It reads the following workbooks when present:

- `FCHK Model V3 - Recruitment Scores.xlsx`
- `FCHK Model V3 - Player Scores.xlsx`
- `FCHK Model V3 - Player Styles.xlsx`
- `FCHK Model V3 - Loaded Leagues.xlsx`
- `FCHK Model V3 - Model Input.xlsx`
- `FCHK Model V3 - Smart Club Closeness.xlsx`
- `FCHK Model V3 - Summary.xlsx`
- `FCHK Model V3 Scores.xlsx`

To point the app at a different output folder, set `FCHK_MODEL_OUTPUT_DIR` before launching Streamlit. If the V3 recruitment workbook is unavailable, the app falls back to the bundled workbook in `data/FCHK Scores Only.xlsx`.

## Included

- Position, league/bundle, archetype, age, minutes, U23, and score filters
- Preset and adjustable Scout Fit score weights
- Role-specific fit scoring for every position group
- Market tiers, readiness labels, and risk bands
- Executive overview with top targets and fit/value market map
- Dedicated position ranking boards
- Ranked shortlist with score bars and risk/readiness context
- Manual shortlist basket with CSV/PDF export
- Player comparison mode with side-by-side profile pizzas
- Similarity search for finding statistical comparables
- Detailed player lab with profile notes, score chart, per-90 metrics, and comparables
- Advanced matplotlib and mplsoccer visuals
- League, position, player, shortlist, and filtered-board PDF reports
- Position/archetype score summaries
- CSV and PDF downloads for the filtered shortlist

## Deploy

This repo includes `.streamlit/config.toml` for a clean light theme. For Streamlit Community Cloud, set the app entrypoint to `app.py`.
