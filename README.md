# FCHK Scouting Streamlit App

Local scouting workbench for `FCHK Scores Only.xlsx`.

## Run

```bash
python3 -m pip install -r requirements.txt
streamlit run app.py
```

The app opens with the bundled workbook in `data/FCHK Scores Only.xlsx`.

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
