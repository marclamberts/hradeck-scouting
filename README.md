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
- Market tiers, readiness labels, and risk bands
- Executive overview with top targets and fit/value market map
- Ranked shortlist with score bars and risk/readiness context
- Detailed player lab with profile notes, score chart, per-90 metrics, and comparables
- Position/archetype score summaries
- CSV and PDF downloads for the filtered shortlist
