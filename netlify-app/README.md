# Lamberts Index — Netlify App

Static single-page scouting app for FC Hradec Králové.

## Deploy to Netlify

1. Connect this repo to Netlify
2. Set **Base directory** to `netlify-app`
3. Set **Publish directory** to `netlify-app/public`
4. No build command needed (static files)

## Regenerate data

```bash
python netlify-app/build_data.py
```

Run this whenever `data/Lamberts_Index_Full_Report.xlsx` is updated, then commit the new JSON files.

## Structure

```
netlify-app/
  public/
    index.html        # Single-page app
    data/
      meta.json
      all_players.json
      gk.json  cb.json  fb.json  dm.json  cm.json  w.json  fw.json
      priority_list.json
      elite_picks.json
      expiring_2026.json
      league_analysis.json
  build_data.py       # Excel → JSON exporter
  netlify.toml        # Netlify config
```
