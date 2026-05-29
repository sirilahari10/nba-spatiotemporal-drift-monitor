# NBA Spatiotemporal Drift Monitor

Most predictive basketball models don't fail with a loud error code; they fail silently because the underlying spatiotemporal data distribution shifted. 

I built this lightweight PySpark pipeline to demonstrate how I catch these "silent failures" in production. 

Rather than just building a model, this repo focuses on the unglamorous plumbing: ingesting raw tracking data, scaling the transformations via PySpark, and implementing a statistical drift monitor to alert MLOps teams when a player's behavior fundamentally changes.

## Architecture

1. **Ingestion:** Pulls raw shot coordinate data (`LOC_X`, `LOC_Y`) using the `nba_api`.
2. **PySpark Engine:** Processes the spatial coordinates, calculating the true Euclidean distance from the hoop ($d = \sqrt{x^2 + y^2}$).
3. **Drift Detection:** Compares a "baseline" time period against a "current" time period using a Two-Sample Kolmogorov-Smirnov (KS) test. 

## The Scenario: Paolo Banchero's Shot Diet
To test the pipeline, I targeted Paolo Banchero's shot data, comparing his early-season shot distribution to his late-season distribution. 

If a player suddenly starts taking 30% more deep threes, a model trained on early-season data will silently degrade. This pipeline catches that drift.
