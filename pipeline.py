import time
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
from nba_api.stats.endpoints import shotchartdetail

# ---------------------------------------------------------
# 1. INITIALIZE PYSPARK 
# ---------------------------------------------------------
spark = SparkSession.builder \
    .appName("Spatiotemporal_Drift_Monitor") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# ---------------------------------------------------------
# 2. INGESTION (NBA API w/ GRACEFUL DEGRADATION)
# ---------------------------------------------------------
def fetch_shot_data(player_id, team_id, context_measure, season_nullable="2023-24"):
    """
    Pulls raw X/Y coordinate shot data. 
    Includes advanced headers to bypass NBA.com bot protection.
    Falls back to a synthetic mock dataset if the API times out, 
    ensuring the pipeline can still be demonstrated.
    """
    advanced_headers = {
        'Host': 'stats.nba.com',
        'Connection': 'keep-alive',
        'Accept': 'application/json, text/plain, */*',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Referer': 'https://www.nba.com/',
        'Origin': 'https://www.nba.com',
        'x-nba-stats-origin': 'stats',
        'x-nba-stats-token': 'true',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    
    print("[INFO] Attempting to connect to NBA API...")
    try:
        shot_data = shotchartdetail.ShotChartDetail(
            team_id=team_id,
            player_id=player_id,
            context_measure_simple=context_measure,
            season_nullable=season_nullable,
            season_type_all_star="Regular Season",
            headers=advanced_headers,
            timeout=10 # Fail fast instead of waiting 30 seconds
        )
        time.sleep(1) 
        df = shot_data.get_data_frames()[0]
        if not df.empty:
            print("[INFO] Successfully retrieved data from NBA API.")
            return df
    except Exception as e:
        print(f"[WARN] NBA API connection blocked or timed out.")
        
    print("[INFO] Gracefully falling back to generating synthetic tracking data for the MLOps demonstration...")
    
    # Generate realistic mock data mimicking the NBA API output format
    np.random.seed(42)
    total_shots = 600
    
    # Baseline: mostly close to the basket (e.g. earlier in season)
    loc_x_base = np.random.normal(0, 50, total_shots // 2)
    loc_y_base = np.random.normal(50, 40, total_shots // 2)
    
    # Current (drifted): shifted further away to trigger the MLOps alert
    loc_x_curr = np.random.normal(0, 80, total_shots // 2)
    loc_y_curr = np.random.normal(200, 50, total_shots // 2)
    
    dates = pd.date_range(start="2023-10-25", periods=total_shots, freq="h").astype(str)
    
    mock_df = pd.DataFrame({
        'GAME_DATE': dates,
        'LOC_X': np.concatenate([loc_x_base, loc_x_curr]),
        'LOC_Y': np.concatenate([loc_y_base, loc_y_curr])
    })
    
    return mock_df

# ---------------------------------------------------------
# 3. PYSPARK TRANSFORMATION
# ---------------------------------------------------------
def process_spatial_data(pdf, period_name):
    print(f"[INFO] Processing {period_name} spatiotemporal data via PySpark...")
    pdf = pdf.dropna(subset=['LOC_X', 'LOC_Y'])
    df = spark.createDataFrame(pdf)
    
    # Calculate the true distance (Hypotenuse) from the basket (0,0)
    df = df.withColumn("LOC_X", F.col("LOC_X").cast(FloatType())) \
           .withColumn("LOC_Y", F.col("LOC_Y").cast(FloatType())) \
           .withColumn("TRUE_SHOT_DISTANCE", 
                       F.sqrt(F.pow(F.col("LOC_X"), 2) + F.pow(F.col("LOC_Y"), 2)) / 10)
    return df

# ---------------------------------------------------------
# 4. DRIFT DETECTION (THE MLOps COMPONENT)
# ---------------------------------------------------------
def detect_model_drift(baseline_df, current_df, p_value_threshold=0.05):
    baseline_dist = [row['TRUE_SHOT_DISTANCE'] for row in baseline_df.select("TRUE_SHOT_DISTANCE").collect()]
    current_dist = [row['TRUE_SHOT_DISTANCE'] for row in current_df.select("TRUE_SHOT_DISTANCE").collect()]
    
    if not baseline_dist or not current_dist:
        print("[ERROR] Insufficient data for drift detection.")
        return
        
    stat, p_val = ks_2samp(baseline_dist, current_dist)
    
    print("-" * 50)
    if p_val < p_value_threshold:
        print("[ALERT] SILENT FAILURE / DATA DRIFT DETECTED!")
        print(f"KS Statistic: {stat:.3f} | P-Value: {p_val:.4f}")
        print("The spatial distribution of shots has fundamentally shifted.")
        print("ACTION REQUIRED: Production models relying on this feature require immediate retraining.")
    else:
        print("[SUCCESS] Pipeline Healthy. No significant data drift detected.")
        print(f"KS Statistic: {stat:.3f} | P-Value: {p_val:.4f}")
    print("-" * 50)

# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    print("-" * 50)
    print("STARTING SPATIOTEMPORAL PIPELINE")
    print("-" * 50)
    
    raw_data = fetch_shot_data(player_id=1631094, team_id=1610612753, context_measure='FGA')
    
    if not raw_data.empty:
        raw_data = raw_data.sort_values(by='GAME_DATE')
        
        midpoint = len(raw_data) // 2
        baseline_pdf = raw_data.iloc[:midpoint]
        current_pdf = raw_data.iloc[midpoint:]
        
        baseline_spark = process_spatial_data(baseline_pdf, "Baseline")
        current_spark = process_spatial_data(current_pdf, "Current")
        
        detect_model_drift(baseline_spark, current_spark)
        
    else:
        print("[ERROR] No data generated.")
        
    spark.stop()
