import time
import pandas as pd
from scipy.stats import ks_2samp
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import FloatType
from nba_api.stats.endpoints import shotchartdetail

# ---------------------------------------------------------
# 1. INITIALIZE PYSPARK 
# ---------------------------------------------------------
# Setting up a local Spark session to demonstrate scalable data handling.
spark = SparkSession.builder \
    .appName("Spatiotemporal_Drift_Monitor") \
    .master("local[*]") \
    .getOrCreate()

# Disable verbose Spark logging for a cleaner console output
spark.sparkContext.setLogLevel("ERROR")

# ---------------------------------------------------------
# 2. INGESTION (NBA API)
# ---------------------------------------------------------
def fetch_shot_data(player_id, team_id, context_measure, season_nullable="2023-24"):
    """
    Pulls raw X/Y coordinate shot data. 
    Using custom headers to prevent API timeouts/blocks.
    """
    headers = {
        'Host': 'stats.nba.com',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.5',
    }
    
    try:
        shot_data = shotchartdetail.ShotChartDetail(
            team_id=team_id,
            player_id=player_id,
            context_measure_simple=context_measure,
            season_nullable=season_nullable,
            season_type_all_star="Regular Season"
        )
        # Sleep to respect API rate limits (the unglamorous part of data engineering)
        time.sleep(1) 
        return shot_data.get_data_frames()[0]
    except Exception as e:
        print(f"[ERROR] Failed to pull data: {e}")
        return pd.DataFrame()

# ---------------------------------------------------------
# 3. PYSPARK TRANSFORMATION
# ---------------------------------------------------------
def process_spatial_data(pdf, period_name):
    """
    Converts Pandas to PySpark, cleans messy coordinates, 
    and calculates true Euclidean distance from the hoop.
    """
    print(f"[INFO] Processing {period_name} spatiotemporal data via PySpark...")
    
    # Drop NAs to prevent pipeline breaking
    pdf = pdf.dropna(subset=['LOC_X', 'LOC_Y'])
    
    # Load into Spark
    df = spark.createDataFrame(pdf)
    
    # NBA API maps court coordinates in tenths of a foot.
    # We calculate the true distance (Hypotenuse) from the basket (0,0)
    df = df.withColumn("LOC_X", F.col("LOC_X").cast(FloatType())) \
           .withColumn("LOC_Y", F.col("LOC_Y").cast(FloatType())) \
           .withColumn("TRUE_SHOT_DISTANCE", 
                       F.sqrt(F.pow(F.col("LOC_X"), 2) + F.pow(F.col("LOC_Y"), 2)) / 10)
    
    return df

# ---------------------------------------------------------
# 4. DRIFT DETECTION (THE MLOps COMPONENT)
# ---------------------------------------------------------
def detect_model_drift(baseline_df, current_df, p_value_threshold=0.05):
    """
    Extracts the spatial distributions and runs a Kolmogorov-Smirnov test 
    to detect if the underlying data generating process has changed.
    """
    # Extract just the feature we are monitoring into local memory for the stat test
    baseline_dist = [row['TRUE_SHOT_DISTANCE'] for row in baseline_df.select("TRUE_SHOT_DISTANCE").collect()]
    current_dist = [row['TRUE_SHOT_DISTANCE'] for row in current_df.select("TRUE_SHOT_DISTANCE").collect()]
    
    if not baseline_dist or not current_dist:
        print("[ERROR] Insufficient data for drift detection.")
        return
        
    # Perform Two-Sample KS Test
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
    
    # Target: Paolo Banchero (ID: 1631094), Orlando Magic (ID: 1610612753)
    # Simulating a split season to check for drift
    
    # 1. Pull Data
    raw_data = fetch_shot_data(player_id=1631094, team_id=1610612753, context_measure='FGA')
    
    if not raw_data.empty:
        # Sort by game date to simulate a time-series flow
        raw_data = raw_data.sort_values(by='GAME_DATE')
        
        # Split into "Baseline" (first half of season) and "Current" (second half)
        midpoint = len(raw_data) // 2
        baseline_pdf = raw_data.iloc[:midpoint]
        current_pdf = raw_data.iloc[midpoint:]
        
        # 2. Process in PySpark
        baseline_spark = process_spatial_data(baseline_pdf, "Baseline")
        current_spark = process_spatial_data(current_pdf, "Current")
        
        # 3. Monitor for Drift
        detect_model_drift(baseline_spark, current_spark)
        
    else:
        print("[ERROR] No data returned from NBA API.")
        
    spark.stop()
