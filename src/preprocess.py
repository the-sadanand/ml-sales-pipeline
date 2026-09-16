import argparse
import logging
import pandas as pd
import yaml
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_params(params_path: str = "params.yaml") -> dict:
    """Load parameters from yaml file."""
    with open(params_path, 'r') as f:
        return yaml.safe_load(f)

def preprocess_data(params: dict) -> None:
    """Preprocess the raw sales data."""
    # Paths relative to project root
    raw_dir = Path(params['data']['raw_dir'])
    processed_dir = Path(params['data']['processed_dir'])
    
    # Ensure processed directory exists
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = raw_dir / 'train.csv'
    store_path = raw_dir / 'store.csv'
    output_path = processed_dir / 'train_cleaned.csv'
    
    logger.info(f"Loading data from {raw_dir}")
    train_df = pd.read_csv(train_path, low_memory=False)
    store_df = pd.read_csv(store_path)
    
    initial_rows = len(train_df)
    logger.info(f"Initial train data rows: {initial_rows}")
    
    # Merge on Store
    logger.info("Merging train and store data")
    df = train_df.merge(store_df, on='Store', how='left')
    
    # Cleaning
    logger.info("Cleaning data (removing closed stores, zero sales)")
    # Remove closed stores and 0 sales
    df = df[(df['Open'] != 0) & (df['Sales'] > 0)].copy()
    
    # Fill missing values
    logger.info("Filling missing values")
    comp_dist_median = df['CompetitionDistance'].median()
    df['CompetitionDistance'] = df['CompetitionDistance'].fillna(comp_dist_median)
    
    fill_zeros = ['CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear', 
                  'Promo2SinceWeek', 'Promo2SinceYear']
    for col in fill_zeros:
        df[col] = df[col].fillna(0)
        
    df['PromoInterval'] = df['PromoInterval'].fillna('None')
    df['StateHoliday'] = df['StateHoliday'].astype(str)
    
    # Parse date and sort
    logger.info("Parsing dates and sorting")
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')
    
    # Drop Customers
    if 'Customers' in df.columns:
        logger.info("Dropping 'Customers' column")
        df = df.drop(columns=['Customers'])
        
    final_rows = len(df)
    logger.info(f"Final cleaned data rows: {final_rows}")
    logger.info(f"Removed {initial_rows - final_rows} rows")
    
    # Save processed data
    logger.info(f"Saving cleaned data to {output_path}")
    df.to_csv(output_path, index=False)
    logger.info("Preprocessing completed successfully")

def main():
    parser = argparse.ArgumentParser(description="Preprocess raw Rossmann sales data")
    parser.add_argument("--params", type=str, default="params.yaml", help="Path to parameters file")
    args = parser.parse_args()
    
    try:
        params = load_params(args.params)
        preprocess_data(params)
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise

if __name__ == "__main__":
    main()
