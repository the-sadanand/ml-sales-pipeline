import pandas as pd
import numpy as np
import os
import yaml
import logging
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_store_data(num_stores: int = 1115, random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic store data."""
    np.random.seed(random_state)
    stores = pd.DataFrame({'Store': range(1, num_stores + 1)})
    stores['StoreType'] = np.random.choice(['a', 'b', 'c', 'd'], size=num_stores, p=[0.5, 0.1, 0.3, 0.1])
    stores['Assortment'] = np.random.choice(['a', 'b', 'c'], size=num_stores, p=[0.5, 0.15, 0.35])
    
    comp_dist = np.random.exponential(scale=5000, size=num_stores)
    comp_dist[np.random.rand(num_stores) < 0.1] = np.nan
    stores['CompetitionDistance'] = comp_dist
    
    comp_month = np.random.randint(1, 13, size=num_stores).astype(float)
    comp_month[np.random.rand(num_stores) < 0.2] = np.nan
    stores['CompetitionOpenSinceMonth'] = comp_month
    
    comp_year = np.random.randint(2000, 2016, size=num_stores).astype(float)
    comp_year[np.isnan(comp_month)] = np.nan
    stores['CompetitionOpenSinceYear'] = comp_year
    
    promo2 = np.random.randint(0, 2, size=num_stores)
    stores['Promo2'] = promo2
    
    promo2_week = np.random.randint(1, 53, size=num_stores).astype(float)
    promo2_week[promo2 == 0] = np.nan
    stores['Promo2SinceWeek'] = promo2_week
    
    promo2_year = np.random.randint(2009, 2016, size=num_stores).astype(float)
    promo2_year[promo2 == 0] = np.nan
    stores['Promo2SinceYear'] = promo2_year
    
    intervals = ['Jan,Apr,Jul,Oct', 'Feb,May,Aug,Nov', 'Mar,Jun,Sept,Dec']
    promo_interval = np.random.choice(intervals, size=num_stores)
    stores['PromoInterval'] = pd.Series(promo_interval).where(promo2 == 1, other=None)
    
    return stores

def generate_train_data(num_rows: int = 100000, num_stores: int = 1115, random_state: int = 42) -> pd.DataFrame:
    """Generate synthetic train data."""
    np.random.seed(random_state + 1) # slightly different seed for train
    dates = pd.date_range(start='2013-01-01', end='2015-07-31')
    
    random_dates = np.random.choice(dates, size=num_rows)
    random_stores = np.random.randint(1, num_stores + 1, size=num_rows)
    
    train = pd.DataFrame({
        'Store': random_stores,
        'Date': random_dates
    })
    
    train['DayOfWeek'] = train['Date'].dt.dayofweek + 1
    
    train['Open'] = np.random.choice([0, 1], size=num_rows, p=[0.1, 0.9])
    
    # Always closed on some Sundays
    sunday_mask = train['DayOfWeek'] == 7
    train.loc[sunday_mask, 'Open'] = np.random.choice([0, 1], size=sunday_mask.sum(), p=[0.95, 0.05])
    
    train['Promo'] = np.random.choice([0, 1], size=num_rows, p=[0.6, 0.4])
    train['StateHoliday'] = np.random.choice(['0', 'a', 'b', 'c'], size=num_rows, p=[0.97, 0.01, 0.01, 0.01])
    train['SchoolHoliday'] = np.random.choice([0, 1], size=num_rows, p=[0.85, 0.15])
    
    base_sales = np.random.lognormal(mean=8.5, sigma=0.5, size=num_rows)
    train['Sales'] = base_sales
    train.loc[train['Promo'] == 1, 'Sales'] *= 1.5
    train.loc[train['Open'] == 0, 'Sales'] = 0
    train['Sales'] = train['Sales'].astype(int)
    
    customers = train['Sales'] / 10 + np.random.normal(0, 10, size=num_rows)
    customers[customers < 0] = 0
    train['Customers'] = customers.astype(int)
    train.loc[train['Open'] == 0, 'Customers'] = 0
    
    # Format Date column
    train['Date'] = train['Date'].dt.strftime('%Y-%m-%d')
    
    return train

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic Rossmann dataset.")
    parser.add_argument('--config', type=str, default='params.yaml', help='Path to params.yaml')
    args = parser.parse_args()

    try:
        project_root = Path(__file__).resolve().parents[1]
        params_path = project_root / args.config
        
        logger.info(f"Reading configuration from {params_path}")
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
            
        data_params = params.get('data', {})
        random_state = data_params.get('random_state', 42)
        raw_dir = project_root / data_params.get('raw_dir', 'data/raw')
        
        os.makedirs(raw_dir, exist_ok=True)
        
        logger.info("Generating store.csv...")
        store_df = generate_store_data(random_state=random_state)
        
        logger.info("Generating train.csv...")
        train_df = generate_train_data(random_state=random_state)
        
        store_path = raw_dir / 'store.csv'
        train_path = raw_dir / 'train.csv'
        
        store_df.to_csv(store_path, index=False)
        train_df.to_csv(train_path, index=False)
        
        logger.info(f"Store data saved to {store_path}")
        logger.info(f"Train data saved to {train_path}")
        
        logger.info("Summary Statistics:")
        logger.info(f"Store shape: {store_df.shape}")
        logger.info(f"Train shape: {train_df.shape}")
        logger.info(f"Total Sales: {train_df['Sales'].sum()}")
        logger.info(f"Total Customers: {train_df['Customers'].sum()}")
        
    except Exception as e:
        logger.error(f"Error during data generation: {e}")
        raise

if __name__ == '__main__':
    main()
