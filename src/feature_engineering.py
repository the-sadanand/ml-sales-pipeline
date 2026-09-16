import pandas as pd
import numpy as np
import yaml
import logging
import argparse
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_params(params_path="params.yaml"):
    with open(params_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Stage 2: Feature Engineering")
    parser.add_argument("--config", default="params.yaml", help="Path to params.yaml")
    args = parser.parse_args()

    params = load_params(args.config)
    
    # Paths
    processed_dir = Path(params['data']['processed_dir'])
    features_dir = Path(params['data']['features_dir'])
    features_dir.mkdir(parents=True, exist_ok=True)
    
    input_file = processed_dir / "train_cleaned.csv"
    logger.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)
    
    # Ensure Date is datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        logger.error("Date column not found in data!")
        return
    
    # Feature flags
    date_features = params['features'].get('date_features', True)
    competition_features = params['features'].get('competition_features', True)
    promo_features = params['features'].get('promo_features', True)
    
    # Date processing
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    
    if date_features:
        logger.info("Extracting date features")
        df['Day'] = df['Date'].dt.day
    
    if competition_features:
        logger.info("Extracting competition features")
        if 'CompetitionOpenSinceYear' in df.columns and 'CompetitionOpenSinceMonth' in df.columns:
            # Months since competition opened
            df['CompetitionOpen'] = 12 * (df['Year'] - df['CompetitionOpenSinceYear']) + (df['Month'] - df['CompetitionOpenSinceMonth'])
            df['CompetitionOpen'] = df['CompetitionOpen'].apply(lambda x: x if x > 0 else 0).fillna(0)
        else:
            logger.warning("Competition columns missing, skipping competition features.")
        
    if promo_features:
        logger.info("Extracting promo features")
        if 'Promo2SinceYear' in df.columns and 'Promo2SinceWeek' in df.columns:
            # Months since Promo2 started
            df['Promo2Open'] = 12 * (df['Year'] - df['Promo2SinceYear']) + (df['WeekOfYear'] - df['Promo2SinceWeek']) / 4.0
            df['Promo2Open'] = df['Promo2Open'].apply(lambda x: x if x > 0 else 0).fillna(0)
        else:
            logger.warning("Promo2 columns missing, skipping Promo2Open feature.")
        
        # Is Promo2 active
        if 'PromoInterval' in df.columns:
            month2str = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 
                         7:'Jul', 8:'Aug', 9:'Sept', 10:'Oct', 11:'Nov', 12:'Dec'}
            df['month_str'] = df['Month'].map(month2str)
            
            def is_promo_active(row):
                if pd.isna(row['PromoInterval']) or pd.isna(row['month_str']):
                    return 0
                if row['month_str'] in str(row['PromoInterval']):
                    return 1
                return 0
                
            df['IsPromo2Active'] = df.apply(is_promo_active, axis=1)
            df.drop(columns=['month_str'], inplace=True)
        else:
            logger.warning("PromoInterval column missing, skipping IsPromo2Active feature.")
        
    # Encoding
    logger.info("Encoding categorical variables")
    # One-hot encode StoreType (a,b,c,d)
    if 'StoreType' in df.columns:
        df = pd.get_dummies(df, columns=['StoreType'], prefix='StoreType', drop_first=False, dtype=int)
        
    # One-hot encode Assortment (a,b,c)
    if 'Assortment' in df.columns:
        df = pd.get_dummies(df, columns=['Assortment'], prefix='Assortment', drop_first=False, dtype=int)
        
    # Map StateHoliday: '0'->0, 'a'->1, 'b'->2, 'c'->3
    if 'StateHoliday' in df.columns:
        state_holiday_map = {'0': 0, 0: 0, 'a': 1, 'b': 2, 'c': 3}
        df['StateHoliday'] = df['StateHoliday'].map(state_holiday_map).fillna(0)
        
    # Drop original categorical columns
    cols_to_drop = ['Date', 'PromoInterval']
    
    # If date_features is False, drop the date components we created just for logic
    if not date_features:
        cols_to_drop.extend(['Year', 'Month', 'WeekOfYear', 'Day'])
        
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    logger.info(f"Dropping columns: {cols_to_drop}")
    df.drop(columns=cols_to_drop, inplace=True)
    
    # Train/test split
    test_size = params['data'].get('test_size', 0.2)
    random_state = params['data'].get('random_state', 42)
    
    logger.info(f"Splitting data with test_size={test_size} and random_state={random_state}")
    
    target_col = 'Sales'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
        
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Save combined
    train_out = X_train.copy()
    train_out[target_col] = y_train
    
    test_out = X_test.copy()
    test_out[target_col] = y_test
    
    train_out_path = features_dir / "train_features.csv"
    test_out_path = features_dir / "test_features.csv"
    
    logger.info(f"Saving train features to {train_out_path} (shape: {train_out.shape})")
    train_out.to_csv(train_out_path, index=False)
    
    logger.info(f"Saving test features to {test_out_path} (shape: {test_out.shape})")
    test_out.to_csv(test_out_path, index=False)
    
    logger.info(f"Feature engineering complete! Train features: {list(train_out.columns)}")

if __name__ == "__main__":
    main()
