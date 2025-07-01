# preprocessing.py

from pathlib import Path
import os
import pandas as pd
import numpy as np
import re
from datetime import datetime

def read_data(data_path, table_path):
    def read_file_by_extension(file_path):
        """Read file based on its extension"""
        file_extension = file_path.lower().split('.')[-1]
        
        if file_extension == 'xlsx':
            return pd.read_excel(file_path, engine='openpyxl')
        elif file_extension == 'xls':
            return pd.read_excel(file_path, engine='xlrd')
        elif file_extension == 'csv':
            return pd.read_csv(file_path)
        elif file_extension == 'json':
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}. Supported formats: xlsx, xls, csv, json")
    
    pre_df = read_file_by_extension(data_path)
    table = read_file_by_extension(table_path)

    pre_df.rename(columns={'Occcupation': 'Occupation'}, inplace=True)
    table.rename(columns={'Project ID': 'projectid'}, inplace=True)

    df = pd.merge(pre_df, table, on='projectid', how='left')
    return df

# def sample_projects(df):
#     df_sampled = df.groupby('projectid', group_keys=False).apply(
#         lambda x: x.sample(n=min(len(x), 2), random_state=42)
#     )
#     return df_sampled.copy()

def fix_year(dt_str):
    if pd.isna(dt_str):
        return pd.NaT
    try:
        parts = str(dt_str).split('-')
        fixed_year = int(parts[0]) - 543
        return pd.to_datetime(f"{fixed_year}-{'-'.join(parts[1:])}")
    except:
        return pd.NaT

def preprocess_dates(df):
    """Process dates by converting Buddhist calendar to Gregorian"""
    df = df.copy()
    
    # Process questiondate
    if 'questiondate' in df.columns:
        df['questiondate'] = df['questiondate'].apply(fix_year)
        
    # Process bookingdate  
    if 'bookingdate' in df.columns:
        df['bookingdate'] = df['bookingdate'].apply(fix_year)
    
    # Add has_booked column
    df['has_booked'] = df['bookingdate'].notna().astype(int)
    return df

def add_seasonal_features(df, date_column):
    """Add seasonal features from a datetime column"""
    if date_column not in df.columns:
        return df
    
    # Ensure the column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    
    # Only process rows with valid dates
    valid_mask = df[date_column].notna()
    
    if not valid_mask.any():
        return df
    
    # Add time-based features
    df['hour'] = np.where(valid_mask, df[date_column].dt.hour, np.nan)
    df['hour_sin'] = np.where(valid_mask, np.sin(2 * np.pi * df['hour'] / 24), np.nan)
    df['hour_cos'] = np.where(valid_mask, np.cos(2 * np.pi * df['hour'] / 24), np.nan)
    df['day'] = np.where(valid_mask, df[date_column].dt.day, np.nan)
    df['month'] = np.where(valid_mask, df[date_column].dt.month, np.nan)
    df['quarter'] = np.where(valid_mask, df[date_column].dt.quarter, np.nan)
    df['year'] = np.where(valid_mask, df[date_column].dt.year, np.nan)
    df['week'] = np.where(valid_mask, df[date_column].dt.isocalendar().week, np.nan)
    df['day_of_week'] = np.where(valid_mask, df[date_column].dt.dayofweek, np.nan)
    df['is_weekend'] = np.where(valid_mask, (df[date_column].dt.dayofweek >= 5).astype(int), 0)
    df['day_of_year'] = np.where(valid_mask, df[date_column].dt.dayofyear, np.nan)
    df['season'] = np.where(valid_mask, df[date_column].dt.month % 12 // 3 + 1, np.nan)
    
    # Drop temporary hour column
    df.drop(columns=['hour'], inplace=True)
    
    return df

def create_bin_assignment_functions():
    budget_bins = [
        (0, 1.01, "≤ 1.0M"),
        (1.01, 1.51, "1.01 - 1.5M"),
        (1.51, 2.01, "1.51 - 2.0M"),
        (2.01, 2.51, "2.01 - 2.5M"),
        (2.51, 3.01, "2.51 - 3.0M"),
        (3.01, 3.51, "3.01 - 3.5M"),
        (3.51, 4.01, "3.51 - 4.0M"),
        (4.01, 4.51, "4.01 - 4.5M"),
        (4.51, 5.01, "4.51 - 5.0M"),
        (5.01, 6.01, "5.01 - 6.0M"),
        (6.01, 7.01, "6.01 - 7.0M"),
        (7.01, 8.01, "7.01 - 8.0M"),
        (8.01, 9.01, "8.01 - 9.0M"),
        (9.01, 10.01, "9.01 - 10.0M"),
        (10.01, 11.01, "10.01 - 11.0M"),
        (11.01, 12.01, "11.01 - 12.0M"),
        (12.01, 13.01, "12.01 - 13.0M"),
        (13.01, 14.01, "13.01 - 14.0M"),
        (14.01, 15.01, "14.01 - 15.0M"),
        (15.01, 16.01, "15.01 - 16.0M"),
        (16.01, 17.01, "16.01 - 17.0M"),
        (17.01, 20.01, "17.01 - 20.0M"),
        (20.01, 25.01, "20.01 - 25.0M"),
        (25.01, float("inf"), "≥ 25.01M")
    ]
    income_bins = [
        (0, 20001, '≤ 20,000'),
        (20001, 35001, '20,001 - 35,000'),
        (35001, 50001, '35,001 - 50,000'),
        (50001, 65001, '50,001 - 65,000'),
        (65001, 80001, '65,001 - 80,000'),
        (80001, 100001, '80,001 - 100,000'),
        (100001, 120001, '100,001 - 120,000'),
        (120001, 140001, '120,001 - 140,000'),
        (140001, 160001, '140,001 - 160,000'),
        (160001, 180001, '160,001 - 180,000'),
        (180001, 200001, '180,001 - 200,000'),
        (200001, 300001, '200,001 - 300,000'),
        (300001, 400001, '300,001 - 400,000'),
        (400001, float('inf'), '≥ 400,001'),
    ]
    individual_income_bins = [
        (0, 15001, '≤ 15,000'),
        (15001, 20001, '15,001 - 20,000'),
        (20001, 30001, '20,001 - 30,000'),
        (30001, 40001, '30,001 - 40,000'),
        (40001, 50001, '40,001 - 50,000'),
        (50001, 65001, '50,001 - 65,000'),
        (65001, 80001, '65,001 - 80,000'),
        (80001, 100001, '80,001 - 100,000'),
        (100001, 120001, '100,001 - 120,000'),
        (120001, 150001, '120,001 - 150,000'),
        (150001, 200001, '150,001 - 200,000'),
        (200001, 300001, '200,001 - 300,000'),
        (300001, 400001, '300,001 - 400,000'),
        (400001, float('inf'), '≥ 400,001'),
    ]

    def parse_value(val):
        if pd.isna(val):
            return np.nan
        val = str(val).replace(',', '').replace('บาท', '').replace('ล้าน', '').strip()
        if 'ไม่เกิน' in val or 'น้อยกว่า' in val:
            nums = re.findall(r'\d+\.\d+|\d+', val)
            return float(nums[0]) - 0.01 if nums else np.nan
        elif 'มากกว่า' in val or 'ขึ้นไป' in val:
            nums = re.findall(r'\d+\.\d+|\d+', val)
            return float(nums[0]) + 0.01 if nums else np.nan
        nums = re.findall(r'\d+\.\d+|\d+', val)
        if len(nums) == 2:
            return (float(nums[0]) + float(nums[1])) / 2
        elif len(nums) == 1:
            return float(nums[0])
        return np.nan

    def assign_bin(mid, bins):
        if np.isnan(mid):
            return "Missing"
        for low, high, label in bins:
            if low <= mid < high:
                return label
        return "Out of Range"

    return {
        'process_budget': lambda x: assign_bin(parse_value(x), budget_bins),
        'process_family_income': lambda x: assign_bin(parse_value(x), income_bins),
        'process_individual_income': lambda x: assign_bin(parse_value(x), individual_income_bins)
    }

def clean_financial_columns(df):
    """Clean and categorize financial columns"""
    df = df.copy()
    
    # Get the bin assignment functions
    assign_funcs = create_bin_assignment_functions()
    
    # Process each financial column
    if 'purchase_budget' in df.columns:
        df['purchase_budget'] = df['purchase_budget'].apply(assign_funcs['process_budget'])
    
    if 'family_monthly_income' in df.columns:
        df['family_monthly_income'] = df['family_monthly_income'].apply(assign_funcs['process_family_income'])
        
    if 'individual_monthly_income_baht' in df.columns:
        df['individual_monthly_income_baht'] = df['individual_monthly_income_baht'].apply(assign_funcs['process_individual_income'])

    return df

def fill_missing_categoricals(df, ordinal_features=None, nominal_features=None):
    ordinal_features = ordinal_features or []
    nominal_features = nominal_features or []
    for col in ordinal_features:
        if col in df.columns:
            df[col] = df[col].fillna("Missing")

    for col in nominal_features:
        if col in df.columns:
            df[col] = df[col].fillna("unknown")

    return df

def group_rare_categories_by_threshold(df, columns, threshold=0.01):
    df = df.copy()
    for col in columns:
        value_counts = df[col].value_counts(normalize=True)
        rare_vals = value_counts[value_counts < threshold].index
        df[col] = df[col].apply(lambda x: 'Other' if x in rare_vals else x)
    return df

def final_cleanup(df):
    df = df.dropna(subset=['questiondate'])
    df_sorted  = df.sort_values(by='questiondate')
    # Drop house-related and unnecessary columns
    columns_to_drop = ['home_purchase_budget','land_house_size_wanted','functions_wanted', 'moving_in_count',
                       'preferred_discount_categories_AssetWise_Clubs', 'decision_influencer',
                       'current_residence_type', 'desired_living_area', 'monthly_family_income_baht',
                       'individual_monthly_income_fill', 'preferred_house_style', 'preferred_house_features',
                       'Type',
                        'customerid','questiondate', 'questiontime', 'fillindate', 'bookingdate', 
                       'saledate', 'salesperson satisfication' #drop this for now because it's after sale
                       ]  # Add actual columns if needed
    df_sorted.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    return df_sorted

def export_data(df, output_csv, output_excel):
    df.to_csv(output_csv, index=False)
    df.to_excel(output_excel, index=False)
    print(f"Exported to {output_csv} and {output_excel}")

def run_pipeline(data_path, table_path, output_csv, output_excel):
    df = read_data(data_path, table_path)
    # df = sample_projects(df)
    df = preprocess_dates(df)
    df = df[df['Type'] == 'คอนโดมิเนียม'].copy()
    df = add_seasonal_features(df, 'questiondate')
    df = clean_financial_columns(df)

    """Fill missing values in ordinal and nominal categorical features."""
    # =============================================
    # ORDINAL ENCODING (for features with natural order)
    # =============================================
    ordinal_features = ['decision_time_frame', 'age', 'car_type', 'room_size_wanted',
                        'purchase_budget', 'residences_count', 'would_recommend', 'family_monthly_income',
                        'individual_monthly_income_baht', 'Project Type'
                        ]  # Add your ordinal feature names here
    # =============================================
    # NOMINAL ENCODING (for features without order)
    # =============================================
    nominal_features = ['gender', 'occupation', 'marital_status', 'information_source',
                    'purchasing_reason', 'decide_purchase_reason', 'not_book_reason',
                    'other_projects_before_deicde', 'condo_payment', 'day_off_activity',
                    'most_interested_activites_participation', 'saw_sign', 'exercise_preference',
                    'condo_living_style', 'car_brand', 'purchase_intent', 'travel_route_today', 
                    'Project Brand', 'Location']

    high_cardinality_cols = [
        'information_source', 'purchasing_reason', 'decide_purchase_reason',
        'not_book_reason', 'other_projects_before_deicde', 'day_off_activity',
        'saw_sign', 'car_brand', 'travel_route_today', 'Location'
    ]
    df = fill_missing_categoricals(df, ordinal_features, nominal_features)
    df = group_rare_categories_by_threshold(df, nominal_features, threshold=0.01)
    df = final_cleanup(df)

    no_y = df.drop(columns=['has_booked'], errors='ignore')
    export_data(no_y, output_csv, output_excel)

def get_raw_preview(df, n=5):
    if len(df) <= 2 * n:
        return df.copy()
    return pd.concat([df.head(n), df.tail(n)])

def preprocess_data(filepath, company_data_path=None, save_dir=None):
    """
    Preprocess uploaded file (all transformations except encoding).
    - filepath: path to uploaded file (in uploads/)
    - company_data_path: path to company data (if None, use default)
    - save_dir: where to save processed file (if None, use backend/preprocessed_unencoded/)
    Returns: processed DataFrame (unencoded)
    """
    # Set defaults
    project_root = Path(__file__).resolve().parent.parent
    backend_dir = project_root / 'backend'
    if company_data_path is None:
        company_data_path = backend_dir / 'notebooks' / 'project_info' / 'ProjectID_Detail.xlsx'
    if save_dir is None:
        save_dir = backend_dir / 'preprocessed_unencoded'
    os.makedirs(save_dir, exist_ok=True)

    # Read company data
    company_df = pd.read_excel(company_data_path, engine='openpyxl')
    company_df.rename(columns={'Project ID': 'projectid'}, inplace=True)

    # Read input file
    ext = str(filepath).split('.')[-1].lower()
    if ext == 'csv':
        df = pd.read_csv(filepath)
    elif ext in ('xlsx', 'xls'):
        df = pd.read_excel(filepath, engine='openpyxl' if ext == 'xlsx' else 'xlrd')
    else:
        raise ValueError('Unsupported file type')

    # Validate projectid
    if 'projectid' not in df.columns:
        raise ValueError('File must contain a "projectid" column.')
    if not df['projectid'].isin(company_df['projectid']).all():
        raise ValueError('Some projectid values are not found in company data.')

    # Merge with company data
    df = pd.merge(df, company_df, on='projectid', how='left')

    # Save original column order
    orig_cols = list(df.columns)

    # Preprocessing steps (no encoding)
    df = preprocess_dates(df)
    #condo only filtering
    df = df[df['Type'] == 'คอนโดมิเนียม'].copy()
    df = add_seasonal_features(df, 'questiondate')
    df = clean_financial_columns(df)
    # Add more features as needed, but skip encoding

    # Save processed file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = Path(filepath).stem
    out_path = Path(save_dir) / f'{fname}_preprocessed_{timestamp}.csv'
    df.to_csv(out_path, index=False)

    # Reorder columns: original + new features at end
    new_cols = [c for c in df.columns if c not in orig_cols]
    df = df[orig_cols + new_cols]
    return df

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess condo data from Excel.")
    parser.add_argument('--data', required=True, help="Path to main Excel file")
    parser.add_argument('--table', required=True, help="Path to project ID table")
    parser.add_argument('--out_csv', default="sample_predict.csv", help="Output CSV file path")
    parser.add_argument('--out_excel', default="sample_predict.xlsx", help="Output Excel file path")

    args = parser.parse_args()

    run_pipeline(args.data, args.table, args.out_csv, args.out_excel)
