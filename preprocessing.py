# preprocessing.py

from pathlib import Path
import os
import pandas as pd
import numpy as np
import re
from datetime import datetime

# Debug configuration - set to False to disable debug prints
DEBUG_PRINTS = False

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
        # Add has_booked column only if bookingdate exists
        df['has_booked'] = df['bookingdate'].notna().astype(int)
    else:
        # If no bookingdate column, create has_booked as 0 for all rows
        df['has_booked'] = 0
        
    return df

def add_seasonal_features(df, date_column):
    """Add seasonal features from a datetime column"""
    if date_column not in df.columns:
        print(f"Warning: Date column '{date_column}' not found - skipping seasonal features")
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
    
    # Condo only filtering - only apply if 'Type' column exists
    if 'Type' in df.columns:
        df = df[df['Type'] == 'คอนโดมิเนียม'].copy()
        print("Filtered for condominiums")
    else:
        print("Warning: 'Type' column not found - skipping condo filtering")
    
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
    # Get the current working directory and script location
    script_dir = Path(__file__).resolve().parent  # This is /app/ in Docker
    current_dir = Path(os.getcwd())  # This might be /app/backend/ in Docker
    
    if DEBUG_PRINTS:
        print(f"Script directory: {script_dir}")
        print(f"Current working directory: {current_dir}")
        print(f"Processing file: {filepath}")
    
    # Determine backend directory
    if current_dir.name == 'backend':
        backend_dir = current_dir
        project_root = current_dir.parent
    else:
        project_root = script_dir
        backend_dir = project_root / 'backend'
    
    if company_data_path is None:
        # Use the SAME logic as run_pipeline() function - this is the correct approach
        company_data_path = backend_dir / 'notebooks' / 'project_info' / 'ProjectID_Detail.xlsx'
        
        print(f"MANDATORY: Using table_path logic - ProjectID_Detail.xlsx at: {company_data_path}")
        print(f"File exists: {company_data_path.exists()}")
        
        if not company_data_path.exists():
            # Try alternative paths based on the Docker setup
            alternative_paths = [
                Path('/app/backend/notebooks/project_info/ProjectID_Detail.xlsx'),
                Path('/app/notebooks/project_info/ProjectID_Detail.xlsx'),
                backend_dir / 'notebooks' / 'project_info' / 'ProjectID_Detail.xlsx',
                project_root / 'notebooks' / 'project_info' / 'ProjectID_Detail.xlsx',
            ]
            
            for alt_path in alternative_paths:
                if DEBUG_PRINTS:
                    print(f"Trying alternative path: {alt_path} - Exists: {alt_path.exists()}")
                if alt_path.exists():
                    company_data_path = alt_path
                    print(f"SUCCESS: Found ProjectID_Detail.xlsx at alternative location: {company_data_path}")
                    break
            else:
                # List what's actually in the directory for debugging
                project_info_dir = backend_dir / 'notebooks' / 'project_info'
                if project_info_dir.exists():
                    print(f"Contents of {project_info_dir}:")
                    for item in project_info_dir.iterdir():
                        print(f"  - {item.name} ({item.stat().st_size} bytes)")
                else:
                    print(f"Directory {project_info_dir} does not exist")
                
                raise FileNotFoundError(
                    f"CRITICAL: ProjectID_Detail.xlsx is REQUIRED but not found. "
                    f"Expected at: {backend_dir / 'notebooks' / 'project_info' / 'ProjectID_Detail.xlsx'}"
                )
    
    if save_dir is None:
        save_dir = backend_dir / 'preprocessed_unencoded'
    os.makedirs(save_dir, exist_ok=True)

    # NOW USE THE SAME LOGIC AS run_pipeline() - Call read_data function
    try:
        print(f"Using read_data() function with data_path={filepath} and table_path={company_data_path}")
        
        # Use the read_data function which properly handles the merge
        df = read_data(str(filepath), str(company_data_path))
        
        print(f"read_data() successful - merged data shape: {df.shape}")
        print(f"read_data() columns: {list(df.columns)}")
        
    except Exception as e:
        print(f"ERROR: read_data() failed: {e}")
        print("Falling back to manual file reading and merging...")
        
        # Fallback: Manual reading and merging
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
        
        # Read company data manually
        if company_data_path and Path(company_data_path).exists():
            try:
                print(f"Loading ProjectID_Detail.xlsx from: {company_data_path}")
                company_df = pd.read_excel(company_data_path, engine='openpyxl')
                
                print(f"Loaded company data shape: {company_df.shape}")
                print(f"Company data columns: {list(company_df.columns)}")
                
                # Use the same renaming logic as read_data()
                company_df.rename(columns={'Project ID': 'projectid'}, inplace=True)
                
                print(f"Available project IDs in company data: {sorted(company_df['projectid'].unique())}")
                
                # Perform merge
                initial_shape = df.shape
                df = pd.merge(df, company_df, on='projectid', how='left')
                
                print(f"MERGE RESULTS:")
                print(f"  - Shape changed from {initial_shape} to {df.shape}")
                
                # Check for unmatched project IDs
                input_projects = set(df['projectid'].unique())
                company_projects = set(company_df['projectid'].unique())
                unmatched_projects = input_projects - company_projects
                matched_projects = input_projects & company_projects
                
                print(f"  - Matched projects: {len(matched_projects)} out of {len(input_projects)}")
                if unmatched_projects:
                    print(f"  - WARNING: Unmatched project IDs: {sorted(unmatched_projects)}")
                
            except Exception as e:
                error_msg = f"CRITICAL ERROR: Failed to load ProjectID_Detail.xlsx from {company_data_path}: {e}"
                print(error_msg)
                raise RuntimeError(error_msg)
        else:
            raise FileNotFoundError(f"ProjectID_Detail.xlsx file is required but not accessible at: {company_data_path}")

    print(f"Input data shape after merge: {df.shape}")
    print(f"Input project IDs: {sorted(df['projectid'].unique())}")

    # Save original column order
    orig_cols = list(df.columns)

    # NOW FOLLOW THE SAME PREPROCESSING STEPS AS run_pipeline()
    try:
        # Step 1: Preprocess dates
        df = preprocess_dates(df)
        
        # Step 2: Condo only filtering - only apply if 'Type' column exists
        if 'Type' in df.columns:
            print("Filtering for condominiums (Type == 'คอนโดมิเนียม')")
            initial_count = len(df)
            df = df[df['Type'] == 'คอนโดมิเนียม'].copy()
            filtered_count = len(df)
            print(f"Filtered from {initial_count} to {filtered_count} rows")
        else:
            print("Warning: 'Type' column not found - skipping condo filtering")
        
        # Step 3: Add seasonal features
        df = add_seasonal_features(df, 'questiondate')
        
        # Step 4: Clean financial columns
        df = clean_financial_columns(df)
        
        # Step 5: Fill missing values - USE THE SAME FEATURE LISTS AS run_pipeline()
        ordinal_features = ['decision_time_frame', 'age', 'car_type', 'room_size_wanted',
                            'purchase_budget', 'residences_count', 'would_recommend', 'family_monthly_income',
                            'individual_monthly_income_baht', 'Project Type'
                            ]
        
        nominal_features = ['gender', 'occupation', 'marital_status', 'information_source',
                        'purchasing_reason', 'decide_purchase_reason', 'not_book_reason',
                        'other_projects_before_deicde', 'condo_payment', 'day_off_activity',
                        'most_interested_activites_participation', 'saw_sign', 'exercise_preference',
                        'condo_living_style', 'car_brand', 'purchase_intent', 'travel_route_today', 
                        'Project Brand', 'Location']

        df = fill_missing_categoricals(df, ordinal_features, nominal_features)
        
        # Step 6: Group rare categories
        df = group_rare_categories_by_threshold(df, nominal_features, threshold=0.01)
        
        # Step 7: Final cleanup - BUT SKIP DROPPING COLUMNS since we need them for display
        # df = final_cleanup(df)  # Skip this to preserve all columns
        
        # Instead, just drop rows with missing questiondate and sort
        df = df.dropna(subset=['questiondate'])
        df = df.sort_values(by='questiondate')
        
        print(f"After preprocessing - data shape: {df.shape}")
        
    except Exception as e:
        print(f"Error during preprocessing steps: {e}")
        print(f"Available columns: {list(df.columns)}")
        raise

    # Save processed file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = Path(filepath).stem
    out_path = Path(save_dir) / f'{fname}_preprocessed_{timestamp}.csv'
    df.to_csv(out_path, index=False)
    print(f"Processed file saved to: {out_path}")

    # Reorder columns: original + new features at end
    new_cols = [c for c in df.columns if c not in orig_cols]
    df = df[orig_cols + new_cols]
    
    print(f"Final preprocessed data shape: {df.shape}")
    print(f"Final columns: {list(df.columns)}")
    
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
