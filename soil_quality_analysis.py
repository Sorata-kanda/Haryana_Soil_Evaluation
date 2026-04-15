import pandas as pd
import numpy as np
import re

# Load the dataset
df = pd.read_csv('Dataset for Python - Sheet1.csv')
# Clean column names (remove leading/trailing spaces)
df.columns = df.columns.str.strip()
print(f"Dataset loaded with {len(df)} rows and {len(df.columns)} columns")

# Create a copy
df_copy = df.copy()
print(f"Copy created: {df_copy.shape}")

# ============================================
# STEP 1: SEMANTIC PARSING (Extract Meaning)
# ============================================

def parse_range(value):
    """
    Parse string to extract semantic meaning
    Returns: (type, value1, value2)
    Types: 'lt' (less than), 'gt' (greater than), 'range', 'exact'
    """
    if pd.isna(value):
        return None
    
    # Clean the string
    value = str(value).strip().replace(" ", "").lower()
    
    # Extract all numbers (including decimals)
    nums = re.findall(r'\d+\.?\d*', value)
    nums = [float(n) for n in nums] if nums else []
    
    if not nums:
        return None
    
    # Determine type based on symbols
    if '<' in value and '>' not in value:
        # Less than: <280
        return ('lt', nums[0])
    
    elif '>' in value and '<' not in value:
        # Greater than: >500
        return ('gt', nums[0])
    
    elif ('<' in value or '>') and ('to' in value or '−' in value or '-' in value):
        # Range with mixed symbols: <120 to 280 or 10 to >25
        if len(nums) >= 2:
            return ('range', nums[0], nums[1])
        else:
            return ('range', nums[0], nums[0])
    
    elif 'to' in value or '−' in value or '-' in value:
        # Simple range: 280-350 or 10 to 25
        if len(nums) >= 2:
            return ('range', nums[0], nums[1])
        else:
            return ('exact', nums[0])
    
    else:
        # Single exact value: 15.3
        return ('exact', nums[0])

# ============================================
# STEP 2: SCORE BASED ON SEMANTIC MEANING
# ============================================

def score_nitrogen(value):
    """Score Nitrogen based on parsed meaning"""
    parsed = parse_range(value)
    if not parsed:
        return None
    
    type_val = parsed[0]
    
    if type_val == 'lt':
        # <150 or <280
        threshold = parsed[1]
        if threshold <= 150:
            return 1.5  # Low-end Poor
        elif threshold <= 280:
            return 3.5  # Poor
        else:
            return 3.5
    
    elif type_val == 'range':
        low, high = parsed[1], parsed[2]
        midpoint = (low + high) / 2
        
        if midpoint < 150:
            return 1.5
        elif midpoint < 250:
            return 3.5
        elif midpoint < 350:
            return 5.5
        elif midpoint < 500:
            return 7.5
        else:
            return 9.5
    
    elif type_val == 'gt':
        threshold = parsed[1]
        if threshold >= 500:
            return 9.5  # Good
        elif threshold >= 350:
            return 7.5  # Moderate
        else:
            return 5.5
    
    elif type_val == 'exact':
        val = parsed[1]
        if val < 150:
            return 1.5
        elif val < 250:
            return 3.5
        elif val < 350:
            return 5.5
        elif val < 500:
            return 7.5
        else:
            return 9.5
    
    return 3.5  # Default

def score_phosphorus(value):
    """Score Phosphorus based on parsed meaning"""
    parsed = parse_range(value)
    if not parsed:
        return None
    
    type_val = parsed[0]
    
    if type_val == 'lt':
        threshold = parsed[1]
        if threshold <= 10:
            return 1.5
        elif threshold <= 20:
            return 3.5
        else:
            return 5.5
    
    elif type_val == 'range':
        low, high = parsed[1], parsed[2]
        midpoint = (low + high) / 2
        
        if midpoint < 10:
            return 1.5
        elif midpoint < 20:
            return 3.5
        elif midpoint < 30:
            return 5.5
        elif midpoint < 50:
            return 7.5
        else:
            return 9.5
    
    elif type_val == 'gt':
        threshold = parsed[1]
        if threshold >= 50:
            return 9.5
        elif threshold >= 30:
            return 7.5
        elif threshold >= 20:
            return 5.5
        else:
            return 3.5
    
    elif type_val == 'exact':
        val = parsed[1]
        if val < 10:
            return 1.5
        elif val < 20:
            return 3.5
        elif val < 30:
            return 5.5
        elif val < 50:
            return 7.5
        else:
            return 9.5
    
    return 3.5

def score_potassium(value):
    """Score Potassium based on parsed meaning"""
    parsed = parse_range(value)
    if not parsed:
        return None
    
    type_val = parsed[0]
    
    if type_val == 'lt':
        threshold = parsed[1]
        if threshold <= 120:
            return 1.5
        elif threshold <= 200:
            return 3.5
        else:
            return 5.5
    
    elif type_val == 'range':
        low, high = parsed[1], parsed[2]
        midpoint = (low + high) / 2
        
        if midpoint < 120:
            return 1.5
        elif midpoint < 200:
            return 3.5
        elif midpoint < 300:
            return 5.5
        elif midpoint < 500:
            return 7.5
        else:
            return 9.5
    
    elif type_val == 'gt':
        threshold = parsed[1]
        if threshold >= 500:
            return 9.5
        elif threshold >= 300:
            return 7.5
        elif threshold >= 200:
            return 5.5
        else:
            return 3.5
    
    elif type_val == 'exact':
        val = parsed[1]
        if val < 120:
            return 1.5
        elif val < 200:
            return 3.5
        elif val < 300:
            return 5.5
        elif val < 500:
            return 7.5
        else:
            return 9.5
    
    return 5.5

def score_micronutrient(value, thresholds):
    """Generic scorer for micronutrients"""
    parsed = parse_range(value)
    if not parsed:
        return None
    
    type_val = parsed[0]
    t1, t2, t3, t4 = thresholds  # 4 threshold values
    
    if type_val == 'lt':
        threshold = parsed[1]
        if threshold <= t1:
            return 1.5
        elif threshold <= t2:
            return 3.5
        else:
            return 5.5
    
    elif type_val == 'range':
        low, high = parsed[1], parsed[2]
        midpoint = (low + high) / 2
        
        if midpoint < t1:
            return 1.5
        elif midpoint < t2:
            return 3.5
        elif midpoint < t3:
            return 5.5
        elif midpoint < t4:
            return 7.5
        else:
            return 9.5
    
    elif type_val == 'gt':
        threshold = parsed[1]
        if threshold >= t4:
            return 9.5
        elif threshold >= t3:
            return 7.5
        elif threshold >= t2:
            return 5.5
        else:
            return 3.5
    
    elif type_val == 'exact':
        val = parsed[1]
        if val < t1:
            return 1.5
        elif val < t2:
            return 3.5
        elif val < t3:
            return 5.5
        elif val < t4:
            return 7.5
        else:
            return 9.5
    
    return 5.5

# Wrapper functions for each micronutrient
def score_boron(value):
    return score_micronutrient(value, (0.3, 0.5, 1, 2))

def score_iron(value):
    return score_micronutrient(value, (2, 4.5, 7, 15))

def score_zinc(value):
    return score_micronutrient(value, (0.3, 0.6, 1, 2))

def score_copper(value):
    return score_micronutrient(value, (0.1, 0.2, 0.5, 1))

def score_sulphur(value):
    return score_micronutrient(value, (5, 10, 20, 40))

def score_manganese(value):
    return score_micronutrient(value, (1, 2, 5, 10))

def score_organic_carbon(value):
    return score_micronutrient(value, (0.3, 0.5, 0.75, 1))

def score_soil_ph(value):
    """Special scoring for pH (optimal range is 6.5-7.5)"""
    parsed = parse_range(value)
    if not parsed:
        return None
    
    type_val = parsed[0]
    
    if type_val == 'range':
        low, high = parsed[1], parsed[2]
        midpoint = (low + high) / 2
        
        # Optimal: 6.5-7.5
        if 6.5 <= midpoint <= 7.5:
            return 9.5
        # Acceptable: 6-8.5
        elif 6 <= midpoint <= 8.5:
            return 5.5
        # Poor: 5-6 or 8.5-9
        elif (5 <= midpoint < 6) or (8.5 < midpoint <= 9):
            return 3.5
        # Very poor: <5 or >9
        else:
            return 1.5
    
    elif type_val == 'exact':
        val = parsed[1]
        if 6.5 <= val <= 7.5:
            return 9.5
        elif 6 <= val <= 8.5:
            return 5.5
        elif (5 <= val < 6) or (8.5 < val <= 9):
            return 3.5
        else:
            return 1.5
    
    return 5.5

def score_soil_salinity(value):
    """Score salinity (lower is better)"""
    parsed = parse_range(value)
    if not parsed:
        return None
    
    type_val = parsed[0]
    
    if type_val == 'lt':
        threshold = parsed[1]
        if threshold <= 0.5:
            return 9.5
        elif threshold <= 1:
            return 7.5
        else:
            return 5.5
    
    elif type_val == 'range':
        low, high = parsed[1], parsed[2]
        midpoint = (low + high) / 2
        
        if midpoint < 0.5:
            return 9.5
        elif midpoint < 1:
            return 7.5
        elif midpoint < 2:
            return 5.5
        elif midpoint < 4:
            return 3.5
        else:
            return 1.5
    
    elif type_val == 'gt':
        threshold = parsed[1]
        if threshold >= 4:
            return 1.5
        elif threshold >= 2:
            return 3.5
        else:
            return 5.5
    
    elif type_val == 'exact':
        val = parsed[1]
        if val < 0.5:
            return 9.5
        elif val < 1:
            return 7.5
        elif val < 2:
            return 5.5
        elif val < 4:
            return 3.5
        else:
            return 1.5
    
    return 7.5

def get_quality_label(score):
    """Convert score to quality label"""
    if pd.isna(score):
        return None
    if score <= 2:
        return "Low-end Poor"
    elif score <= 4:
        return "Poor"
    elif score <= 6:
        return "Low-End Moderate"
    elif score <= 8:
        return "Moderate"
    else:
        return "Good"

# ============================================
# STEP 3: APPLY SCORING
# ============================================

print("\nApplying semantic parsing and scoring...\n")

scoring_functions = {
    'Nitrogen': score_nitrogen,
    'Phosphorus': score_phosphorus,
    'Potassium': score_potassium,
    'Boron': score_boron,
    'Iron': score_iron,
    'Zinc': score_zinc,
    'Copper': score_copper,
    'Sulphur': score_sulphur,
    'Manganese': score_manganese,
    'Organic Carbon': score_organic_carbon,
    'Soil pH': score_soil_ph,
    'Soil Salinity': score_soil_salinity
}

for mineral, score_func in scoring_functions.items():
    if mineral in df_copy.columns:
        scale_col = f"{mineral}_scale(1-10)"
        quality_col = f"{mineral}_quality"
        
        df_copy[scale_col] = df_copy[mineral].apply(score_func)
        df_copy[quality_col] = df_copy[scale_col].apply(get_quality_label)
        
        print(f"✓ Created {scale_col} and {quality_col}")

print("\n✅ Semantic parsing and scoring complete!")

# ============================================
# STEP 4: DISPLAY RESULTS
# ============================================

print("\n" + "="*80)
print("TESTING PARSE FUNCTION")
print("="*80)
test_values = ['<280 kg/ha', '10 to >25 kg/ha', '<120 to 280 kg/ha', '>500 kg/ha', '15.3', '>0.5 ppm']
for val in test_values:
    parsed = parse_range(val)
    print(f"{val:30s} → {parsed}")

print("\n" + "="*80)
print("SAMPLE RESULTS - Nitrogen (First 10 rows)")
print("="*80)
print(df_copy[['State', 'District', 'Nitrogen', 'Nitrogen_scale(1-10)', 'Nitrogen_quality']].head(10))

print("\n" + "="*80)
print("SAMPLE RESULTS - Phosphorus (First 10 rows)")
print("="*80)
print(df_copy[['State', 'District', 'Phosphorus', 'Phosphorus_scale(1-10)', 'Phosphorus_quality']].head(10))

print("\n" + "="*80)
print("SAMPLE RESULTS - Potassium (First 10 rows)")
print("="*80)
print(df_copy[['State', 'District', 'Potassium', 'Potassium_scale(1-10)', 'Potassium_quality']].head(10))

print("\n" + "="*80)
print("SAMPLE RESULTS - Micronutrients (First 10 rows)")
print("="*80)
print(df_copy[['Boron', 'Boron_scale(1-10)', 'Boron_quality', 
               'Iron', 'Iron_scale(1-10)', 'Iron_quality']].head(10))

print("\n" + "="*80)
print("SAMPLE RESULTS - Soil Properties (First 10 rows)")
print("="*80)
print(df_copy[['Soil pH', 'Soil pH_scale(1-10)', 'Soil pH_quality',
               'Soil Salinity', 'Soil Salinity_scale(1-10)', 'Soil Salinity_quality']].head(10))

print("\n" + "="*80)
print("ALL NEW COLUMNS CREATED")
print("="*80)
new_cols = [col for col in df_copy.columns if '_scale(1-10)' in col or '_quality' in col]
for i, col in enumerate(new_cols, 1):
    print(f"{i:2d}. {col}")

print("\n" + "="*80)
print("QUALITY DISTRIBUTION SUMMARY")
print("="*80)
for mineral in ['Nitrogen', 'Phosphorus', 'Potassium', 'Boron', 'Iron', 'Zinc']:
    quality_col = f"{mineral}_quality"
    if quality_col in df_copy.columns:
        print(f"\n{mineral}:")
        print(df_copy[quality_col].value_counts())

print("\n" + "="*80)
print("OVERALL STATISTICS")
print("="*80)
score_cols = [col for col in df_copy.columns if '_scale(1-10)' in col]
print("\nAverage Scores Across All Minerals:")
print(df_copy[score_cols].mean().round(2))

print("\n" + "="*80)
print("TOP 5 DISTRICTS BY OVERALL SOIL QUALITY")
print("="*80)
district_avg = df_copy.groupby('District')[score_cols].mean()
district_avg['Overall_Avg'] = district_avg.mean(axis=1)
print(district_avg.sort_values('Overall_Avg', ascending=False).head())

# ============================================
# STEP 5: EXPORT TO EXCEL
# ============================================

print("\n" + "="*80)
print("EXPORTING TO EXCEL")
print("="*80)

# Export the complete dataframe with all scores and quality labels
output_filename = 'soil_quality_analysis_results.xlsx'
df_copy.to_excel(output_filename, index=False, engine='openpyxl')
print(f"✅ Complete results exported to: {output_filename}")
print(f"   Total rows: {len(df_copy)}")
print(f"   Total columns: {len(df_copy.columns)}")
print(f"   New columns added: {len([col for col in df_copy.columns if '_scale(1-10)' in col or '_quality' in col])}")

print("\n" + "="*80)
print("EXPORT COMPLETE!")
print("="*80)
