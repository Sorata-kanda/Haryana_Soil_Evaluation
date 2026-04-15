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
    # IMPORTANT: Check for ranges FIRST before checking simple < or >
    
    if (('to' in value) or ('-' in value) or ('−' in value)) and len(nums) >= 2:
        # Range with validation: <120 to 280, 10 to >25, 280-350, 10 to 25
        low, high = nums[0], nums[1]
        
        # Preserve inequality meaning in ranges
        if '<' in value and '>' not in value:
            return ('range_lt', low, high)  # <120 to 280 means "less than 120 to 280"
        elif '>' in value and '<' not in value:
            return ('range_gt', low, high)  # 10 to >25 means "10 to greater than 25"
        else:
            return ('range', low, high)  # Simple range: 10 to 25
    
    elif '<' in value and '>' not in value:
        # Less than: <280
        return ('lt', nums[0])
    
    elif '>' in value and '<' not in value:
        # Greater than: >500
        return ('gt', nums[0])
    
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
    
    elif type_val in ['range', 'range_lt', 'range_gt']:
        low, high = parsed[1], parsed[2]
        
        # Handle range with inequality bias
        if type_val == 'range_lt':
            # <150 to 250 - bias toward lower (more conservative)
            midpoint = (low + high) / 2 * 0.7
        elif type_val == 'range_gt':
            # 150 to >250 - bias toward higher
            midpoint = (low + high) / 2 * 1.3
        else:
            # Simple range - use midpoint
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
    
    elif type_val in ['range', 'range_lt', 'range_gt']:
        low, high = parsed[1], parsed[2]
        
        # Handle range with inequality bias
        if type_val == 'range_lt':
            midpoint = (low + high) / 2 * 0.7
        elif type_val == 'range_gt':
            midpoint = (low + high) / 2 * 1.3
        else:
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
    
    elif type_val in ['range', 'range_lt', 'range_gt']:
        low, high = parsed[1], parsed[2]
        
        # Handle range with inequality bias
        if type_val == 'range_lt':
            midpoint = (low + high) / 2 * 0.7
        elif type_val == 'range_gt':
            midpoint = (low + high) / 2 * 1.3
        else:
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
    
    elif type_val in ['range', 'range_lt', 'range_gt']:
        low, high = parsed[1], parsed[2]
        
        # Handle range with inequality bias
        if type_val == 'range_lt':
            midpoint = (low + high) / 2 * 0.7
        elif type_val == 'range_gt':
            midpoint = (low + high) / 2 * 1.3
        else:
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
    """
    Score Organic Carbon - CRITICAL soil health parameter
    Organic Carbon is the foundation of soil health, not a micronutrient
    Optimal: >0.75%
    """
    parsed = parse_range(value)
    if not parsed:
        return None
    
    type_val = parsed[0]
    
    # Get value based on type
    if type_val in ['range', 'range_lt', 'range_gt']:
        low, high = parsed[1], parsed[2]
        
        # Handle inequality bias
        if type_val == 'range_lt':
            # <0.5 to 0.75 - bias toward lower (more conservative)
            val = (low + high) / 2 * 0.7
        elif type_val == 'range_gt':
            # 0.5 to >0.75 - bias toward higher
            val = (low + high) / 2 * 1.3
        else:
            # Simple range - use midpoint
            val = (low + high) / 2
    
    elif type_val == 'lt':
        val = parsed[1] * 0.7  # Conservative estimate
    
    elif type_val == 'gt':
        val = parsed[1] * 1.3  # Optimistic estimate
    
    elif type_val == 'exact':
        val = parsed[1]
    
    else:
        return None
    
    # Scoring based on agricultural standards for Organic Carbon
    if val < 0.3:
        return 1.5  # Very poor - soil degraded
    elif val < 0.5:
        return 3.5  # Poor - needs improvement
    elif val < 0.75:
        return 5.5  # Moderate - acceptable
    elif val < 1.0:
        return 7.5  # Good - healthy soil
    else:
        return 9.5  # Excellent - very healthy soil

def score_soil_ph(value):
    """Special scoring for pH (optimal range is 6.5-7.5)"""
    parsed = parse_range(value)
    if not parsed:
        return None
    
    type_val = parsed[0]
    
    if type_val in ['range', 'range_lt', 'range_gt']:
        low, high = parsed[1], parsed[2]
        
        # Handle range with inequality bias
        if type_val == 'range_lt':
            midpoint = (low + high) / 2 * 0.7
        elif type_val == 'range_gt':
            midpoint = (low + high) / 2 * 1.3
        else:
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
    
    elif type_val == 'lt':
        val = parsed[1]
        if val <= 6.5:
            return 3.5  # Acidic soil
        else:
            return 5.5
    
    elif type_val == 'gt':
        val = parsed[1]
        if val >= 8.5:
            return 3.5  # Alkaline soil
        else:
            return 5.5
    
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
    
    elif type_val in ['range', 'range_lt', 'range_gt']:
        low, high = parsed[1], parsed[2]
        
        # Handle range with inequality bias
        if type_val == 'range_lt':
            midpoint = (low + high) / 2 * 0.7
        elif type_val == 'range_gt':
            midpoint = (low + high) / 2 * 1.3
        else:
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
    """Convert score to quality label (simplified to 3 categories)"""
    if pd.isna(score):
        return None
    if score < 4:
        return "Poor"
    elif score < 7:
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
# STEP 3.5: SCORE 2025 DATA (UPDATED COLUMNS)
# ============================================

print("\n" + "="*80)
print("SCORING 2025 DATA (UPDATED COLUMNS)")
print("="*80)

# Mapping of 2025 columns to their scoring functions
updated_scoring_map = {
    'Updated_Nitrogen': ('Nitrogen', score_nitrogen),
    'Phosphorus.1': ('Phosphorus', score_phosphorus),  # Note: weird column name
    'Updated_Potassium': ('Potassium', score_potassium),
    'Updated_Boron': ('Boron', score_boron),
    'Updated_Iron': ('Iron', score_iron),
    'Updated_Zinc': ('Zinc', score_zinc),
    'Updates_Copper': ('Copper', score_copper),
    'Updated_Sulphur': ('Sulphur', score_sulphur),
    'Updated_Manganese': ('Manganese', score_manganese),
    'Updated_Organic Carbon': ('Organic Carbon', score_organic_carbon),
    'Updates_Soil pH': ('Soil pH', score_soil_ph),
    'Updates_Soil Salinity': ('Soil Salinity', score_soil_salinity)
}

for updated_col, (base_name, score_func) in updated_scoring_map.items():
    if updated_col in df_copy.columns:
        scale_col = f"{base_name}_2025_scale(1-10)"
        quality_col = f"{base_name}_2025_quality"
        
        df_copy[scale_col] = df_copy[updated_col].apply(score_func)
        df_copy[quality_col] = df_copy[scale_col].apply(get_quality_label)
        
        print(f"✓ Created {scale_col} and {quality_col}")

print("\n✅ 2025 data scoring complete!")

# ============================================
# STEP 3.6: COMPUTE WEIGHTED SOIL HEALTH FOR 2025
# ============================================

def compute_soil_health(row):
    """
    Compute overall soil health using weighted scoring
    Priority: Organic Carbon > NPK > pH > Micronutrients
    """
    weights = {
        'Nitrogen': 2,
        'Phosphorus': 2,
        'Potassium': 2,
        'Organic Carbon': 3,
        'Soil pH': 2,
        'Boron': 1,
        'Iron': 1,
        'Zinc': 1,
        'Copper': 1,
        'Sulphur': 1,
        'Manganese': 1,
        'Soil Salinity': 1  # Changed from 1.5 to 1 for cleaner weights
    }
    
    total_score = 0
    total_weight = 0
    
    for mineral, weight in weights.items():
        col = f"{mineral}_scale(1-10)"
        if col in row and pd.notna(row[col]):
            total_score += row[col] * weight
            total_weight += weight
    
    if total_weight == 0:
        return None
    
    avg_score = total_score / total_weight
    
    # Critical rules: Organic Carbon and Nitrogen are essential
    # Lowered OC threshold from 5 to 4 (less harsh, more balanced)
    if pd.notna(row.get('Organic Carbon_scale(1-10)')) and row['Organic Carbon_scale(1-10)'] < 4:
        return "Poor"
    
    # Nitrogen threshold also adjusted to 4 for consistency
    if pd.notna(row.get('Nitrogen_scale(1-10)')) and row['Nitrogen_scale(1-10)'] < 4:
        return "Moderate"
    
    # High salinity is a critical issue
    if pd.notna(row.get('Soil Salinity_scale(1-10)')) and row['Soil Salinity_scale(1-10)'] < 4:
        return "Poor"
    
    # Use weighted average for final classification
    if avg_score < 4:
        return "Poor"
    elif avg_score < 7:
        return "Moderate"
    else:
        return "Good"

print("\nComputing weighted Soil Health scores...")
df_copy['Soil_Health'] = df_copy.apply(compute_soil_health, axis=1)
print("✓ Soil_Health column created")

print("\n✅ Weighted soil health computation complete!")

# ============================================
# STEP 3.7: COMPUTE SOIL HEALTH FOR 2025 DATA
# ============================================

def compute_soil_health_2025(row):
    """
    Compute overall soil health for 2025 data using weighted scoring
    """
    weights = {
        'Nitrogen': 2,
        'Phosphorus': 2,
        'Potassium': 2,
        'Organic Carbon': 3,
        'Soil pH': 2,
        'Boron': 1,
        'Iron': 1,
        'Zinc': 1,
        'Copper': 1,
        'Sulphur': 1,
        'Manganese': 1,
        'Soil Salinity': 1  # Changed from 1.5 to 1 for cleaner weights
    }
    
    total_score = 0
    total_weight = 0
    
    for mineral, weight in weights.items():
        col = f"{mineral}_2025_scale(1-10)"
        if col in row and pd.notna(row[col]):
            total_score += row[col] * weight
            total_weight += weight
    
    if total_weight == 0:
        return None
    
    avg_score = total_score / total_weight
    
    # Critical rules (same as 2016-2017 data)
    if pd.notna(row.get('Organic Carbon_2025_scale(1-10)')) and row['Organic Carbon_2025_scale(1-10)'] < 4:
        return "Poor"
    
    if pd.notna(row.get('Nitrogen_2025_scale(1-10)')) and row['Nitrogen_2025_scale(1-10)'] < 4:
        return "Moderate"
    
    if pd.notna(row.get('Soil Salinity_2025_scale(1-10)')) and row['Soil Salinity_2025_scale(1-10)'] < 4:
        return "Poor"
    
    # Use weighted average for final classification
    if avg_score < 4:
        return "Poor"
    elif avg_score < 7:
        return "Moderate"
    else:
        return "Good"

print("\nComputing weighted Soil Health scores for 2025 data...")
df_copy['Soil_Health_2025'] = df_copy.apply(compute_soil_health_2025, axis=1)
print("✓ Soil_Health_2025 column created")

print("\n✅ 2025 soil health computation complete!")

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
new_cols = [col for col in df_copy.columns if '_scale(1-10)' in col or '_quality' in col or col in ['Soil_Health', 'Soil_Health_2025']]
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
print("OVERALL SOIL HEALTH DISTRIBUTION")
print("="*80)

print("\n2016-2017 Data:")
if 'Soil_Health' in df_copy.columns:
    print(df_copy['Soil_Health'].value_counts())
    print(f"\nPercentage Distribution:")
    print(df_copy['Soil_Health'].value_counts(normalize=True).mul(100).round(2))

print("\n2025 Data:")
if 'Soil_Health_2025' in df_copy.columns:
    print(df_copy['Soil_Health_2025'].value_counts())
    print(f"\nPercentage Distribution:")
    print(df_copy['Soil_Health_2025'].value_counts(normalize=True).mul(100).round(2))

print("\n" + "="*80)
print("SOIL HEALTH COMPARISON (2016-2017 vs 2025)")
print("="*80)
if 'Soil_Health' in df_copy.columns and 'Soil_Health_2025' in df_copy.columns:
    comparison = pd.crosstab(df_copy['Soil_Health'], df_copy['Soil_Health_2025'], 
                             rownames=['2016-2017'], colnames=['2025'])
    print(comparison)
    print("\nInterpretation:")
    print("- Rows: Original soil health (2016-2017)")
    print("- Columns: Current soil health (2025)")
    print("- Diagonal: No change")
    print("- Above diagonal: Improvement")
    print("- Below diagonal: Degradation")

print("\n" + "="*80)
print("OVERALL STATISTICS")
print("="*80)
score_cols = [col for col in df_copy.columns if '_scale(1-10)' in col]
score_cols_2016 = [col for col in score_cols if '2025' not in col]
score_cols_2025 = [col for col in score_cols if '2025' in col]

print("\n2016-2017 Average Scores:")
print(df_copy[score_cols_2016].mean().round(2))

print("\n2025 Average Scores:")
print(df_copy[score_cols_2025].mean().round(2))

print("\n" + "="*80)
print("SCORE CHANGES (2025 vs 2016-2017)")
print("="*80)
for col_2016 in score_cols_2016:
    mineral = col_2016.replace('_scale(1-10)', '')
    col_2025 = f"{mineral}_2025_scale(1-10)"
    if col_2025 in df_copy.columns:
        avg_2016 = df_copy[col_2016].mean()
        avg_2025 = df_copy[col_2025].mean()
        change = avg_2025 - avg_2016
        change_pct = (change / avg_2016 * 100) if avg_2016 != 0 else 0
        status = "↑" if change > 0 else "↓" if change < 0 else "="
        print(f"{mineral:20s}: {avg_2016:5.2f} → {avg_2025:5.2f} ({change:+5.2f}, {change_pct:+5.1f}%) {status}")

print("\n" + "="*80)
print("TOP 5 DISTRICTS BY OVERALL SOIL QUALITY")
print("="*80)

print("\n2016-2017 Data:")
district_avg = df_copy.groupby('District')[score_cols_2016].mean()
district_avg['Overall_Avg'] = district_avg.mean(axis=1)
print(district_avg.sort_values('Overall_Avg', ascending=False).head())

print("\n2025 Data:")
district_avg_2025 = df_copy.groupby('District')[score_cols_2025].mean()
district_avg_2025['Overall_Avg_2025'] = district_avg_2025.mean(axis=1)
print(district_avg_2025.sort_values('Overall_Avg_2025', ascending=False).head())

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
print(f"   2016-2017 columns: {len([col for col in df_copy.columns if '_scale(1-10)' in col and '2025' not in col])}")
print(f"   2025 columns: {len([col for col in df_copy.columns if '2025' in col])}")
print(f"   Total new columns: {len([col for col in df_copy.columns if '_scale(1-10)' in col or '_quality' in col])}")

print("\n" + "="*80)
print("EXPORT COMPLETE!")
print("="*80)
