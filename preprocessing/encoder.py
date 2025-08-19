import pandas as pd

def encode_features(df: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Encode using mappings
    owner_map = {
        'First Owner': 1,
        'Second Owner': 2,
        'Third Owner': 3,
        'Fourth & Above Owner': 4,
        'Test Drive Car': 5
    }
    df['owner'] = df['owner'].map(owner_map)

    fuel_map = {
        'Diesel': 1,
        'Petrol': 2,
        'LPG': 3,
        'CNG': 4
    }
    df['fuel'] = df['fuel'].map(fuel_map)

    seller_map = {
        'Individual': 1,
        'Dealer': 2,
        'Trustmark Dealer': 3
    }
    df['seller_type'] = df['seller_type'].map(seller_map)

    transmission_map = {
        'Manual': 1,
        'Automatic': 2
    }
    df['transmission'] = df['transmission'].map(transmission_map)

    # Generate consistent mapping for car brands
    unique_brands = sorted(ref_df['name'].unique())
    brand_map = {brand: idx + 1 for idx, brand in enumerate(unique_brands)}
    df['name'] = df['name'].map(brand_map)

    return df
