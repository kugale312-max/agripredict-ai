"""
backend/soil_map.py
===================
Dominant Soil Type mapping for Indian States and Union Territories.

ACADEMIC CITATION & APPROXIMATION NOTICE:
-----------------------------------------
This mapping assigns a single dominant soil type (Alluvial, Black, Red, Sandy, Loamy, Clay)
to each state based on macro-level approximations synthesized from published national soil maps:
- ICAR (Indian Council of Agricultural Research) - Soils of India
- NBSS&LUP (National Bureau of Soil Survey and Land Use Planning) - Soil Orders of India
- NCERT Indian Physical Geography (Classification of Soils in India)

IMPORTANT FOR CITATION:
In academic publications, dissertations, project reports, and presentations, this MUST be
explicitly cited as a generalized macro-level approximation. Actual agricultural soil characteristics
exhibit spatial heterogeneity across districts, agro-climatic zones, and individual farm parcels.

This dictionary is intentionally editable so that researchers, agronomists, or project teams
can update or customize dominant classifications as needed.
"""

# Supported soil types in the AgriPredict AI frontend form:
# - Alluvial: Indo-Gangetic and coastal deltaic plains (high fertility, loamy/clay-silt mixture)
# - Black: Deccan plateau basalt/regur soils (high clay content, moisture retentive)
# - Red: Archean crystalline rocks, peninsular plateau (ferruginous, sandy/clayey loam)
# - Sandy: Arid western desert/semi-arid plains (coarse textured, high permeability)
# - Loamy: Sub-Himalayan, montane, and lateritic coastal belts (balanced sand-silt-clay mix)
# - Clay: Heavy textured, low drainage basins and waterlogged estuaries

VALID_SOIL_TYPES = {"Alluvial", "Black", "Red", "Sandy", "Loamy", "Clay"}

STATE_SOIL_MAP = {
    # Northern States & UTs
    "Punjab": "Alluvial",
    "Haryana": "Alluvial",
    "Uttar Pradesh": "Alluvial",
    "Bihar": "Alluvial",
    "Chandigarh": "Alluvial",
    "Himachal Pradesh": "Loamy",
    "Jammu and Kashmir": "Loamy",
    "Uttarakhand": "Loamy",

    # Central & Western States
    "Madhya Pradesh": "Black",
    "Maharashtra": "Black",
    "Gujarat": "Black",
    "Rajasthan": "Sandy",
    "Goa": "Loamy",
    "Dadra and Nagar Haveli": "Black",

    # Eastern States
    "West Bengal": "Alluvial",
    "Odisha": "Red",
    "Jharkhand": "Red",
    "Chhattisgarh": "Red",

    # Southern States
    "Andhra Pradesh": "Red",
    "Telangana": "Red",
    "Karnataka": "Red",
    "Tamil Nadu": "Red",
    "Kerala": "Loamy",
    "Puducherry": "Alluvial",
    "Andaman and Nicobar Islands": "Loamy",

    # North-Eastern States
    "Assam": "Alluvial",
    "Arunachal Pradesh": "Loamy",
    "Manipur": "Red",
    "Meghalaya": "Red",
    "Mizoram": "Red",
    "Nagaland": "Red",
    "Sikkim": "Loamy",
    "Tripura": "Red",
}


def get_soil_type(state_name: str) -> str:
    """
    Retrieve the dominant soil type for a given Indian state or union territory.
    
    Args:
        state_name: Name of the state (whitespace stripped, title-cased or matched).
        
    Returns:
        One of {'Alluvial', 'Black', 'Red', 'Sandy', 'Loamy', 'Clay'}.
        
    Raises:
        KeyError: If state is not present in STATE_SOIL_MAP.
    """
    cleaned_name = state_name.strip()
    if cleaned_name in STATE_SOIL_MAP:
        return STATE_SOIL_MAP[cleaned_name]
    
    # Case-insensitive fallback
    for known_state, soil in STATE_SOIL_MAP.items():
        if known_state.lower() == cleaned_name.lower():
            return soil
            
    raise KeyError(
        f"State '{state_name}' not found in STATE_SOIL_MAP. "
        f"Available states ({len(STATE_SOIL_MAP)}): {list(STATE_SOIL_MAP.keys())}"
    )


if __name__ == "__main__":
    print(f"Total mapped states: {len(STATE_SOIL_MAP)}")
    soil_counts = {}
    for soil in STATE_SOIL_MAP.values():
        soil_counts[soil] = soil_counts.get(soil, 0) + 1
    print("Soil distribution across states/UTs:")
    for soil, count in sorted(soil_counts.items(), key=lambda x: -x[1]):
        print(f"  - {soil}: {count} states")
