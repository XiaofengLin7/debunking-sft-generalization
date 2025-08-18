"""
Configuration file for face card mappings in the gp-l dataset.
Add new mappings here to extend the training data diversity.
"""

# Face card mapping configurations
# Each mapping defines how J, Q, K cards are converted to numeric values
FACE_CARD_MAPPINGS = {
    # All face cards map to the same value
    "all_10": {"J": 10, "Q": 10, "K": 10},
    "all_11": {"J": 11, "Q": 11, "K": 11},
    "all_9": {"J": 9, "Q": 9, "K": 9},
    "all_8": {"J": 8, "Q": 8, "K": 8},
    "all_12": {"J": 12, "Q": 12, "K": 12},
    "all_13": {"J": 13, "Q": 13, "K": 13},
    
    # Mixed mappings with different values
    "mixed_9_10_11": {"J": 9, "Q": 10, "K": 11},
    "mixed_10_11_12": {"J": 10, "Q": 11, "K": 12},
    "mixed_8_9_10": {"J": 8, "Q": 9, "K": 10},
    "mixed_11_12_13": {"J": 11, "Q": 12, "K": 13},
    "mixed_9_11_13": {"J": 9, "Q": 11, "K": 13},
    "mixed_7_9_11": {"J": 7, "Q": 9, "K": 11},
    "mixed_8_10_12": {"J": 8, "Q": 10, "K": 12},
    
    # Sequential mappings
    "sequential_7_8_9": {"J": 7, "Q": 8, "K": 9},
    
    # Skip mappings (every other number)
    "skip_7_9_11": {"J": 7, "Q": 9, "K": 11},
    "skip_8_10_12": {"J": 8, "Q": 10, "K": 12},
    "skip_9_11_13": {"J": 9, "Q": 11, "K": 13},
    
    # Custom mappings for specific training scenarios
    "low_range_5_7_9": {"J": 5, "Q": 7, "K": 9},
    "mid_range_8_10_12": {"J": 8, "Q": 10, "K": 12},
}

# Training configuration presets
TRAINING_PRESETS = {
    "basic": [
        "all_10",           # Standard mapping
    ],
    
    "diverse": [
        "all_10", "all_11", "all_13",
        "mixed_9_10_11", "mixed_10_11_12", 
        "skip_7_9_11", "skip_8_10_12", "skip_9_11_13",

    ],
    
    "extensive": [
        "all_10", "all_11", "all_9", "all_8", "all_12",
        "mixed_9_10_11", "mixed_10_11_12", "mixed_8_9_10",
        "mixed_11_12_13", "mixed_9_11_13", "mixed_7_9_11",
        "sequential_7_8_9", 
        "skip_7_9_11", "skip_8_10_12", "skip_9_11_13",
        "low_range_5_7_9", "mid_range_8_10_12", 
    ],
    
    "custom": [
        # Add your custom selection here
        "all_10", "mixed_9_10_11", "sequential_8_9_10"
    ]
}

def add_custom_mapping(name: str, mapping: dict):
    """
    Add a custom face card mapping.
    
    Args:
        name: Name for the mapping (e.g., "my_custom_mapping")
        mapping: Dictionary with J, Q, K as keys and their numeric values
    
    Example:
        add_custom_mapping("my_mapping", {"J": 6, "Q": 8, "K": 10})
    """
    if not all(k in mapping for k in ["J", "Q", "K"]):
        raise ValueError("Mapping must contain J, Q, and K keys")
    
    FACE_CARD_MAPPINGS[name] = mapping
    print(f"Added custom mapping: {name} -> {mapping}")

def get_mapping_info(mapping_name: str) -> dict:
    """
    Get information about a specific mapping.
    
    Args:
        mapping_name: Name of the mapping to get info for
    
    Returns:
        Dictionary with mapping details
    """
    if mapping_name not in FACE_CARD_MAPPINGS:
        raise ValueError(f"Unknown mapping: {mapping_name}")
    
    mapping = FACE_CARD_MAPPINGS[mapping_name]
    values = list(mapping.values())
    
    info = {
        "name": mapping_name,
        "mapping": mapping,
        "values": values,
        "min_value": min(values),
        "max_value": max(values),
        "unique_values": len(set(values)),
        "is_uniform": len(set(values)) == 1,
        "is_sequential": len(set(values)) == 3 and max(values) - min(values) == 2,
        "is_skip": len(set(values)) == 3 and (max(values) - min(values)) % 2 == 0
    }
    
    return info

def list_all_mappings():
    """Print all available mappings with their details."""
    print("Available Face Card Mappings:")
    print("=" * 50)
    
    for name, mapping in FACE_CARD_MAPPINGS.items():
        info = get_mapping_info(name)
        print(f"{name:20} -> {mapping}")
        if info["is_uniform"]:
            print(f"{'':20}   (Uniform: all face cards = {info['values'][0]})")
        elif info["is_sequential"]:
            print(f"{'':20}   (Sequential: {info['min_value']} -> {info['min_value']+1} -> {info['max_value']})")
        elif info["is_skip"]:
            print(f"{'':20}   (Skip pattern: {info['min_value']} -> {info['min_value']+2} -> {info['max_value']})")
        else:
            print(f"{'':20}   (Mixed: {info['min_value']} -> {info['max_value']})")
        print()

if __name__ == "__main__":
    # Example usage and demonstration
    list_all_mappings()
    
    # Example of adding a custom mapping
    print("Adding custom mapping example...")
    add_custom_mapping("example_6_8_10", {"J": 6, "Q": 8, "K": 10})
    
    # Show the new mapping info
    info = get_mapping_info("example_6_8_10")
    print(f"New mapping info: {info}")
