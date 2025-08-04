import numpy as np

np.set_printoptions(threshold=10000, edgeitems=10, linewidth=200)

def inspect_meta_npy(meta_path):
    """
    Inspect the contents of a .npy file containing metadata.
    
    Args:
        meta_path (str): Path to the .npy file.
    """
    meta = np.load(meta_path, allow_pickle=True)
    print(f"Loaded type: {type(meta)}, shape: {getattr(meta, 'shape', 'N/A')}")

    # Try accessing items if it's a list or array of dicts
    if isinstance(meta, np.ndarray) and meta.dtype == object:
        print(f"Array contains {len(meta)} items")
        for i, item in enumerate(meta[:5]):  # Preview first 5
            print(f"\n--- Entry {i} ---")
            if isinstance(item, dict):
                for key in item:
                    print(f"{key}: {item[key]}")
            else:
                print(item)

    elif isinstance(meta, dict):
        print("Top-level dict keys:", list(meta.keys()))
        for key, val in meta.items():
            print(f"\nKey: {key}, Type: {type(val)}, Shape: {getattr(val, 'shape', 'N/A')}")
            print(val)

    else:
        print("Data preview:")
        print(meta)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python inspect_meta_npy.py <path_to_meta.npy>")
        sys.exit(1)
    
    meta_path = sys.argv[1]
    print(f"Inspecting metadata from: {meta_path}")
    inspect_meta_npy(meta_path)
