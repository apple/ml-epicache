# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.

import json
import glob
from pathlib import Path

def merge_json_files(input_dir: str, output_file: str):
    """
    Merge multiple JSON files from a directory into a single JSON file as a list.
    
    Args:
        input_dir: Directory containing JSON files
        output_file: Output file path for the merged JSON
    """
    # Get all JSON files in the directory
    json_files = glob.glob(f"{input_dir}/*.json")
    
    # List to store all JSON data
    merged_data = []
    
    # Read each JSON file and append to the list
    for json_file in sorted(json_files):  # Sort to ensure consistent order
        print(f"Processing: {json_file}")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                merged_data.append(data)
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
    
    # Write the merged data to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=2, ensure_ascii=False)
    
    print(f"Successfully merged {len(merged_data)} files into {output_file}")

if __name__ == "__main__":
    # Paths
    input_directory = "InfiniPot-V2/data/realtalk"
    output_file = "merged_realtalk_data.json"
    
    # Merge the files
    merge_json_files(input_directory, output_file) 