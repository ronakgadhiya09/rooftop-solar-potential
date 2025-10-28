#!/usr/bin/env python3
"""
Batch convert all JSON masks from json_masks folder to tilted_masks folder.
"""

import os
from pathlib import Path
import convert_json_to_npy

def batch_convert_json_to_tilted():
    """Convert all JSON files in json_masks to tilted_masks directory."""
    
    # Paths
    base_dir = Path("india_dataset/twoChannels_in/train")
    json_dir = base_dir / "json_masks"
    output_dir = base_dir / "tilted_masks"
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Get all JSON files
    json_files = sorted(list(json_dir.glob("*.json")))
    
    print(f"Found {len(json_files)} JSON files in {json_dir}")
    print(f"Output directory: {output_dir}\n")
    
    # Convert each file
    success_count = 0
    error_count = 0
    
    for json_file in json_files:
        output_file = output_dir / f"{json_file.stem}.npy"
        
        try:
            print(f"Converting: {json_file.name} -> {output_file.name}")
            convert_json_to_npy.convert_json_to_npy(
                str(json_file),
                str(output_file),
                image_size=None  # Auto-detect from images
            )
            success_count += 1
        except Exception as e:
            print(f"  ERROR: {str(e)}\n")
            error_count += 1
    
    print(f"\n{'='*50}")
    print(f"Conversion Complete!")
    print(f"  Successfully converted: {success_count}")
    print(f"  Errors: {error_count}")
    print(f"  Total files: {len(json_files)}")
    print(f"  Output location: {output_dir}")

if __name__ == '__main__':
    batch_convert_json_to_tilted()

