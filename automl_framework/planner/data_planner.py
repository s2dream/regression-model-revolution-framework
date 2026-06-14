import os
import glob
import json
import pandas as pd
from typing import List, Dict, Any

class DataPlanner:
    """
    DataPlanner class for Multi-Dataset Batch Experiments.
    Scans a directory for dataset files, gathers column lists, and infers target columns.
    """
    def __init__(self):
        pass

    def scan_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Scans directory_path for .csv, .tsv, .txt, .parquet, and .jsonl files.
        Extracts column names and sizes, and infers target columns for each.
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        if not os.path.isdir(directory_path):
            raise ValueError(f"Path is not a directory: {directory_path}")

        pattern = os.path.join(directory_path, "*")
        all_files = glob.glob(pattern)
        supported_extensions = ('.csv', '.tsv', '.txt', '.parquet', '.jsonl')
        
        dataset_jobs = []
        for filepath in all_files:
            if os.path.isdir(filepath):
                continue
            if not filepath.endswith(supported_extensions):
                continue

            filename = os.path.basename(filepath)
            size_bytes = os.path.getsize(filepath)
            
            # Format human readable size
            if size_bytes < 1024:
                size_str = f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                size_str = f"{size_bytes / 1024:.1f} KB"
            else:
                size_str = f"{size_bytes / (1024 * 1024):.1f} MB"

            # Parse columns safely
            columns = []
            try:
                if filepath.endswith('.csv'):
                    columns = list(pd.read_csv(filepath, nrows=1).columns)
                elif filepath.endswith('.tsv') or filepath.endswith('.txt'):
                    columns = list(pd.read_csv(filepath, sep='\t', nrows=1).columns)
                elif filepath.endswith('.parquet'):
                    columns = list(pd.read_parquet(filepath).columns)
                elif filepath.endswith('.jsonl'):
                    with open(filepath, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                columns = list(json.loads(line).keys())
                                break
            except Exception as e:
                # Log error and skip parsing columns for this file
                columns = []

            inferred_target = self.infer_target_column(columns)

            dataset_jobs.append({
                "file_path": filepath,
                "file_name": filename,
                "file_size": size_str,
                "columns": columns,
                "target_column": inferred_target
            })

        # Sort by file name for consistency
        dataset_jobs.sort(key=lambda x: x["file_name"])
        return dataset_jobs

    def infer_target_column(self, columns: List[str]) -> str:
        """
        Infers the target column from a list of column names.
        Prioritizes exact matches like 'Target_Y', 'target', 'label', 'y',
        then partial substring matches, falling back to the last column.
        """
        if not columns:
            return "Target_Y"

        exact_keywords = ["target_y", "target", "label", "y", "target_column", "class"]
        columns_lower = [col.lower() for col in columns]

        # 1. Look for exact matches
        for kw in exact_keywords:
            if kw in columns_lower:
                idx = columns_lower.index(kw)
                return columns[idx]

        # 2. Look for partial substring matches
        for kw in ["target", "label", "class"]:
            for col in columns:
                if kw in col.lower():
                    return col

        # 3. Fallback to the last column
        return columns[-1]
