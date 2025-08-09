import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import polars as pl
from datasets import Dataset, get_dataset_config_names
from datasets.exceptions import DatasetNotFoundError
from huggingface_hub import login

source_dir: Path = Path("data")


class SimplePartitioner:
    languages = {
        "en": 0.01,
        **{lang: 0.08 for lang in ["zh", "es"]},
        **{lang: 0.06 for lang in ["fr", "de"]},
        **{lang: 0.04 for lang in ["ja", "ru", "ar"]},
        **{lang: 0.03 for lang in ["pt", "it"]},
        **{lang: 0.025 for lang in ["ko", "hi", "nl"]},
        **{lang: 0.02 for lang in ["pl", "tr", "sv", "da", "no"]},
        **{lang: 0.015 for lang in ["fi", "he", "cs", "hu"]},
        **{lang: 0.01 for lang in ["el", "th", "id", "ms", "vi"]},
        **{
            lang: 0.005
            for lang in ["uk", "ca", "sk", "hr", "bg", "lt", "lv", "et", "sl", "mk"]
        },
    }
    hf_username: str = "permutans"
    dataset_name: str = "dummy-lang-subset-dataset-1m-chunks"

    def generate_fake_dataset(
        self, num_chunks: int = 9, rows_per_chunk: int = 1_000_000
    ):
        """Generate deterministic test dataset with exact proportions"""
        total_rows = num_chunks * rows_per_chunk

        # Calculate exact row counts for each language
        lang_counts = {}
        for lang, freq in self.languages.items():
            lang_counts[lang] = int(total_rows * freq)

        # Create deterministic sequence of languages
        all_languages = []
        for lang, count in lang_counts.items():
            all_languages.extend([lang] * count)

        # Fill remaining rows with English if rounding left gaps
        while len(all_languages) < total_rows:
            all_languages.append("en")

        # Shuffle once for distribution across chunks
        np.random.seed(42)  # Fixed seed for reproducibility
        np.random.shuffle(all_languages)

        for chunk_id in range(num_chunks):
            filename = f"chunk{chunk_id}_{chunk_id:05d}-of-{num_chunks:05d}.parquet"
            filepath = source_dir / filename

            if filepath.exists():
                print(f"{filename} exists, skipping")
                continue

            # Get exact slice for this chunk
            start_idx = chunk_id * rows_per_chunk
            end_idx = start_idx + rows_per_chunk
            chunk_languages = all_languages[start_idx:end_idx]

            df = pl.DataFrame(
                {
                    "id": [f"Q{chunk_id}{i:06d}" for i in range(rows_per_chunk)],
                    "text": [
                        f"Sample claim {i} in {lang}"
                        for i, lang in enumerate(chunk_languages)
                    ],
                    "language": chunk_languages,
                }
            )

            df.lazy().sink_parquet(filepath, mkdir=True)
            print(f"Generated {filename}")

    def ds_subset_exists(self, dataset_id: str, subset_name: str) -> bool:
        """Check if dataset subset exists"""
        try:
            configs = get_dataset_config_names(dataset_id)
            return subset_name in configs
        except DatasetNotFoundError:
            return False

    def process_language(self, language: str):
        """Process all source files for one language and upload"""
        repo_id = f"{self.hf_username}/{self.dataset_name}"

        # Skip if already exists
        if self.ds_subset_exists(repo_id, language):
            print(f"Subset {language} already exists, skipping")
            return

        # Process all source files
        all_data = []
        for source_file in source_dir.glob("*.parquet"):
            df = pl.read_parquet(source_file)
            filtered = df.filter(pl.col("language") == language)
            if len(filtered) > 0:
                all_data.append(filtered)
                print(f"  {source_file.name}: {len(filtered)} rows")

        if not all_data:
            print(f"No data found for {language}")
            return

        # Combine and upload
        combined = pl.concat(all_data)
        dataset = Dataset.from_polars(combined)

        print(f"Uploading {language} with {len(combined)} rows...")
        dataset.push_to_hub(repo_id, config_name=language, split="train")
        print(f"✅ Uploaded {language}")

    def process_all_languages(self):
        """Process all languages"""
        for language in self.languages:
            print(f"\nProcessing {language}...")
            self.process_language(language)


def main():
    print("Simple Language Dataset Partitioner")

    partitioner = SimplePartitioner()

    # Generate test data
    print("1. Generating test dataset...")
    partitioner.generate_fake_dataset()

    # Process languages
    print("\n2. Processing languages...")
    partitioner.process_all_languages()

    print("\nDone!")


if __name__ == "__main__":
    main()
