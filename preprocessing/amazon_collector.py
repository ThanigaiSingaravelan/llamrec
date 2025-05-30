import os
import json
import pandas as pd
import logging
from typing import List, Dict, Any
import argparse
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AmazonDatasetCollector:
    """
    Processes Amazon dataset JSON files and converts them to structured CSV format.

    This class handles the extraction of reviews and metadata from the Amazon datasets,
    focusing on specific domains (Books, Movies_and_TV, etc.) and extracting relevant fields.
    """

    def __init__(self, base_path: str, max_records: int = 4000000):
        """
        Initialize the Amazon dataset collector.

        Args:
            base_path: Path to directory containing Amazon JSON files
            max_records: Maximum number of records to process per file (default: 4M)
        """
        self.base_path = base_path
        self.max_records = max_records
        self.categories = {
            'Books': {
                'reviews': os.path.join(base_path, 'Books.json'),
                'meta': os.path.join(base_path, 'meta_Books.json')
            },
            'Movies_and_TV': {
                'reviews': os.path.join(base_path, 'Movies_and_TV.json'),
                'meta': os.path.join(base_path, 'meta_Movies_and_TV.json')
            },
            'CDs': {
                'reviews': os.path.join(base_path, 'CDs_and_Vinyl.json'),
                'meta': os.path.join(base_path, 'meta_CDs_and_Vinyl.json')
            },
            'Digital_Music': {
                'reviews': os.path.join(base_path, 'Digital_Music.json'),
                'meta': os.path.join(base_path, 'meta_Digital_Music.json')
            }
        }

    def read_json_lines(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Read a JSON Lines file and return a list of records.

        Args:
            file_path: Path to JSON Lines file

        Returns:
            List of parsed JSON objects
        """
        records = []
        line_count = 0

        try:
            # First count lines for progress bar
            with open(file_path, 'r', encoding='utf-8') as f:
                for _ in f:
                    line_count += 1
                    if line_count >= self.max_records:
                        break

            # Process file with progress bar
            with open(file_path, 'r', encoding='utf-8') as file:
                for i, line in tqdm(enumerate(file), total=min(line_count, self.max_records),
                                    desc=f"Processing {os.path.basename(file_path)}"):
                    if i >= self.max_records:
                        break
                    try:
                        records.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to decode line {i} in {file_path}")

        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")

        return records

    def process_reviews(self, category: str) -> pd.DataFrame:
        """
        Process reviews for a specific category.

        Args:
            category: Category name (e.g., 'Books', 'Movies_and_TV')

        Returns:
            DataFrame containing processed reviews
        """
        file_path = self.categories[category]['reviews']
        logger.info(f"Processing reviews for {category} from {file_path}")

        records = self.read_json_lines(file_path)
        if not records:
            logger.warning(f"No review records found for {category}")
            return pd.DataFrame()

        # Create DataFrame and select relevant columns
        df = pd.DataFrame(records)
        df['category'] = category

        # Select and reorder columns
        columns = ['reviewerID', 'asin', 'overall', 'unixReviewTime', 'reviewText', 'summary', 'category']
        available_columns = [col for col in columns if col in df.columns]

        # Add any missing columns with NaN values
        for col in columns:
            if col not in df.columns:
                df[col] = None

        return df[columns]

    def process_metadata(self, category: str) -> pd.DataFrame:
        """
        Process metadata for a specific category.

        Args:
            category: Category name (e.g., 'Books', 'Movies_and_TV')

        Returns:
            DataFrame containing processed metadata
        """
        file_path = self.categories[category]['meta']
        logger.info(f"Processing metadata for {category} from {file_path}")

        records = self.read_json_lines(file_path)
        if not records:
            logger.warning(f"No metadata records found for {category}")
            return pd.DataFrame()

        # Create DataFrame and handle nested structures
        df = pd.DataFrame(records)
        df['category'] = category

        # Process and flatten nested fields
        if 'category' in df.columns and df['category'].dtype == object:
            df['categories_list'] = df['category'].apply(lambda x: x if isinstance(x, list) else [])

        # Extract primary category if available
        if 'categories' in df.columns:
            df['primary_category'] = df['categories'].apply(
                lambda x: x[0][0] if isinstance(x, list) and len(x) > 0 and len(x[0]) > 0 else None
            )

        # Select and reorder columns
        columns = ['asin', 'title', 'price', 'brand', 'category']
        available_columns = [col for col in columns if col in df.columns]

        # Add any missing columns with NaN values
        for col in columns:
            if col not in df.columns:
                df[col] = None

        result_df = df[columns]
        return result_df

    def collect_and_save(self, output_dir: str):
        """
        Collect and save data for all categories.

        Args:
            output_dir: Directory to save processed CSV files
        """
        os.makedirs(output_dir, exist_ok=True)

        for category in self.categories:
            logger.info(f"Processing {category}")

            # Process reviews
            reviews_df = self.process_reviews(category)
            if not reviews_df.empty:
                output_path = os.path.join(output_dir, f"{category}_reviews.csv")
                reviews_df.to_csv(output_path, index=False)
                logger.info(f"Saved {len(reviews_df)} reviews for {category} to {output_path}")

            # Process metadata
            metadata_df = self.process_metadata(category)
            if not metadata_df.empty:
                output_path = os.path.join(output_dir, f"{category}_metadata.csv")
                metadata_df.to_csv(output_path, index=False)
                logger.info(f"Saved {len(metadata_df)} metadata records for {category} to {output_path}")

    def print_dataset_stats(self):
        """
        Print statistics about available datasets.
        """
        logger.info("Available Amazon datasets:")
        for category, paths in self.categories.items():
            reviews_exists = os.path.exists(paths['reviews'])
            meta_exists = os.path.exists(paths['meta'])

            status = []
            if reviews_exists:
                status.append("Reviews ✓")
            else:
                status.append("Reviews ✗")

            if meta_exists:
                status.append("Metadata ✓")
            else:
                status.append("Metadata ✗")

            logger.info(f"{category}: {' | '.join(status)}")


def main():
    """Main function to run the Amazon dataset collector."""
    parser = argparse.ArgumentParser(description="Amazon Dataset Collector")
    parser.add_argument("--base_path", type=str, required=True,
                        help="Path to Amazon JSON files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save processed CSV files")
    parser.add_argument("--max_records", type=int, default=4000000,
                        help="Maximum number of records to process per file")
    parser.add_argument("--stats_only", action="store_true",
                        help="Only print dataset statistics without processing")

    args = parser.parse_args()

    collector = AmazonDatasetCollector(
        base_path=args.base_path,
        max_records=args.max_records
    )

    if args.stats_only:
        collector.print_dataset_stats()
    else:
        collector.collect_and_save(output_dir=args.output_dir)


if __name__ == "__main__":
    main()