import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from Components.joern_manager import JoernManager
from Components.enhancer import analyze_c_code, get_context, save_context
from Components.slice import merge


class SliceConstructor:
    """
    1. Processes code samples through Joern to extract vulnerability paths.
    2. Enhances extracted paths into readable code snippets.
    3. Stores the results for further analysis.
    """

    def __init__(
        self,
        joern_port: int,
        dataset_slice: List[Dict],
        output_path: str,
        log_path: str,
        server_recreation_interval: int = 5,
        max_paths_per_sample: int = 10,
        enhanced_code_output_dir: Optional[str] = None,
    ):
        """
        Initialize a constructor for a specific port and dataset slice.

        Args:
            joern_port: Joern server port number.
            dataset_slice: Subset of the dataset to process.
            output_path: Path to the output JSON file for results.
            log_path: Path to the log JSON file for errors and progress.
            server_recreation_interval: Number of samples to process before recreating the Joern server.
            max_paths_per_sample: Maximum number of vulnerability paths to process per sample.
            enhanced_code_output_dir: Optional directory to save enhanced code snippets.
        """
        self.port = joern_port
        self.dataset_slice = dataset_slice
        self.output_file = output_path
        self.logs_file = log_path
        self.recreate_interval = server_recreation_interval
        self.max_paths = max_paths_per_sample
        self.current_sample = ""

        # Create output directories
        Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(log_path)).mkdir(parents=True, exist_ok=True)

        # Create enhanced code directory
        if enhanced_code_output_dir:
            self.enhanced_dir = enhanced_code_output_dir
        else:
            self.enhanced_dir = os.path.join(
                os.path.dirname(output_path), "enhanced_code"
            )
        Path(self.enhanced_dir).mkdir(parents=True, exist_ok=True)

        # Configure logger
        self.logger = logging.getLogger("SliceConstructor")
        self.logger.setLevel(logging.INFO)

        # The JoernManager will be initialized in process_dataset

    def process_dataset(self):
        """
        Process the assigned slice of the dataset.
        """
        self.logger.info(f"Starting processing of {len(self.dataset_slice)} samples")

        try:
            # Initialize JoernManager
            self.joern_manager = JoernManager(self.port)

            # Process the dataset in smaller chunks to avoid memory issues
            for slice_start in range(
                0, len(self.dataset_slice), self.recreate_interval
            ):
                slice_end = min(
                    slice_start + self.recreate_interval, len(self.dataset_slice)
                )

                # Recreate the Joern server if this isn't the first iteration
                if slice_start > 0:
                    self.logger.info(
                        f"Recreating Joern server after processing {self.recreate_interval} samples"
                    )
                    is_healthy = self.joern_manager.recreate_server()

                    if not is_healthy:
                        self.logger.error(
                            "Exiting due to unhealthy Joern server"
                        )
                        self._write_error_logs("Unhealthy Joern server")
                        return

                # Check server health for the first batch
                elif not self.joern_manager.check_server_health():
                    self.logger.error("Initial Joern server health check failed")
                    self._write_error_logs("Initial health check failed")
                    return

                # Process each sample in the current slice
                for sample in self.dataset_slice[slice_start:slice_end]:
                    self._process_sample(sample)

            self.logger.info("Completed processing all assigned samples")

        except Exception as e:
            self.logger.exception(f"Fatal error in process_dataset: {e}")
            self._write_error_logs(f"Fatal error: {str(e)}")

    def _process_sample(self, sample: Dict):
        """
        Process a single code sample through the full pipeline.

        Args:
            sample: The code sample to process.
        """
        self.current_sample = sample["file_name"]

        try:
            self.logger.info(f"Processing sample: {sample['file_name']}")

            # Extract filename from path
            file_name = os.path.basename(sample["file_name"])

            # Load code into Joern
            load_output = self.joern_manager.load_project(file_name)
            if "ConsoleException" in load_output:
                self.logger.error(f"Failed to load project: {load_output}")
                raise ValueError(f"Failed to load project: {load_output}")

            # Run the queries and extract paths
            success, paths = self.joern_manager.run_queries(
                sample["llm_queries"], sample['details']["code"]
            )

            if not success or not paths:
                self.logger.warning("Failed to extract paths or no paths found")
                return

            self.logger.info(f"Number of flows detected: {len(paths)}")

            # Limit the number of paths to process
            paths_to_process = paths[: self.max_paths]
            self.logger.info(
                f"Processing {len(paths_to_process)} paths out of {len(paths)} detected"
            )

            # Process each path to create enhanced code snippets
            processed_results = self._process_paths(sample, paths_to_process)

            if processed_results:
                self.logger.info(
                    f"Successfully processed {len(processed_results)} paths"
                )
                for result in processed_results:
                    self._write_processed_sample(result)
            else:
                self.logger.warning("No viable paths were processed")

            # Clean up by deleting the project from Joern
            self.joern_manager.delete_project(file_name)

            self.logger.info(f"Successfully processed sample: {sample['file_name']}")

        except Exception as e:
            self.logger.exception(f"Error processing sample: {e}")
            self._write_error_logs(f"Error processing sample: {str(e)}")

    def _process_paths(self, sample: Dict, paths: List) -> List[Dict]:
        """
        Process extracted paths to create enhanced code snippets.

        Args:
            sample: The original sample data.
            paths: The extracted paths from Joern.

        Returns:
            List of processed results including enhanced code snippets.
        """
        results = []
        source_code = sample.get("code", "")
        if not source_code:
            self.logger.error("Missing source code in sample")
            return results

        # Create merged paths using the merger functionality
        merged_paths = merge(paths)
        self.logger.info(
            f"Created {len(merged_paths)} merged paths from {len(paths)} original paths"
        )

        # Process each merged path
        for path_idx, merged_path in enumerate(merged_paths):
            try:
                # Analyze code blocks in the source code
                blocks = analyze_c_code(source_code)

                # Extract line numbers from the path
                path_line_numbers = [
                    node.get("line_number")
                    for node in merged_path
                    if node.get("line_number") is not None
                ]

                if not path_line_numbers:
                    self.logger.warning(
                        f"Path {path_idx}: No line numbers found, skipping"
                    )
                    continue

                # Get context lines that should be included in the enhanced snippet
                context_lines = get_context(path_line_numbers, blocks)

                # Create enhanced code file path
                base_name = os.path.basename(sample["file_name"])
                enhanced_file_path = os.path.join(
                    self.enhanced_dir, f"{base_name}_path{path_idx}_enhanced.c"
                )

                # Save the enhanced code to file and get the enhanced code as a string
                enhanced_code = save_context(
                    list(context_lines), source_code, enhanced_file_path
                )

                # Create the result entry
                result = {
                    "dataset": sample.get("dataset", "unknown"),
                    "transformation_idx": sample.get("transformation_idx", 0),
                    "original_file_name": sample.get("original_file_name", ""),
                    "file_name": sample.get("file_name", ""),
                    "queries": sample.get("queries", []),
                    "path_idx": path_idx,
                    "cwe": sample.get("cwe", ""),
                    "label": sample.get("label", ""),
                    "original_code": sample.get("original_code", ""),
                    "code": source_code,
                    "path": merged_path,
                    "context_lines": list(context_lines),
                    "enhanced_code_file": enhanced_file_path,
                    "enhanced_code": enhanced_code,
                }

                results.append(result)
                self.logger.debug(f"Successfully processed path {path_idx}")

            except Exception as e:
                self.logger.exception(f"Error processing path {path_idx}: {e}")

        return results

    def _write_processed_sample(self, processed_sample: Dict):
        """
        Write a processed sample to the output file.

        Args:
            processed_sample: The processed sample data.
        """
        try:
            # If file exists, read existing data, otherwise start with an empty list
            try:
                with open(self.output_file, "r") as f:
                    existing_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                existing_data = []

            # Append new sample
            existing_data.append(processed_sample)

            # Write back to file (atomic write)
            temp_file = f"{self.output_file}.tmp"
            with open(temp_file, "w") as f:
                json.dump(existing_data, f, indent=4)

            # Rename for atomic replacement
            os.replace(temp_file, self.output_file)

            self.logger.info(f"Wrote processed sample to {self.output_file}")

        except Exception as e:
            self.logger.exception(f"Error writing processed sample: {e}")
            self._write_error_logs(f"Error writing processed sample: {str(e)}")

    def _write_error_logs(self, error_message: str):
        """
        Write error logs to the logs file.

        Args:
            error_message: Brief description of the error.
        """
        try:
            # If file exists, read existing data, otherwise start with an empty list
            try:
                with open(self.logs_file, "r") as f:
                    existing_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                existing_data = []

            # Append new log entry
            existing_data.append(
                {
                    "sample": self.current_sample,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "error": error_message,
                }
            )

            # Write back to file (atomic write)
            temp_file = f"{self.logs_file}.tmp"
            with open(temp_file, "w") as f:
                json.dump(existing_data, f, indent=4)

            # Rename for atomic replacement
            os.replace(temp_file, self.logs_file)

            self.logger.info(f"Wrote logs to file {self.logs_file}")

        except Exception as e:
            self.logger.error(f"Error writing logs: {e}")


def parse_arguments():
    """
    Parse command-line arguments for the vulnerability analyzer.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Slice Constructor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-d",
        "--dataset-path",
        type=str,
        required=True,
        help="Path to the JSON file containing the dataset",
    )

    parser.add_argument(
        "-p",
        "--joern-port",
        type=int,
        default=16240,
        help="Port number for the Joern server",
    )

    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save the processed results",
    )

    parser.add_argument(
        "--server-recreation-interval",
        type=int,
        default=5,
        help="Number of samples to process before recreating the Joern server",
    )

    parser.add_argument(
        "--max-paths-per-sample",
        type=int,
        default=10,
        help="Maximum number of vulnerability paths to process per sample",
    )

    parser.add_argument(
        "--enhanced-code-dir",
        type=str,
        default=None,
        help="Optional directory to save enhanced code snippets",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )

    return parser.parse_args()


def setup_logging(log_level):
    """
    Set up logging configuration.

    Args:
        log_level: Logging level as string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main():
    """
    Main entry point for the vulnerability analyzer.
    """
    args = parse_arguments()
    setup_logging(args.log_level)

    logger = logging.getLogger("Construct_Slice")
    logger.info("Starting Slice Construction")
    logger.info(f"Configuration: {vars(args)}")

    # Load dataset
    try:
        logger.info(f"Loading dataset from {args.dataset_path}")
        with open(args.dataset_path, "r") as f:
            dataset = json.load(f)
        logger.info(f"Loaded dataset with {len(dataset)} samples")
    except FileNotFoundError:
        logger.error(f"Dataset file not found at {args.dataset_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(f"Could not decode JSON from {args.dataset_path}")
        sys.exit(1)

    # Set up output paths
    results_file = os.path.join(args.output_dir, "results.json")
    logs_file = os.path.join(args.output_dir, "logs.json")
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine enhanced code output directory
    enhanced_code_dir = args.enhanced_code_dir or os.path.join(
        args.output_dir, "enhanced_code"
    )
    os.makedirs(enhanced_code_dir, exist_ok=True)

    # Create and run the analyzer
    analyzer = SliceConstructor(
        joern_port=args.joern_port,
        dataset_slice=dataset,
        output_path=results_file,
        log_path=logs_file,
        server_recreation_interval=args.server_recreation_interval,
        max_paths_per_sample=args.max_paths_per_sample,
        enhanced_code_output_dir=enhanced_code_dir,
    )

    try:
        analyzer.process_dataset()
        logger.info("Slice construction completed successfully")
    except Exception as e:
        logger.exception(f"Fatal error in main: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
