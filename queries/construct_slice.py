import argparse
import json
import logging
import os
import sys
import threading
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Optional

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
        joern_path: str,
        server_recreation_interval: int = 5,
        max_paths_per_sample: int = 10,
        enhanced_code_output_dir: Optional[str] = None,
    ):
        """
        Initialize a slice constructor for a specific Joern port and dataset slice.

        Args:
            joern_port: Joern server port number.
            dataset_slice: Subset of the dataset to process.
            output_path: Path to the output JSON file for results.
            log_path: Path to the log JSON file for errors and progress.
            joern_path: Path to the joern-cli directory containing the 'joern' binary.
            server_recreation_interval: Number of samples to process before recreating the Joern server.
            max_paths_per_sample: Maximum number of vulnerability paths to process per sample.
            enhanced_code_output_dir: Optional directory to save enhanced code snippets.
        """
        self.port = joern_port
        self.dataset_slice = dataset_slice
        self.output_file = output_path
        self.logs_file = log_path
        self.joern_path = joern_path
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
            self.enhanced_dir = os.path.join(os.path.dirname(output_path), "enhanced_code")
        Path(self.enhanced_dir).mkdir(parents=True, exist_ok=True)

        # The JoernManager will be initialized in process_dataset
        self.joern_manager = None

    def process_dataset(self):
        """
        Process the assigned slice of the dataset.
        """
        self.logger = logging.getLogger("SliceConstructor")
        self.logger.info(f"Starting processing of {len(self.dataset_slice)} samples")

        try:
            # Set up a new event loop for this thread
            import nest_asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            nest_asyncio.apply(loop)

            # Initialize JoernManager with the new event loop
            self.joern_manager = JoernManager(self.port, self.joern_path)

            # Ensure the Joern server is running before processing
            if not self.joern_manager.check_server_health():
                self.logger.info("Joern server not running. Starting it...")
                if not self.joern_manager.recreate_server():
                    self.logger.error("Failed to start Joern server. Exiting.")
                    self._write_error_logs("Failed to start Joern server")
                    return

            for i, sample in enumerate(self.dataset_slice):
                # Joern Server Recreation Logic
                if i > 0 and i % self.recreate_interval == 0:
                    self.logger.info(f"Processed {self.recreate_interval} samples. Recreating Joern server.")
                    is_healthy = self.joern_manager.recreate_server()
                    if not is_healthy:
                        self.logger.error("Joern server unhealthy after recreation. Exiting.")
                        self._write_error_logs("Unhealthy Joern server after recreation")
                        return
                    self.logger.info("Joern server recreated successfully.")

                self._process_sample(sample)

            self.logger.info("Completed processing all assigned samples")

        except Exception as e:
            self.logger.exception(f"Fatal error in process_dataset: {e}")
            self._write_error_logs(f"Fatal error: {str(e)}")
        finally:
            # Clean up the event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.stop()
            loop.close()

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

            # Resolve code and queries from the sample.
            # The output of generate_and_run_queries.py nests 'code' inside
            # 'details' and stores queries under 'llm_queries'; direct-input
            # datasets use top-level 'code' and 'queries'.
            code = sample.get("code") or sample.get("details", {}).get("code", "")
            queries = sample.get("queries") or sample.get("llm_queries", [])

            if not code:
                self.logger.error("No source code found in sample")
                return
            if not queries:
                self.logger.error("No queries found in sample")
                return

            # Load code into Joern
            load_output = self.joern_manager.load_project(file_name, code)
            if "ConsoleException" in load_output:
                self.logger.error(f"Failed to load project: {load_output}")
                raise ValueError(f"Failed to load project: {load_output}")

            # Get the number of data flows
            num_flows = self.joern_manager.get_number_of_flows(queries)
            self.logger.info(f"Number of flows detected: {num_flows}")

            if num_flows == 0:
                self.logger.info("No flows detected, skipping sample")
                return

            # Run the queries and extract paths
            success, paths = self.joern_manager.run_queries(queries, code)
            
            if not success or not paths:
                self.logger.warning("Failed to extract paths or no paths found")
                return
            
            # Limit the number of paths to process
            paths_to_process = paths[:min(num_flows, self.max_paths)]
            self.logger.info(f"Processing {len(paths_to_process)} paths out of {num_flows} detected")
            
            # Process each path to create enhanced code snippets
            processed_results = self._process_paths(sample, paths_to_process)
            
            if processed_results:
                self.logger.info(f"Successfully processed {len(processed_results)} paths")
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
        source_code = sample.get("code") or sample.get("details", {}).get("code", "")
        if not source_code:
            self.logger.error("Missing source code in sample")
            return results
        
        # Create merged paths using the merger functionality
        merged_paths = merge(paths)
        self.logger.info(f"Created {len(merged_paths)} merged paths from {len(paths)} original paths")
        
        # Process each merged path
        for path_idx, merged_path in enumerate(merged_paths):
            try:
                # Analyze code blocks in the source code
                blocks = analyze_c_code(source_code)
                
                # Extract line numbers from the path
                path_line_numbers = [node.get('line_number') for node in merged_path 
                                    if node.get('line_number') is not None]
                
                if not path_line_numbers:
                    self.logger.warning(f"Path {path_idx}: No line numbers found, skipping")
                    continue
                
                # Get context lines that should be included in the enhanced snippet
                context_lines = get_context(path_line_numbers, blocks)
                
                # Create enhanced code file path
                base_name = os.path.basename(sample["file_name"])
                enhanced_file_path = os.path.join(
                    self.enhanced_dir, 
                    f"{base_name}_path{path_idx}_enhanced.c"
                )
                
                # Save the enhanced code to file and get the enhanced code as a string
                enhanced_code = save_context(list(context_lines), source_code, enhanced_file_path)
                
                # Create the result entry
                result = {
                    "dataset": sample.get("dataset", "unknown"),
                    "transformation_idx": sample.get("transformation_idx", 0),
                    "original_file_name": sample.get("original_file_name", ""),
                    "file_name": sample.get("file_name", ""),
                    "queries": sample.get("queries") or sample.get("llm_queries", []),
                    "path_idx": path_idx,
                    "cwe": sample.get("cwe", ""),
                    "label": sample.get("label", ""),
                    "original_code": sample.get("original_code", ""),
                    "code": source_code,
                    "path": merged_path,
                    "context_lines": list(context_lines),
                    "enhanced_code_file": enhanced_file_path,
                    "enhanced_code": enhanced_code
                }
                
                results.append(result)
                self.logger.debug(f"Successfully processed path {path_idx}")
                
            except Exception as e:
                self.logger.exception(f"Error processing path {path_idx}: {e}")
        
        return results

    def _write_processed_sample(self, processed_sample: Dict):
        """Appends a processed sample to the output JSON file."""
        try:
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    self.logger.warning(f"Output file {self.output_file} does not contain a JSON list. Overwriting.")
                    existing_data = []
            except (FileNotFoundError, json.JSONDecodeError):
                existing_data = []

            existing_data.append(processed_sample)

            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=4)

            self.logger.info(f"Appended result to {self.output_file}")

        except Exception as e:
            self.logger.exception(f"Error writing processed sample: {e}")
            self._write_error_logs(f"Error writing processed sample: {str(e)}")

    def _write_error_logs(self, error_message: str):
        """Appends an error log entry to the logs JSON file."""
        try:
            try:
                with open(self.logs_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                if not isinstance(existing_data, list):
                    existing_data = []
            except (FileNotFoundError, json.JSONDecodeError):
                existing_data = []

            existing_data.append({
                "sample": self.current_sample,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "error": error_message
            })

            with open(self.logs_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=4)

            self.logger.info(f"Appended error log to {self.logs_file}")

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
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "-d", "--dataset-path", type=str,
        required=True,
        help="Path to the JSON file containing the dataset"
    )
    
    parser.add_argument(
        "-b", "--base-joern-port", type=int,
        default=16240,
        help="Starting port number for the Joern servers"
    )
    
    parser.add_argument(
        "-n", "--num-workers", type=int,
        default=1,
        help="Number of parallel threads/workers (and Joern instances) to use."
    )
    
    parser.add_argument(
        "-o", "--output-dir", type=str,
        required=True,
        help="Directory to save the processed results"
    )
    
    parser.add_argument(
        "-j", "--joern-path", type=str,
        required=True,
        help="Path to the joern-cli directory containing the 'joern' binary"
    )
    
    parser.add_argument(
        "--server-recreation-interval", type=int,
        default=5,
        help="Number of samples to process before recreating each Joern server"
    )
    
    parser.add_argument(
        "--max-paths-per-sample", type=int,
        default=10,
        help="Maximum number of vulnerability paths to process per sample"
    )
    
    parser.add_argument(
        "--enhanced-code-dir",
        type=str,
        default=None,
        help="Optional directory to save enhanced code snippets"
    )
    
    parser.add_argument(
        "--log-level", type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level"
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
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(threadName)s - %(name)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def main():
    """
    Main function to parse arguments, set up, and run dataset processing.
    """
    args = parse_arguments()
    setup_logging(args.log_level)

    # --- Argument Validation ---
    if not os.path.isfile(args.dataset_path):
        logging.error(f"Dataset file not found: {args.dataset_path}")
        sys.exit(1)
    if args.num_workers <= 0:
        logging.error("--num-workers must be positive.")
        sys.exit(1)

    # --- Setup Directories ---
    results_dir = os.path.join(args.output_dir, "results")
    logs_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    logging.info(f"Outputs will be saved under: {args.output_dir}")
    logging.info(f"Number of workers: {args.num_workers}")

    # --- Load and Slice Dataset ---
    try:
        with open(args.dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        if not isinstance(dataset, list):
            raise TypeError("Dataset file does not contain a valid JSON list.")
        logging.info(f"Loaded dataset with {len(dataset)} samples from {args.dataset_path}")
    except (json.JSONDecodeError, TypeError, FileNotFoundError, OSError) as e:
        logging.error(f"Failed to load or parse dataset file {args.dataset_path}: {e}")
        sys.exit(1)

    if not dataset:
        logging.warning("Dataset is empty. Exiting.")
        sys.exit(0)

    # Divide dataset into slices for each worker
    slice_size = len(dataset) // args.num_workers
    remainder = len(dataset) % args.num_workers
    threads = []
    start_idx = 0

    # --- Create and Start Threads ---
    logging.info("Starting worker threads...")
    for i in range(args.num_workers):
        current_slice_size = slice_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_slice_size

        if start_idx >= len(dataset):
            logging.warning(f"Worker {i+1} has no data assigned, reducing effective worker count.")
            continue

        dataset_slice = dataset[start_idx:end_idx]

        output_file = os.path.join(results_dir, f'thread_{i+1}_results.json')
        logs_file = os.path.join(logs_dir, f'thread_{i+1}_logs.json')
        port = args.base_joern_port + i

        # Enhanced code output directory for this worker
        worker_enhanced_dir = args.enhanced_code_dir if args.enhanced_code_dir else os.path.join(args.output_dir, f'thread_{i+1}_enhanced_code')
        Path(worker_enhanced_dir).mkdir(parents=True, exist_ok=True)

        processor = SliceConstructor(
            joern_port=port,
            dataset_slice=dataset_slice,
            output_path=output_file,
            log_path=logs_file,
            joern_path=args.joern_path,
            server_recreation_interval=args.server_recreation_interval,
            max_paths_per_sample=args.max_paths_per_sample,
            enhanced_code_output_dir=worker_enhanced_dir,
        )

        thread = threading.Thread(target=processor.process_dataset, name=f"Worker-{i+1}")
        thread.start()
        threads.append(thread)
        logging.info(f"Started Worker-{i+1} (Port {port}) processing {len(dataset_slice)} samples. Output: {output_file}, Logs: {logs_file}")

        start_idx = end_idx

    # --- Wait for Threads to Complete ---
    for thread in threads:
        thread.join()

    logging.info("All worker threads have completed processing.")


if __name__ == "__main__":
    main()