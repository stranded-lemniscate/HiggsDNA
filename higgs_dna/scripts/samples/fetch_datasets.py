#!/usr/bin/env python
import argparse
import json
import sys
from typing import List, Iterable, Dict
from pathlib import Path
import subprocess
from higgs_dna.utils.logger_utils import setup_logger
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define xrootd prefixes for different regions
xrootd_pfx = {
    "Americas": "root://cmsxrootd.fnal.gov/",
    "Eurasia": "root://xrootd-cms.infn.it/",
    "Yolo": "root://cms-xrd-global.cern.ch/",
}


def get_fetcher_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a JSON file mapping dataset names to file paths."
    )

    parser.add_argument(
        "-i",
        "--input",
        help="Input dataset definition file to process.",
        required=True,
    )
    parser.add_argument(
        "-w",
        "--where",
        help="Specify the region for xrootd prefix (only for grid mode).",
        default="Eurasia",
        choices=["Americas", "Eurasia", "Yolo"],
    )
    parser.add_argument(
        "-x",
        "--xrootd",
        help="Override xrootd prefix with the one given.",
        default=None,
    )
    parser.add_argument(
        "--dbs-instance",
        dest="instance",
        help="The DBS instance to use for querying datasets (only for grid mode).",
        type=str,
        default="prod/global",
        choices=["prod/global", "prod/phys01", "prod/phys02", "prod/phys03"],
    )
    parser.add_argument(
        "--mode",
        help="Mode of operation: 'grid' to fetch remote datasets or 'local' to fetch local file paths.",
        choices=["grid", "local"],
        default="grid",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively include files in subdirectories (only for local mode).",
    )
    parser.add_argument(
        "--file-extension",
        nargs='*',
        help="Filter files by extensions (e.g., .root .txt) (only for local mode). If not specified, all files are included.",
    )

    return parser.parse_args()


def get_dataset_dict_grid(fset: Iterable[Iterable[str]], xrd: str, dbs_instance: str, logger) -> Dict[str, List[str]]:
    """
    Fetch file lists for grid datasets using dasgoclient.
    This function is parallelised and will restart stuck requests after 10 seconds.

    :param fset: Iterable of tuples (dataset-short-name, dataset-path)
    :param xrd: xrootd prefix
    :param dbs_instance: DBS instance for dasgoclient
    :param logger: Logger instance
    :return: Dictionary mapping dataset names to list of file paths
    """
    def fetch_dataset(name: str, dataset: str) -> (str, List[str]):
        logger.info(f"Fetching files for dataset '{name}': '{dataset}'")
        private_appendix = "" if not dataset.endswith("/USER") else " instance=prod/phys03"
        cmd = f"/cvmfs/cms.cern.ch/common/dasgoclient -query='instance={dbs_instance} file dataset={dataset}{private_appendix}'"
        logger.debug(f"Executing command: {cmd}")
        while True:
            try:
                flist = subprocess.check_output(cmd, shell=True, universal_newlines=True, timeout=10).splitlines()
                break
            except subprocess.TimeoutExpired:
                logger.warning(f"Timeout reached for dataset '{dataset}', retrying...")
            except subprocess.CalledProcessError as e:
                logger.error(f"dasgoclient command failed for dataset '{dataset}': {e}")
                return name, []
            except Exception as e:
                logger.error(f"Unexpected error while fetching files for dataset '{dataset}': {e}")
                return name, []
        flist = [xrd + f for f in flist if f.strip()]
        logger.info(f"Found {len(flist)} files for dataset '{name}'.")
        return name, flist


    fdict = {}
    with ThreadPoolExecutor() as executor:
        future_to_dataset = {executor.submit(fetch_dataset, name, dataset): name for name, dataset in fset}
        for future in as_completed(future_to_dataset):
            name, flist = future.result()
            if name not in fdict:
                fdict[name] = flist
            else:
                fdict[name].extend(flist)

    # Reorder fdict to match the order in which datasets were passed in fset
    ordered_fdict = {}
    for name, _ in fset:
        ordered_fdict[name] = fdict.get(name, [])
    return ordered_fdict


def get_dataset_dict_local(
    fset: Iterable[Iterable[str]], recursive: bool, extensions: List[str], logger
) -> Dict[str, List[str]]:
    """
    Collect file lists for local directories.

    :param fset: Iterable of tuples (dataset-short-name, directory-path)
    :param recursive: Whether to search directories recursively
    :param extensions: List of file extensions to filter (case-insensitive)
    :param logger: Logger instance
    :return: Dictionary mapping dataset names to list of local file paths
    """
    fdict = {}

    for name, dir_path in fset:
        logger.info(f"Collecting files for local dataset '{name}': '{dir_path}'")
        directory = Path(dir_path)
        if not directory.is_dir():
            logger.error(f"Directory '{dir_path}' does not exist or is not a directory.")
            continue

        # Choose the appropriate glob method
        pattern = '**/*' if recursive else '*'
        try:
            files = []
            for file in directory.glob(pattern):
                if file.is_file():
                    if extensions:
                        if file.suffix.lower() in [ext.lower() for ext in extensions]:
                            files.append(str(file.resolve()))
                    else:
                        files.append(str(file.resolve()))
            if name not in fdict:
                fdict[name] = files
            else:
                fdict[name].extend(files)
            logger.info(f"Found {len(files)} files for local dataset '{name}'.")
        except Exception as e:
            logger.error(f"Error while collecting files from directory '{dir_path}': {e}")

    return fdict


def read_input_file(input_txt: str, mode: str, logger) -> List[tuple]:
    """
    Read the input text file and parse dataset names and paths.

    :param input_txt: Path to the input text file
    :param mode: Mode of operation ('grid' or 'local') for validation
    :param logger: Logger instance
    :return: List of tuples (dataset-name, dataset-path)
    """
    fset = []
    with open(input_txt, 'r') as fp:
        for i, line in enumerate(fp, start=1):
            stripped_line = line.strip()
            if not stripped_line or stripped_line.startswith("#"):
                continue  # Skip empty lines and comments
            parts = stripped_line.split(None, 1)  # Split by whitespace into at most 2 parts
            if len(parts) != 2:
                logger.warning(f"Line {i} in '{input_txt}' is malformed: '{line.strip()}'")
                continue
            name, path = parts
            if mode == "local":
                # Optionally, you can add more validation for local paths here
                pass
            elif mode == "grid":
                # Optionally, validate grid dataset paths
                pass
            fset.append((name, path))
    return fset


def main():
    args = get_fetcher_args()

    logger = setup_logger(level="INFO")

    if not args.input.endswith(".txt"):
        logger.error("Input file must have a '.txt' extension and be a text file!")
        sys.exit(1)

    # Read and parse the input file
    fset = read_input_file(args.input, args.mode, logger)
    if not fset:
        logger.error(f"No valid entries found in '{args.input}'. Exiting.")
        sys.exit(1)

    logger.info(f"Using the following dataset names and paths: {fset}")

    if args.mode == "grid":
        # Determine xrootd prefix
        xrd = xrootd_pfx.get(args.where, "")
        if args.xrootd:
            xrd = args.xrootd
        logger.info(f"Using xrootd prefix: '{xrd}'")

        # Fetch grid file paths
        fdict = get_dataset_dict_grid(fset, xrd, args.instance, logger)

    elif args.mode == "local":
        # Fetch local file paths
        fdict = get_dataset_dict_local(fset, args.recursive, args.file_extension, logger)

    # Check if any data was collected
    if not fdict:
        logger.error("No files were collected. Exiting without creating JSON.")
        sys.exit(1)

    # Define output JSON file path
    output_json = Path(args.input).with_suffix('.json')

    # Write the JSON data to the output file
    try:
        with open(output_json, 'w') as fp:
            json.dump(fdict, fp, indent=4)
        logger.info(f"Successfully wrote data to JSON file '{output_json}'.")
    except Exception as e:
        logger.error(f"Error writing to JSON file '{output_json}': {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
