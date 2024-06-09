import os
import glob
import subprocess
import argparse

def filter_pod5_files(directory_path, filtered_dir_path, text_file):
    # Get list of all .pod5 files in the given directory and subdirectories
    pod5_files = glob.glob(os.path.join(directory_path, '*.pod5'), recursive=True)

    for file in pod5_files:
        print(file)
        # Get the relative path from the root directory to the current file
        relative_path = os.path.relpath(file, directory_path)
        # Get the absolute path of the output directory and create it if it doesn't exist
        os.makedirs(filtered_dir_path, exist_ok=True)

        # Run the pod5 filter command using subprocess
        output_file_path = os.path.join(filtered_dir_path, os.path.basename(relative_path))
        subprocess.run(['pod5', 'filter', file, '--output', output_file_path, '--ids', text_file, '--missing-ok'])

def main():
    parser = argparse.ArgumentParser(description="Filter POD5 files based on given text file.")
    parser.add_argument("--directory_path", required=True, help="Directory containing input POD5 files")
    parser.add_argument("--filtered_dir_path", required=True, help="Directory to output filtered POD5 files")
    parser.add_argument("--text_file", required=True, help="Text file containing the list of read IDs to filter")

    args = parser.parse_args()

    filter_pod5_files(args.directory_path, args.filtered_dir_path, args.text_file)

if __name__ == "__main__":
    main()
