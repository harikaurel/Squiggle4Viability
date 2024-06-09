import os
import glob
import subprocess
import uuid
import pod5 as p5
from uuid import UUID, uuid4

def get_read_ids(path):
    read_ids = []
    pod5_files = glob.glob(os.path.join(path, '**', '*.pod5'), recursive=True)
    for file_path in pod5_files:
        with p5.Reader(file_path) as reader:
            for read in reader.reads():
                read_ids.append(str(read.read_id))
    return read_ids

def get_long_reads_ids(input_file_path, min_signal_length):
    with p5.Reader(input_file_path) as reader:
        return [str(read.read_id) for read in reader.reads() if len(read.signal) >= min_signal_length]
    
def save_read_ids_to_file(read_ids, file_path):
    with open(file_path, 'w') as file:
        file.write('\n'.join(read_ids))

def filter_directory(directory_path, filtered_dir_path, min_signal_length):
    # Get list of all .pod5 files in the given directory and subdirectories
    # 1500: for nanopore noise and adapter and barcode, 10000 = prediction or the segment size, in total = 11500
    pod5_files = glob.glob(os.path.join(directory_path, '**', '*.pod5'), recursive=True)

    for file in pod5_files:
        print(file)
        # Get the relative path from the root directory to the current file
        relative_path = os.path.relpath(file, directory_path)
        # Get the absolute path of the output directory and create it if it doesn't exist
        os.makedirs(filtered_dir_path, exist_ok=True)
        # Generate the read IDs file name for this file using the folder name as part of the path
        folder_name = os.path.basename(os.path.dirname(relative_path))
        read_ids_file = os.path.join(filtered_dir_path, f'wanted_read_ids.txt')

        # Get the read IDs and save them to the file
        long_reads_ids = get_long_reads_ids(file, min_signal_length)
        save_read_ids_to_file(long_reads_ids, read_ids_file)

        # Run the pod5 filter command using subprocess
        output_file_path = os.path.join(filtered_dir_path, os.path.basename(relative_path))
        subprocess.run(['pod5', 'filter', file, '--output', output_file_path, '--ids', read_ids_file, '--missing-ok'])

def divide_signal_into_chunks(signal, chunk_size, start_index):
    chunks = []
    num_points = len(signal)
    for i in range(start_index, num_points, chunk_size):
        chunk = signal[i: i + chunk_size]
        if len(chunk) == chunk_size:
            chunks.append(chunk)
    return chunks

def process_file(file_path, chunk_size=3000, start_index=1500):
    print(file_path)
    new_reads = []

    # Load the original .pod5 file
    with p5.Reader(file_path) as reader:
        # Read all the reads from the original .pod5 file
        for read in reader.reads():
            # Create a list to store new Read objects with modified signal chunks
            # Divide the signal into chunks
            chunks = divide_signal_into_chunks(read.signal, chunk_size, start_index)

            # Create new Read objects for the chunks and add them to the list
            for i in range(len(chunks)):
                new_uuid = uuid4()
                orig_parts = str(read.read_id).split('-')
                new_parts = str(new_uuid).split('-')
                new_parts[:2] = orig_parts[:2]
                new_read_id = '-'.join(new_parts)

                new_read = p5.Read(
                    read_id=UUID(new_read_id),
                    end_reason=read.end_reason,
                    calibration=read.calibration,
                    pore=read.pore,
                    run_info=read.run_info,
                    signal=chunks[i],
                    read_number = read.read_number,
                    start_sample = read.start_sample,
                    median_before = read.median_before,
                )
                new_reads.append(new_read)

    # Remove the original .pod5 file
    os.remove(file_path)

    # Open a new .pod5 file for writing
    with p5.Writer(file_path) as writer:
        # Add all the new Read objects to the .pod5 file
        for read in new_reads:
            writer.add_read(read)

def process_directory(directory_path, chunk_size=3000, start_index=1500):
    # Get list of all .pod5 files in the given directory and subdirectories
    pod5_files = glob.glob(os.path.join(directory_path, '**', '*.pod5'), recursive=True)

    for file_path in pod5_files:
        process_file(file_path, chunk_size, start_index)