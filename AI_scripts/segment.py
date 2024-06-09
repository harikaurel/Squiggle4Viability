import argparse
import pod5 as p5
import glob
import os
import pod5_process as pp


def main(path_dir, out_dir, chunk_size, start_index):
    os.makedirs(out_dir, exist_ok=True)

    filter_threshold = chunk_size + start_index
    pp.filter_directory(path_dir, out_dir, filter_threshold)
    pp.process_directory(out_dir, chunk_size=chunk_size, start_index=start_index)
    read_ids = pp.get_read_ids(out_dir)
    print(len(read_ids))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process POD5 files")
    parser.add_argument("--path_dir", required=True, help="Directory containing input POD5 files")
    parser.add_argument("--out_dir", required=True, help="Directory to output filtered POD5 files")
    parser.add_argument("--chunk_size", type=int, default=10000, help="Chunk size for processing reads")
    parser.add_argument("--start_index", type=int, default=1500, help="Start index for processing reads")

    args = parser.parse_args()

    main(args.path_dir, args.out_dir, args.chunk_size, args.start_index)
