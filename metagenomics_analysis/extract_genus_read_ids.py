import os
import argparse
import pandas as pd

def extract_read_ids(input_dir, genus, output_file=None):
    columns = ["Type", "UUID", "Organism", "Value", "Details"]
    matching_ids = []

    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_dir, filename)
            try:
                df = pd.read_csv(file_path, sep="\t", names=columns, header=None)
                filtered_df = df[df["Organism"].str.contains(genus, case=False, na=False)]
                matching_ids.extend(filtered_df["UUID"].tolist())
            except Exception as e:
                print(f"⚠️ Failed to read {file_path}: {e}")

    print(f"✅ Found {len(matching_ids)} read IDs matching genus '{genus}'.")

    if output_file:
        with open(output_file, "w") as f:
            for rid in matching_ids:
                f.write(rid + "\n")
        print(f"📁 Read IDs written to: {output_file}")

    return matching_ids

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract read IDs from Kraken2 outputs matching a given genus.")
    parser.add_argument("-i", "--input_dir", required=True, help="Directory with Kraken2 output files (*.txt)")
    parser.add_argument("-g", "--genus", required=True, help="Target genus to filter for (e.g., 'Chlamydia')")
    parser.add_argument("-o", "--output", help="Optional output file to save the read IDs")
    args = parser.parse_args()

    extract_read_ids(args.input_dir, args.genus, args.output)
