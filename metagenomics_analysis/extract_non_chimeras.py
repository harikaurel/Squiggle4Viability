import argparse
import pysam

def extract_reads_without_pi_tag(bam_path, output_path):
    bamfile = pysam.AlignmentFile(bam_path, "rb")

    with open(output_path, "w") as out_f:
        count = 0
        for read in bamfile:
            if not read.has_tag("pi"):
                out_f.write(read.query_name + "\n")
                count += 1

    print(f"✅ Done. Found {count} reads without 'pi' tag.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract read IDs *without* 'pi' tag from BAM file.")
    parser.add_argument("-b", "--bam", required=True, help="Path to input BAM file")
    parser.add_argument("-o", "--output", required=True, help="Path to output text file for read IDs")

    args = parser.parse_args()
    extract_reads_without_pi_tag(args.bam, args.output)
