#!/bin/bash

# Make directories for output
mkdir -p processing/kraken2_contigs

# Specify Kraken2 DB path
kraken2_db_path="/path/to/kraken2/database"

# Loop over all .fasta files in ./processing/racon
for fasta_file in ./processing/racon/*.fasta; do
    # Extract the base name of the file without the directory and extension
    base_name=$(basename -- "$fasta_file")
    base_name_no_ext="${base_name%.*}"

    # Run Kraken2 on each .fasta file
    if [ -f "$fasta_file" ]; then
        # Kraken2 is a taxonomic classification tool for assigning taxonomic labels to short DNA sequences
        # --db: specify the path to the Kraken2 database
        # --use-names: print scientific names instead of just taxids
        # --report: file to output the report
        # --output: file to output the classification results
        # --memory-mapping: enable memory mapping for database
        # --threads: number of threads to use
        kraken2 --db "${kraken2_db_path}" --use-names --report "./processing/kraken2_contigs/report_${base_name_no_ext}.txt" --output "./processing/kraken2_contigs/output_${base_name_no_ext}.txt" "$fasta_file" --memory-mapping --threads 28
    else
        # Print a message if the file does not exist
        echo "File $fasta_file does not exist. Skipping."
    fi
done

