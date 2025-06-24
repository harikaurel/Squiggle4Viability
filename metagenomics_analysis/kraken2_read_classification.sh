#!/bin/bash

# Define input and output directories
INPUT_DIR="/path/to/your/fastq/files"
OUTPUT_DIR="/path/to/your/output/kraken2_read_classification"

# Define Kraken2 database path
KRAKEN2_DB="/path/to/your/output/kraken2_database"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run Kraken2 classification on each FASTQ file
start_time=$(date +%s)
for FASTQ in "$INPUT_DIR"/*.fastq; do
    BASENAME=$(basename "$FASTQ" .fastq)
    if [ -f "$FASTQ" ]; then
        echo "Classifying: $FASTQ"
        kraken2 --db "$KRAKEN2_DB" \
                --use-names \
                --threads 28 \
                --report "$OUTPUT_DIR/report_${BASENAME}.txt" \
                --output "$OUTPUT_DIR/output_${BASENAME}.txt" \
                "$FASTQ"
    else
        echo "No FASTQ files found at: $FASTQ"
    fi
done
end_time=$(date +%s)
echo "Total classification time: $((end_time - start_time)) seconds"
