#!/bin/bash
# Run fMRIPrep preprocessing pipeline on VGDL fMRI data
# This replaces the SPM/ccnl-fmri pipeline with a more robust, standardized approach

# Configuration
BIDS_DIR="$HOME/Downloads/ds004323-download"
OUTPUT_DIR="$HOME/Downloads/VGDL_analysis/fmriprep_output"
WORK_DIR="$HOME/Downloads/VGDL_analysis/fmriprep_work"
FREESURFER_LICENSE="$HOME/.freesurfer.license"

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$WORK_DIR"

# Check if FreeSurfer license exists
if [ ! -f "$FREESURFER_LICENSE" ]; then
    echo "ERROR: FreeSurfer license not found at $FREESURFER_LICENSE"
    echo "Please obtain a license from: https://surfer.nmr.mgh.harvard.edu/registration.html"
    echo "Save it as: $FREESURFER_LICENSE"
    exit 1
fi

# Function to run fMRIPrep for a single subject
run_single_subject() {
    SUBJECT=$1
    echo "=========================================="
    echo "Processing subject: ${SUBJECT}"
    echo "=========================================="

    echo "System resources detected:"
    echo "  - Total RAM: 48 GB"
    echo "  - CPU cores: 14"
    echo "  - Docker allocated: $(docker system info 2>/dev/null | grep "Total Memory:" | awk '{print $3}')"
    echo ""
    echo "Processing stages for subject ${SUBJECT}:"
    echo "1. Anatomical preprocessing (brain extraction, tissue segmentation)"
    echo "2. Functional realignment and motion correction"
    echo "3. Registration to anatomical and standard space"
    echo "4. Confound extraction and denoising"
    echo "5. Report generation with visualizations"
    echo "----------------------------------------"
    echo "Starting at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # Run fMRIPrep using Docker (with platform specification for Apple Silicon)
    # Using maximum safe resources based on Docker allocation
    docker run --rm \
        --platform linux/amd64 \
        -v "$BIDS_DIR":/data:ro \
        -v "$OUTPUT_DIR":/out \
        -v "$WORK_DIR":/work \
        -v "$FREESURFER_LICENSE":/license/license.txt:ro \
        nipreps/fmriprep:24.1.0 \
        /data /out participant \
        --participant-label "$SUBJECT" \
        --work-dir /work \
        --clean-workdir \
        --fs-license-file /license/license.txt \
        --output-spaces MNI152NLin2009cAsym:res-2 anat \
        --nthreads 8 \
        --omp-nthreads 4 \
        --mem-mb 24000 \
        --stop-on-first-crash \
        --skip-bids-validation \
        --return-all-components \
        --fd-spike-threshold 0.5 \
        --dvars-spike-threshold 1.5 \
        --skull-strip-template OASIS30ANTs \
        --skull-strip-fixed-seed \
        --no-submm-recon \
        --fs-no-reconall \
        --bold2anat-dof 9 \
        --force-bbr \
        --medial-surface-nan \
        --dummy-scans 0 \
        --random-seed 23 \
        --write-graph \
        --resource-monitor \
        --verbose

    # Check completion status
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ Subject ${SUBJECT} completed successfully!"
        echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
        echo ""
        echo "View visual report: open $OUTPUT_DIR/sub-${SUBJECT}.html"
    else
        echo ""
        echo "✗ Subject ${SUBJECT} failed"
        echo "Check logs in: $WORK_DIR"
    fi
}

# Parse command line arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <subject_number> [subject_number2 ...]"
    echo "Example: $0 02"
    echo "Example: $0 01 02 03"
    echo ""
    echo "To run all subjects (1-32, excluding 14, 17, 22):"
    echo "$0 all"
    exit 1
fi

# Check if Docker is installed and running
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed."
    echo "Please install Docker Desktop from: https://www.docker.com/products/docker-desktop"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "ERROR: Docker is not running."
    echo "Please start Docker Desktop and try again."
    exit 1
fi

# Process subjects
if [ "$1" == "all" ]; then
    # Run all good subjects
    SUBJECTS="01 02 03 04 05 06 07 08 09 10 11 12 13 15 16 18 19 20 21 23 24 25 26 27 28 29 30 31 32"
    for subj in $SUBJECTS; do
        run_single_subject "$subj"
    done
else
    # Run specified subjects
    for subj in "$@"; do
        # Pad single digit subjects with zero
        if [ ${#subj} -eq 1 ]; then
            subj="0$subj"
        fi
        run_single_subject "$subj"
    done
fi

echo ""
echo "=========================================="
echo "fMRIPrep processing complete!"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo ""
echo "Key outputs for each subject:"
echo "  - Preprocessed BOLD: $OUTPUT_DIR/sub-XX/func/*_space-MNI152NLin2009cAsym_*_bold.nii.gz"
echo "  - Confound regressors: $OUTPUT_DIR/sub-XX/func/*_desc-confounds_timeseries.tsv"
echo "  - Brain mask: $OUTPUT_DIR/sub-XX/func/*_space-MNI152NLin2009cAsym_*_mask.nii.gz"
echo "  - Quality reports: $OUTPUT_DIR/sub-XX.html"
