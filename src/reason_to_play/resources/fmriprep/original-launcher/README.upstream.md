# fMRIPrep Pipeline for VGDL fMRI Data

This directory contains scripts for preprocessing the VGDL fMRI dataset using fMRIPrep, a robust and standardized preprocessing pipeline.

## Why fMRIPrep?

fMRIPrep provides several advantages over the custom SPM/ccnl-fmri pipeline:

1. **Robustness**: Extensively tested on diverse datasets
2. **Standardization**: BIDS-compliant inputs and outputs
3. **Comprehensive QC**: Detailed visual reports for quality assessment
4. **Modern algorithms**: State-of-the-art registration and correction methods
5. **Reproducibility**: Containerized environment ensures consistent results

## Prerequisites

### 1. Docker Desktop
Install Docker Desktop from: https://www.docker.com/products/docker-desktop

After installation:
- Start Docker Desktop
- Allocate at least 16GB RAM in Docker preferences
- Ensure you have ~100GB free disk space

### 2. FreeSurfer License
Obtain a free license from: https://surfer.nmr.mgh.harvard.edu/registration.html

Save the license file as:
```bash
~/.freesurfer.license
```

### 3. Data Structure
Your BIDS data should be in:
```
~/Downloads/ds004323-download/
├── sub-01/
│   ├── anat/
│   │   └── sub-01_T1w.nii.gz
│   └── func/
│       ├── sub-01_task-gameplay_run-01_bold.nii.gz
│       ├── sub-01_task-gameplay_run-02_bold.nii.gz
│       └── ...
├── sub-02/
└── ...
```

## Running fMRIPrep

### Process a single subject:
```bash
chmod +x run_fmriprep.sh
./run_fmriprep.sh 02
```

### Process multiple subjects:
```bash
./run_fmriprep.sh 01 02 03
```

### Process all valid subjects (excludes 14, 17, 22):
```bash
./run_fmriprep.sh all
```

## Output Structure

fMRIPrep outputs will be saved in:
```
~/Downloads/VGDL_analysis/fmriprep_output/
├── sub-01/
│   ├── anat/
│   │   ├── sub-01_space-MNI152NLin2009cAsym_desc-preproc_T1w.nii.gz
│   │   ├── sub-01_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz
│   │   └── sub-01_desc-brain_probseg.nii.gz
│   └── func/
│       ├── sub-01_task-gameplay_run-01_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz
│       ├── sub-01_task-gameplay_run-01_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz
│       ├── sub-01_task-gameplay_run-01_desc-confounds_timeseries.tsv
│       └── ...
├── sub-01.html  # Quality control report
└── ...
```

## Key Outputs

### 1. Preprocessed BOLD data
- **File**: `*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz`
- **Description**: Motion-corrected, slice-time corrected, registered to MNI space
- **Resolution**: 2mm isotropic (as specified)

### 2. Confound regressors
- **File**: `*_desc-confounds_timeseries.tsv`
- **Includes**:
  - 6 motion parameters (translation/rotation)
  - Motion derivatives
  - Framewise displacement (FD)
  - DVARS
  - CompCor components (aCompCor, tCompCor)
  - ICA-AROMA components (if --use-aroma flag is used)
  - Global signals (CSF, WM, whole-brain)

### 3. Brain masks
- **File**: `*_desc-brain_mask.nii.gz`
- **Description**: Binary brain mask in native or MNI space

### 4. QC Reports
- **File**: `sub-XX.html`
- **Description**: Interactive HTML reports with:
  - Registration quality visualization
  - Motion parameter plots
  - Artifact detection
  - BOLD summary statistics

## Using fMRIPrep Outputs

### Python Example
Use the provided `load_fmriprep_outputs.py` script:

```python
from load_fmriprep_outputs import FMRIPrepLoader

# Load data for subject 2
loader = FMRIPrepLoader(
    fmriprep_dir='~/Downloads/VGDL_analysis/fmriprep_output',
    subject_id='02'
)

# Get preprocessed data
glm_data = loader.prepare_glm_data(confound_strategy='minimal')

# Access the data
bold_images = glm_data['bold_imgs']  # List of NIfTI images
confounds = glm_data['confounds']    # List of confound DataFrames
masks = glm_data['masks']            # List of brain masks
```

### Confound Selection Strategies

The `load_fmriprep_outputs.py` script provides three confound strategies:

1. **minimal**: Basic motion correction
   - 6 motion parameters
   - CSF and white matter signals

2. **comprehensive**: Extensive denoising
   - Motion parameters + derivatives + quadratic terms
   - Tissue signals
   - Global signal

3. **aroma**: ICA-based denoising
   - ICA-AROMA noise components
   - Automated artifact removal

## Processing Time Estimates

Per subject (8 runs):
- **With FreeSurfer**: ~6-8 hours
- **Without FreeSurfer** (--fs-no-reconall): ~2-3 hours

Total dataset (29 subjects):
- **Sequential**: ~5-7 days
- **Parallel** (4 subjects): ~2-3 days

## Troubleshooting

### Docker Issues
```bash
# Check Docker is running
docker info

# Check available resources
docker system df

# Clean up space if needed
docker system prune -a
```

### Memory Issues
If fMRIPrep crashes with memory errors:
1. Increase Docker memory allocation
2. Reduce `--nthreads` parameter
3. Process fewer subjects in parallel

### Missing FreeSurfer License
```bash
# Download license from website, then:
mv ~/Downloads/license.txt ~/.freesurfer.license
```

## Comparison with SPM Pipeline

| Feature | SPM/ccnl-fmri | fMRIPrep |
|---------|---------------|----------|
| Slice timing | Optional | Automatic |
| Motion correction | SPM realign | MCFLIRT/3dvolreg |
| Distortion correction | SPM unwarp | Fieldmap/SyN |
| Registration | SPM normalize | ANTs/FSL |
| Surface analysis | No | FreeSurfer integration |
| QC reports | Manual | Automated HTML |
| Confounds | Basic | Comprehensive |
| BIDS compliance | No | Full |

## Next Steps

After preprocessing:

1. **Review QC reports**: Check `sub-XX.html` for each subject
2. **Select confounds**: Choose appropriate denoising strategy
3. **Run GLM**: Use preprocessed data with your task model
4. **Group analysis**: Perform second-level statistics

## References

- fMRIPrep paper: Esteban et al. (2019) Nature Methods
- fMRIPrep documentation: https://fmriprep.org
- BIDS specification: https://bids.neuroimaging.io