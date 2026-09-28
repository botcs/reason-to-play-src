"""ROI support must stay fixed when a result cohort is filtered."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from reason_to_play.analysis.neural import roi


@pytest.fixture
def fixture_data(tmp_path):
    shape = (4, 1, 1)
    affine = np.eye(4)
    masks = {f"sub-{number:02d}": np.ones(shape, dtype=bool) for number in range(1, 33)}
    masks["sub-01"] = np.array([1, 0, 0, 0], dtype=bool).reshape(shape)
    masks["sub-13"] = np.array([1, 0, 1, 0], dtype=bool).reshape(shape)
    common = np.logical_and.reduce(list(masks.values()))
    pinned = tmp_path / "common-mask.npz"
    np.savez_compressed(
        pinned, mask=common, mask_affine=affine, subjects=np.array(sorted(masks))
    )

    results = tmp_path / "results"
    for subject, values in [
        ("sub-12", [[0.1, 0.2], [0.9, 0.8], [0.5, 0.6], [0.7, 0.4]]),
        ("sub-13", [[0.3, 0.4], [0.6, 0.7]]),
    ]:
        folder = results / subject
        folder.mkdir(parents=True)
        np.savez(
            folder / "encoding_results_llm_qwen35_9b__all__main_layer_1.npz",
            subject=subject,
            layer="llm_qwen35_9b__all__main_layer_1",
            band_names=["main"],
            mask=masks[subject],
            mask_affine=affine,
            performances_main=np.array(values, dtype=np.float32),
            partition_ids=[4, 8],
        )

    atlas = tmp_path / "atlas.nii.gz"
    nib.save(nib.Nifti1Image(np.ones(shape, dtype=np.int16), affine), atlas)
    labels = tmp_path / "atlas.txt"
    bases = {base for names in roi.ROI_GROUPING.values() for base in names}
    names = ["Calcarine_L"] + [f"{base}_L" for base in sorted(bases - {"Calcarine"})]
    labels.write_text("\n".join(f"label {name} {i}" for i, name in enumerate(names, 1)))
    selected_masks = {
        subject: (masks[subject], affine, int(masks[subject].sum()))
        for subject in ("sub-12", "sub-13")
    }
    return results, pinned, atlas, labels, selected_masks


def run_roi(tmp_path, fixture_data, name, *, subjects=None, pinned=True):
    results, mask, atlas, labels, _ = fixture_data
    output = tmp_path / f"{name}.csv"
    command = [
        sys.executable,
        "-m",
        "reason_to_play.analysis.neural.roi",
        "--results-dir",
        str(results),
        "--atlas",
        str(atlas),
        "--atlas-labels",
        str(labels),
        "--output",
        str(output),
        "--fit-condition",
        "main-only",
        "--bands",
        "main",
        "--workers",
        "1",
    ]
    if subjects:
        command.extend(["--subjects", *subjects])
    if pinned:
        command.extend(["--common-mask", str(mask)])
    # Exercise the installed module away from the repository working directory.
    subprocess.run(command, cwd=tmp_path, check=True, capture_output=True, text=True)
    return pd.read_csv(output), json.loads(
        output.with_suffix(".csv.reproducibility.json").read_text()
    )


def test_pinned_support_survives_filtering_result_cohort(fixture_data, tmp_path):
    full, full_record = run_roi(tmp_path, fixture_data, "both-subjects")
    subset, subset_record = run_roi(
        tmp_path, fixture_data, "one-subject", subjects=["sub-12"]
    )
    pd.testing.assert_frame_equal(
        full.loc[full.subject == "sub-12"].reset_index(drop=True), subset
    )
    np.testing.assert_array_equal(subset.partition, [4, 8])
    np.testing.assert_allclose(subset.performance, [0.1, 0.2])
    assert subset.n_voxels.tolist() == [1, 1]
    assert full_record["selection"]["result_subjects"] == ["sub-12", "sub-13"]
    assert subset_record["selection"]["result_subjects"] == ["sub-12"]
    assert full_record["common_mask"] == subset_record["common_mask"]
    assert subset_record["common_mask"]["subjects"] == [
        f"sub-{number:02d}" for number in range(1, 33)
    ]
    assert subset_record["selection"]["fit_conditions"] == ["main-only"]
    for field in ["atlas", "atlas_labels", "common_mask", "output"]:
        artifact = subset_record[field]
        assert (
            artifact["sha256"]
            == hashlib.sha256(Path(artifact["path"]).read_bytes()).hexdigest()
        )

    inferred, inferred_record = run_roi(
        tmp_path, fixture_data, "new-experiment", subjects=["sub-12"], pinned=False
    )
    assert inferred.n_voxels.tolist() == [4, 4]
    assert not np.allclose(subset.performance, inferred.performance)
    assert inferred_record["common_mask"]["subjects"] == ["sub-12"]
    assert Path(inferred_record["common_mask"]["path"]).is_file()


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("empty", "empty"),
        ("shape", "shape"),
        ("affine", "affine"),
        ("nonfinite_affine", "affine"),
        ("nonbinary", "binary"),
        ("outside_subject", "not a subset of sub-13"),
        ("missing_subjects", "lacks"),
        ("duplicate_subjects", "unique"),
    ],
)
def test_invalid_fixed_mask_is_rejected(fixture_data, tmp_path, defect, message):
    _, pinned, _, _, selected_masks = fixture_data
    with np.load(pinned, allow_pickle=False) as archive:
        data = dict(archive)
    if defect == "empty":
        data["mask"][:] = False
    elif defect == "shape":
        data["mask"] = np.ones((2, 1, 1), dtype=bool)
    elif defect == "affine":
        data["mask_affine"][0, 3] = 10
    elif defect == "nonfinite_affine":
        data["mask_affine"][0, 3] = np.nan
    elif defect == "nonbinary":
        data["mask"] = data["mask"].astype(float)
        data["mask"][0, 0, 0] = 0.5
    elif defect == "outside_subject":
        data["mask"][1, 0, 0] = True
    elif defect == "missing_subjects":
        data.pop("subjects")
    elif defect == "duplicate_subjects":
        data["subjects"] = np.array(["sub-12", "sub-12"])
    invalid = tmp_path / "invalid-mask.npz"
    np.savez(invalid, **data)
    with pytest.raises(ValueError, match=message):
        roi.build_common_mask_and_indexing(selected_masks, invalid)
