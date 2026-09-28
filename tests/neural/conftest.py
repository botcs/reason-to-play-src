"""Small explicit participant-file fixtures for neural integration tests."""

import numpy as np
import pytest

from data.neural import (
    BOLD_FIELDS,
    NUISANCE_FIELDS,
    SAMPLE_ARCHIVE_FIELDS,
    coverage_from_missing,
    write_core_inputs,
    write_feature_archive,
)


@pytest.fixture
def participant_writer():
    def write(path, values):
        samples = {key: values[key] for key in SAMPLE_ARCHIVE_FIELDS if key in values}
        n_samples = len(samples["tr_play_idx"])
        bold = {key: values[key] for key in BOLD_FIELDS if key in values}
        bold.setdefault("voxel_ts", np.zeros((1, n_samples), dtype=np.float32))
        bold.setdefault("mask", np.ones((bold["voxel_ts"].shape[0], 1, 1), dtype=bool))
        bold.setdefault("mask_affine", np.eye(4))
        bold.setdefault("n_voxels", bold["voxel_ts"].shape[0])
        nuisance = {key: values[key] for key in NUISANCE_FIELDS if key in values}
        write_core_inputs(
            path,
            bold,
            samples,
            nuisance,
            nuisance_coverage=coverage_from_missing(samples, []),
        )
        if "fc1_aligned" in values:
            write_feature_archive(
                path / "model-features/ddqn.npz",
                {"fc1_aligned": values["fc1_aligned"]},
                path,
                coverage_from_missing(samples, []),
            )
        return path

    return write
