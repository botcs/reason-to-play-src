"""Dependency-free checks for preprocessing configurations and the container boundary."""

import argparse
import hashlib
import importlib.util
import json
import shlex
import tempfile
import unittest
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "run_fmriprep", ROOT / "reconstruction/tomov23/fmriprep.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def fmriprep_option_parser():
    """Reconstruct option semantics from the actual pinned upstream parser AST."""
    fixture = json.loads(
        (ROOT / "tests/neural/fixtures/fmriprep-24.1.0-options.json").read_text()
    )
    parser = argparse.ArgumentParser()
    converters = {
        "int": int,
        "PositiveInt": int,
        "float": float,
        "SliceTimeRef": float,
        "_to_gb": lambda value: int(value) / 1000,
    }
    for option in fixture["options"]:
        if not option["flags"][0].startswith("-"):
            continue
        action = option["action"]
        kwargs = {
            "dest": option["dest"],
            "default": option["default"],
            "action": "store"
            if action in ("OutputReferencesAction", "ToDict")
            else action,
        }
        if kwargs["action"] == "store":
            kwargs["type"] = converters.get(option["type"], str)
            if option["nargs"] is not None:
                kwargs["nargs"] = option["nargs"]
            if "choices" in option:
                kwargs["choices"] = option["choices"]
            if "const" in option:
                kwargs["const"] = option["const"]
        parser.add_argument(*option["flags"], **kwargs)
    return parser


class FmriprepPipelineTests(unittest.TestCase):
    def test_original_launcher_is_preserved_and_agrees_with_selected_configs(self):
        directory = runner.PROVENANCE / "original-launcher"
        record = json.loads((directory / "source.json").read_text())
        self.assertEqual(record["commit"], "969141d09369ecd783965fadbbe2a1920065cf24")
        for file_record in record["files"]:
            path = runner.PROVENANCE / file_record["path"]
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), file_record["sha256"]
            )
        launcher = (directory / "run_fmriprep.sh").read_text()
        raw_options = launcher.split("nipreps/fmriprep:24.1.0", 1)[1].split(
            "--verbose", 1
        )[0]
        raw_options = raw_options.replace("\\\n", " ")
        original, _ = fmriprep_option_parser().parse_known_args(
            shlex.split(raw_options)
        )
        selected = json.loads((runner.PROVENANCE / "manifest.json").read_text())[
            "selected_configs"
        ]
        for relative in selected.values():
            config = tomllib.loads((runner.PROVENANCE / relative).read_text())
            effective = vars(
                fmriprep_option_parser().parse_args(
                    runner.preprocessing_arguments(config)
                )
            )
            for key in (
                "bold2anat_dof",
                "use_bbr",
                "dummy_scans",
                "run_reconall",
                "hires",
                "regressors_all_comps",
                "regressors_fd_th",
                "regressors_dvars_th",
                "skull_strip_template",
                "skull_strip_fixed_seed",
                "medial_surface_nan",
                "_random_seed",
                "output_spaces",
                "nprocs",
                "omp_nthreads",
                "memory_gb",
                "write_graph",
                "resource_monitor",
            ):
                self.assertEqual(
                    getattr(original, key), effective[key], (relative, key)
                )

    def test_effective_preprocessing_options_match_every_selected_configuration(self):
        parser = fmriprep_option_parser()
        # --config-file alone is overwritten by the pinned upstream parser's
        # non-None defaults.
        self.assertEqual(parser.parse_args([]).bold2anat_dof, 6)
        self.assertTrue(parser.parse_args([]).run_reconall)
        manifest = json.loads((runner.PROVENANCE / "manifest.json").read_text())
        for relative in manifest["selected_configs"].values():
            cfg = tomllib.loads((runner.PROVENANCE / relative).read_text())
            options = vars(parser.parse_args(runner.preprocessing_arguments(cfg)))
            for key, expected in cfg["workflow"].items():
                effective = options.get(key)
                if effective is None:
                    effective = expected
                self.assertEqual(effective, expected, (relative, key))
            for key in (
                "nprocs",
                "omp_nthreads",
                "memory_gb",
                "resource_monitor",
                "stop_on_first_crash",
            ):
                self.assertEqual(options[key], cfg["nipype"][key], (relative, key))
            self.assertEqual(options["write_graph"], cfg["execution"]["write_graph"])
            self.assertEqual(
                " ".join(options["output_spaces"]), cfg["execution"]["output_spaces"]
            )
            self.assertEqual(options["_random_seed"], cfg["seeds"]["master"])

    def test_replay_config_drops_old_database_without_changing_workflow(self):
        manifest = json.loads((runner.PROVENANCE / "manifest.json").read_text())
        for relative in manifest["selected_configs"].values():
            path = runner.PROVENANCE / relative
            original = tomllib.loads(path.read_text())
            replay = tomllib.loads(runner.replay_config_text(path))
            self.assertEqual(replay["workflow"], original["workflow"])
            self.assertEqual(replay["seeds"], original["seeds"])
            self.assertEqual(replay["nipype"], original["nipype"])
            for key in (
                "bids_database_dir",
                "run_uuid",
                "layout",
                "bids_description_hash",
                "log_dir",
            ):
                self.assertNotIn(key, replay["execution"])

    def test_all_original_configurations_are_preserved_and_match_checksums(self):
        manifest = json.loads((runner.PROVENANCE / "manifest.json").read_text())
        self.assertEqual(manifest["configuration_count"], 38)
        self.assertEqual(len(manifest["selected_configs"]), 32)
        self.assertEqual(
            set(manifest["selected_configs"]), {f"sub-{n:02d}" for n in range(1, 33)}
        )
        dofs = []
        for record in manifest["configs"]:
            path = runner.PROVENANCE / record["path"]
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(), record["sha256"]
            )
            cfg = tomllib.loads(path.read_text())
            self.assertEqual(cfg["environment"]["version"], "24.1.0")
            self.assertEqual(cfg["seeds"]["master"], 23)
            dofs.append(cfg["workflow"]["bold2anat_dof"])
        self.assertEqual(dofs.count(6), 1)
        self.assertEqual(dofs.count(9), 37)

    def test_subject_binding_and_config_keep_raw_inputs_read_only(self):
        with tempfile.TemporaryDirectory(prefix="fmri input ") as temporary:
            base = Path(temporary)
            for number in range(1, 33):
                args = argparse.Namespace(
                    subject=f"sub-{number:02d}",
                    bids_dir=base / "raw",
                    output_dir=base / "out",
                    work_dir=base / "work",
                    fs_license=base / "license.txt",
                    templateflow_dir=base / "templateflow",
                    image=base / "image.sif",
                    runtime="apptainer",
                )
                argv, config = runner.command(args)
                cfg = tomllib.loads(config.read_text())
                self.assertEqual(
                    cfg["execution"]["participant_label"], [f"{number:02d}"]
                )
                self.assertEqual(cfg["workflow"]["bold2anat_dof"], 9)
                self.assertIn(f"{args.bids_dir}:/data:ro", argv)
                self.assertIn(f"{args.fs_license}:/license/license.txt:ro", argv)
                self.assertEqual(
                    argv[argv.index("--participant-label") + 1], f"{number:02d}"
                )
                self.assertIn(str(args.image), argv)


if __name__ == "__main__":
    unittest.main()
