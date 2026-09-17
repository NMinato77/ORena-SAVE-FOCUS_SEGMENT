from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

QWEN_FILES = (
    "config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "merges.txt",
    "vocab.json",
    "chat_template.json",
    "model.safetensors.index.json",
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
)

SIGLIP_FILES = (
    ".gitattributes",
    "README.md",
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer_config.json",
)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"expected JSON object: {path}")
    return value


class PublicReleaseContractTests(unittest.TestCase):
    def test_build_entrypoint_stages_assets_automatically(self) -> None:
        build = (ROOT / "do_build.sh").read_text(encoding="utf-8")
        dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn('source "${SCRIPT_DIR}/prepare_assets.sh"', build)
        self.assertIn("docker build", build)
        self.assertIn("COPY --chown=user:user resources/ /opt/app/resources/", dockerfile)
        self.assertIn("COPY --chown=user:user inference.py /opt/app/", dockerfile)

    def test_build_context_excludes_development_payloads(self) -> None:
        dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
        for entry in ("*.tar.gz", "test/", "tests/", "training/", "docs/"):
            self.assertIn(entry, dockerignore)

    def test_generated_submission_staging_assets_are_not_tracked(self) -> None:
        gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("resources/specialists/", gitignore)
        self.assertIn("resources/qwen3-vl-8b/", gitignore)
        self.assertIn("resources/q2_siglip_base_patch16_384/", gitignore)

    def test_public_runtime_has_no_private_absolute_paths(self) -> None:
        forbidden = ("/home/minato/", "/mnt/cloudy", "/data0/", "/workspace/")
        suffixes = {".py", ".sh", ".md", ".json", ".txt", ".gitattributes", ".gitignore"}
        for path in ROOT.rglob("*"):
            if (
                not path.is_file()
                or ".git" in path.parts
                or "tests" in path.parts
                or path.suffix not in suffixes
            ):
                continue
            text = path.read_text(encoding="utf-8")
            for value in forbidden:
                self.assertNotIn(value, text, f"private path in {path}")

    def test_public_release_provenance_is_post_host_record(self) -> None:
        release_path = ROOT / "docs/release_provenance_manifest.json"
        release = read_json(release_path)
        self.assertEqual(release["record_type"], "post_host_submission_release_provenance")
        self.assertEqual(
            release["submitted_image"]["image_id"],
            "sha256:21ed8ffde4ba5938b3ad1cdda97b3b2e41e3f701df460a6eade23fe229b2c1f3",
        )
        self.assertEqual(
            release["submitted_image"]["archive"]["sha256"],
            "5d7d0ffc10728d001bf086d432a06386a8f48d381ffe1e3a30efedbce7716087",
        )
        self.assertEqual(release["submitted_image"]["archive"]["manifest_json"], "present")
        self.assertEqual(release["validation"]["official_test_run"]["status"], "PASS")
        self.assertFalse((ROOT / "resources/final_provenance_manifest.json").exists())

    def test_training_manifests_are_path_portable(self) -> None:
        manifest_dir = ROOT / "training/artifacts/final_manifests"
        manifests = sorted(manifest_dir.glob("*.jsonl"))
        self.assertGreaterEqual(len(manifests), 7)
        for path in manifests:
            with path.open(encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            self.assertTrue(rows, path)
            for row in rows:
                self.assertTrue(
                    str(row["video_path"]).startswith("${FOCUS_DATA_ROOT}/"),
                    f"non-portable video path in {path}: {row['video_path']}",
                )

    def test_specialist_release_bindings_have_public_sources(self) -> None:
        config = read_json(ROOT / "resources/final_candidate_config.json")
        specialists = config["specialists"]
        for name, filename in (
            ("general", "general_final_union_adapted_weights.safetensors"),
            ("agg", "agg_final_union_adapted_weights.safetensors"),
        ):
            self.assertEqual(specialists[name]["checkpoint"], f"specialists/{filename}")
            source = ROOT / "training/weights" / filename
            self.assertTrue(source.is_file(), source)
            self.assertGreater(source.stat().st_size, 1_000_000, source)

    def test_q2_classifier_hash_matches_runtime_contract(self) -> None:
        manifest = read_json(ROOT / "resources/q2_siglip_manifest.json")
        expected = str(manifest["classifier"]["sha256"])
        classifier = ROOT / "resources/q2_siglip_text_r5.joblib"
        actual = hashlib.sha256(classifier.read_bytes()).hexdigest()
        self.assertEqual(actual, expected)

    def test_shell_scripts_parse(self) -> None:
        for name in ("do_build.sh", "do_save.sh", "do_test_run.sh", "prepare_assets.sh"):
            result = subprocess.run(["bash", "-n", str(ROOT / name)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_prepare_assets_stages_all_required_asset_classes(self) -> None:
        """Exercise the staging script with tiny synthetic model files."""

        with tempfile.TemporaryDirectory(prefix="seg010-assets-") as directory:
            sandbox = Path(directory)
            script = sandbox / "prepare_assets.sh"
            shutil.copy2(ROOT / "prepare_assets.sh", script)
            script.chmod(script.stat().st_mode | stat.S_IXUSR)

            qwen = sandbox / "qwen"
            siglip = sandbox / "siglip"
            qwen.mkdir()
            siglip.mkdir()
            for name in QWEN_FILES:
                (qwen / name).write_bytes(b"qwen-fixture")
            for name in SIGLIP_FILES:
                (siglip / name).write_bytes(b"siglip-fixture")

            general = sandbox / "general.safetensors"
            aggregation = sandbox / "aggregation.safetensors"
            general.write_bytes(b"general-fixture")
            aggregation.write_bytes(b"aggregation-fixture")

            environment = {
                **os.environ,
                "QWEN_MODEL_PATH": str(qwen),
                "Q2_ENCODER_PATH": str(siglip),
                "Q2_ROUTER_PATH": str(ROOT / "resources/q2_siglip_text_r5.joblib"),
                "GENERAL_FINAL_STATE_PATH": str(general),
                "AGG_FINAL_STATE_PATH": str(aggregation),
            }
            result = subprocess.run(
                ["bash", str(script)],
                cwd=sandbox,
                env=environment,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Assets ready", result.stdout)
            resources = sandbox / "resources"
            for name in QWEN_FILES:
                self.assertTrue((resources / "qwen3-vl-8b" / name).is_file())
            for name in SIGLIP_FILES:
                self.assertTrue((resources / "q2_siglip_base_patch16_384" / name).is_file())
            self.assertTrue((resources / "q2_siglip_text_r5.joblib").is_file())
            self.assertEqual(
                (resources / "specialists/general_final_union_adapted_weights.safetensors").read_bytes(),
                b"general-fixture",
            )
            self.assertEqual(
                (resources / "specialists/agg_final_union_adapted_weights.safetensors").read_bytes(),
                b"aggregation-fixture",
            )


if __name__ == "__main__":
    unittest.main()
