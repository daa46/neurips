from __future__ import annotations

import hashlib
import json

from hrms.artifacts import (
    artifact_manifest,
    sha256_file,
    source_tree_record,
    write_artifact_manifest,
)
from scripts.write_manifest import build_tree_manifest


def test_artifact_manifest_hashes_inputs_outputs_and_omits_timestamps(tmp_path):
    source = tmp_path / "input.csv"
    product = tmp_path / "output.csv"
    source.write_text("a\n1\n")
    product.write_text("b\n2\n")

    manifest = artifact_manifest(
        inputs={"source": source},
        artifacts={"product": product},
        parameters={"seed": 42},
        base_dir=tmp_path,
    )

    assert manifest["inputs"]["source"]["path"] == "input.csv"
    assert manifest["artifacts"]["product"]["sha256"] == hashlib.sha256(
        product.read_bytes()
    ).hexdigest()
    assert manifest["parameters"] == {"seed": 42}
    assert "created_at" not in manifest


def test_source_tree_digest_is_order_invariant_and_content_sensitive(tmp_path):
    first = tmp_path / "a.py"
    second = tmp_path / "b.py"
    first.write_text("a = 1\n")
    second.write_text("b = 2\n")
    original = source_tree_record([first, second], base_dir=tmp_path)
    permuted = source_tree_record([second, first], base_dir=tmp_path)
    assert original == permuted
    second.write_text("b = 3\n")
    changed = source_tree_record([first, second], base_dir=tmp_path)
    assert changed["sha256"] != original["sha256"]


def test_written_manifest_and_tree_manifest_are_self_excluding(tmp_path):
    root = tmp_path / "bundle"
    root.mkdir()
    artifact = root / "result.txt"
    artifact.write_text("deterministic\n")
    direct = root / "direct.json"
    write_artifact_manifest(
        direct,
        inputs={},
        artifacts={"result": artifact},
        base_dir=tmp_path,
    )
    assert sha256_file(artifact) == json.loads(direct.read_text())["artifacts"]["result"][
        "sha256"
    ]

    tree = root / "manifest.json"
    build_tree_manifest(root=root, output=tree)
    first = json.loads(tree.read_text())
    build_tree_manifest(root=root, output=tree)
    second = json.loads(tree.read_text())
    assert first == second
    assert "manifest.json" not in first["artifacts"]
