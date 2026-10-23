"""Regression tests for safe zip staging in build_acceptance_package.

Run directly in isolated mode without installing the project:

  python -I -B scripts/check_build_package_safety.py

The tests verify that an existing destination zip is never overwritten or
removed, that a partially written destination is preserved after a copy error,
and that only the temporary zip created by this process is cleaned up.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import build_acceptance_package as build  # noqa: E402


def read_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_stage(root: Path) -> Path:
    stage = root / "stage"
    stage.mkdir(parents=True, exist_ok=True)
    (stage / "hello.txt").write_text("hello staging\n", encoding="utf-8")
    return stage


def test_existing_destination_is_preserved() -> None:
    with tempfile.TemporaryDirectory(prefix="rag_pkg_safe_") as temp:
        root = Path(temp)
        stage = make_stage(root)
        output = root / "target.zip"
        output.write_bytes(b"pre-existing user artifact, do not delete\n")
        before = read_hash(output)
        try:
            build._zip_stage(stage, output)
        except RuntimeError as exc:
            assert "destination left intact" in str(exc)
        else:
            raise AssertionError("expected RuntimeError for existing destination")
        after = read_hash(output)
        assert output.read_bytes().startswith(b"pre-existing user artifact")
        assert before == after, "existing destination hash changed"
        leftovers = list(root.glob("rag_submission_*.zip"))
        assert not leftovers, f"temporary zip not cleaned: {leftovers}"


def test_copy_error_preserves_partial_output() -> None:
    with tempfile.TemporaryDirectory(prefix="rag_pkg_safe_") as temp:
        root = Path(temp)
        stage = make_stage(root)
        output = root / "target.zip"
        real_copy = build.shutil.copyfileobj
        original_copy = shutil.copyfileobj

        def failing_copy(source, target, *args, **kwargs):
            # Preserve the behavior used while writing the temp zip. Only the
            # final exclusive-copy step should fail.
            if type(target).__name__ == "_ZipWriteFile":
                return real_copy(source, target, *args, **kwargs)
            target.write(b"partial bytes")
            raise OSError("simulated copy failure after opening destination")

        build.shutil.copyfileobj = failing_copy
        try:
            try:
                build._zip_stage(stage, output)
            except RuntimeError as exc:
                assert "output was NOT deleted" in str(exc)
            else:
                raise AssertionError("expected RuntimeError for simulated copy failure")
        finally:
            build.shutil.copyfileobj = original_copy
        assert output.exists(), "partial destination must remain for inspection"
        assert output.read_bytes().startswith(b"partial bytes")
        leftovers = list(root.glob("rag_submission_*.zip"))
        assert not leftovers, f"temporary zip not cleaned: {leftovers}"


def test_zip_produced_only_when_target_absent() -> None:
    with tempfile.TemporaryDirectory(prefix="rag_pkg_safe_") as temp:
        root = Path(temp)
        stage = make_stage(root)
        output = root / "target.zip"
        build._zip_stage(stage, output)
        assert output.is_file(), "zip should be created when target is absent"
        with zipfile.ZipFile(output) as archive:
            names = archive.namelist()
        assert "hello.txt" in names
        assert output.stat().st_size > 0


def main() -> int:
    tests = (
        test_existing_destination_is_preserved,
        test_copy_error_preserves_partial_output,
        test_zip_produced_only_when_target_absent,
    )
    for test in tests:
        test_name = test.__name__
        try:
            test()
        except Exception as exc:
            print(f"FAIL {test_name}: {exc!r}")
            return 1
        print(f"PASS {test_name}")
    print("All build package safety tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
