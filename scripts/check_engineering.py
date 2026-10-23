#!/usr/bin/env python3
"""Offline unit checks for the T8 deployment contract; no LLM/model requests."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["RAG_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("rag_start_app", ROOT / "scripts/start_app.py")
starter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(starter)


class StartupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "data/corpus").mkdir(parents=True)
        (self.root / "data/corpus/local.txt").write_text("Synthetic test document.")
        self.env = {"LLM_BASE_URL": "https://example.invalid/v1",
                    "LLM_API_KEY": "unit-test-only", "LLM_MODEL": "unit-model"}

    def tearDown(self):
        self.tmp.cleanup()

    def test_valid_inputs(self):
        self.assertEqual(starter.validate_inputs(self.root, self.env), [])

    def test_missing_key(self):
        self.env.pop("LLM_API_KEY")
        self.assertTrue(any("LLM_API_KEY" in x for x in starter.validate_inputs(self.root, self.env)))

    def test_placeholder(self):
        self.env["LLM_API_KEY"] = "your_api_key"
        self.assertTrue(starter.validate_inputs(self.root, self.env))

    def test_no_documents(self):
        (self.root / "data/corpus/local.txt").unlink()
        self.assertTrue(any("document" in x for x in starter.validate_inputs(self.root, self.env)))

    def test_empty_documents(self):
        (self.root / "data/corpus/local.txt").write_text("")
        self.assertTrue(starter.validate_inputs(self.root, self.env))

    def test_word_temporary_file_is_not_corpus(self):
        (self.root / "data/corpus/local.txt").unlink()
        (self.root / "data/corpus/~$draft.docx").write_bytes(b"not a document")
        self.assertTrue(starter.validate_inputs(self.root, self.env))

    def test_custom_corpus(self):
        self.env["RAG_CORPUS_DIR"] = "alternate"
        (self.root / "alternate").mkdir()
        (self.root / "alternate/item.md").write_text("Test.")
        self.assertEqual(starter.validate_inputs(self.root, self.env), [])

    def test_docker_uses_fixed_mount(self):
        self.env["RAG_CORPUS_DIR"] = "does-not-exist"
        self.assertEqual(starter.validate_inputs(self.root, self.env, docker=True), [])

    def test_invalid_port(self):
        for value in ("0", "65536", "not-a-port"):
            with self.subTest(value=value):
                self.env["RAG_PORT"] = value
                self.assertTrue(any("RAG_PORT" in x for x in starter.validate_inputs(self.root, self.env)))

    def test_invalid_url(self):
        self.env["LLM_BASE_URL"] = "not-a-url"
        self.assertTrue(starter.validate_inputs(self.root, self.env))

    def test_url_credentials_rejected(self):
        self.env["LLM_BASE_URL"] = "https://user:private@example.invalid/v1"
        self.assertTrue(any("credentials" in x for x in starter.validate_inputs(self.root, self.env)))

    def test_docker_rejects_host_loopback_llm(self):
        self.env["LLM_BASE_URL"] = "http://127.0.0.1:8000/v1"
        self.assertTrue(starter.validate_inputs(self.root, self.env, docker=True))
        self.assertEqual(starter.validate_inputs(self.root, self.env), [])

    def test_invalid_bind_url(self):
        self.env["RAG_HOST"] = "http://localhost"
        self.assertTrue(starter.validate_inputs(self.root, self.env))

    def test_env_quotes_and_comments(self):
        path = self.root / ".env"
        path.write_text('# ignored\nexport LLM_MODEL="test"\nBAD KEY=bad\nX=one=two\n', encoding="utf-8")
        self.assertEqual(starter.read_env(path), {"LLM_MODEL": "test", "X": "one=two"})

    def test_environment_and_cli_precedence(self):
        (self.root / ".env").write_text("RAG_PORT=8000\nLLM_MODEL=file\n")
        args = argparse.Namespace(offline=True, host="127.0.0.1", port=8002, docker=False)
        env = starter.runtime_env(self.root, {"LLM_MODEL": "process", "RAG_PORT": "8001"}, args)
        self.assertEqual(env["LLM_MODEL"], "process")
        self.assertEqual(env["RAG_PORT"], "8002")
        self.assertEqual(env["HF_HUB_OFFLINE"], "1")
        self.assertEqual(env["TRANSFORMERS_OFFLINE"], "1")

    def test_check_never_installs_or_launches(self):
        with patch.object(starter, "ROOT", self.root), patch.dict(os.environ, self.env, clear=True), \
                patch.object(starter, "run") as run, patch.object(starter, "local_python") as python:
            self.assertEqual(starter.main(["--check"]), 0)
            run.assert_not_called()
            python.assert_not_called()
            self.assertFalse((self.root / ".env").exists())

    def test_missing_env_check_is_read_only(self):
        with patch.object(starter, "ROOT", self.root), patch.dict(os.environ, {}, clear=True):
            self.assertEqual(starter.main(["--check"]), 2)
            self.assertFalse((self.root / ".env").exists())

    def test_existing_env_never_overwritten(self):
        path = self.root / ".env"
        path.write_text("LLM_MODEL=preserve\n")
        before = path.read_bytes()
        with patch.object(starter, "ROOT", self.root), patch.dict(os.environ, {}, clear=True):
            self.assertEqual(starter.main(["--check"]), 2)
        self.assertEqual(path.read_bytes(), before)

    def test_template_creation_stops_before_launch(self):
        (self.root / ".env.example").write_text("LLM_MODEL=your_model\n")
        with patch.object(starter, "ROOT", self.root), patch.dict(os.environ, {}, clear=True), \
                patch.object(starter, "run") as run:
            self.assertEqual(starter.main([]), 2)
            run.assert_not_called()
        self.assertEqual((self.root / ".env").read_text(), "LLM_MODEL=your_model\n")


class DeploymentContractTests(unittest.TestCase):
    def test_docker_no_broad_copy_or_secret_copy(self):
        text = (ROOT / "Dockerfile").read_text()
        copies = [line for line in text.splitlines() if line.startswith("COPY ")]
        self.assertEqual(len(copies), 3)
        self.assertFalse(any("COPY . " in x or ".env" in x or "data/" in x for x in copies))
        self.assertIn("USER app", text)
        self.assertIn("HEALTHCHECK", text)
        self.assertIn('CMD ["python", "-m", "demo.app"]', text)

    def test_docker_context_allowlist(self):
        rules = [x for x in (ROOT / ".dockerignore").read_text().splitlines() if x and not x.startswith("#")]
        self.assertEqual(rules[0], "**")
        self.assertEqual({x for x in rules if x.startswith("!")},
                         {"!Dockerfile", "!.dockerignore", "!requirements.txt", "!src/", "!src/*.py", "!demo/", "!demo/*.py"})

    def test_compose_security_and_offline_flags(self):
        text = (ROOT / "compose.yaml").read_text()
        for token in ('127.0.0.1', 'read_only: true', 'create_host_path: false',
                      'model-cache:', 'runtime-results:', 'no-new-privileges:true',
                      'RAG_OFFLINE:', 'HF_HUB_OFFLINE:', 'TRANSFORMERS_OFFLINE:'):
            self.assertIn(token, text)

    def test_git_ignores_private_inputs_but_keeps_deployment(self):
        private = [".env", "data/eval/qa_set.jsonl", "results/test.csv", "repair_outputs/test.json"]
        proc = subprocess.run(["git", "check-ignore", "-z", "--stdin"],
                              input=("\0".join(private)+"\0").encode(),
                              cwd=ROOT, capture_output=True, check=False)
        self.assertEqual(set(proc.stdout.decode().rstrip("\0").split("\0")), set(private))
        public = ["Dockerfile", ".dockerignore", ".gitattributes", "compose.yaml", "start.ps1", "start.sh",
                  "docs/T8_ENGINEERING.md", "scripts/start_app.py", "scripts/check_engineering.py"]
        proc = subprocess.run(["git", "check-ignore", "--no-index", "-z", "--stdin"],
                              input=("\0".join(public)+"\0").encode(), cwd=ROOT,
                              capture_output=True, check=False)
        self.assertEqual(proc.returncode, 1, proc.stdout + proc.stderr)

    def test_startup_scripts_point_to_shared_launcher(self):
        self.assertIn("scripts\\start_app.py", (ROOT / "start.ps1").read_text())
        self.assertIn('"$ROOT/scripts/start_app.py" "$@"', (ROOT / "start.sh").read_text())

    def test_ui_startup_and_empty_corpus(self):
        from demo import app
        with patch.dict(os.environ, {"RAG_HOST": "0.0.0.0", "RAG_PORT": "17860"}), \
                patch.object(app, "build_ui") as ui:
            app.main()
            ui.return_value.queue.return_value.launch.assert_called_once_with(
                server_name="0.0.0.0", server_port=17860, share=False, show_error=False)
        with patch.object(app, "_PIPE", None), patch.object(app.loader, "read_all_docs", return_value={}), \
                patch.object(app.pipeline, "RAGPipeline") as pipeline:
            with self.assertRaises(RuntimeError):
                app.get_pipe()
            pipeline.assert_not_called()

    def test_ui_still_builds_without_models(self):
        from demo import app
        with patch.object(app, "get_pipe", side_effect=AssertionError("Unexpected model load")):
            ui = app.build_ui()
            self.assertIn("components", ui.config)
            self.assertGreater(len(ui.config["components"]), 5)

    def test_python_sources_parse(self):
        for name in ("scripts/start_app.py", "scripts/check_engineering.py", "demo/app.py", "src/config.py"):
            ast.parse((ROOT / name).read_text(encoding="utf-8-sig"), filename=name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
