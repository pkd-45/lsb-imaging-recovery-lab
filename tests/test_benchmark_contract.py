from pathlib import Path


def test_scripts_exist():
    assert Path("scripts/benchmark.py").is_file()
    assert Path("scripts/run_demo.py").is_file()
    assert Path("scripts/validate_outputs.py").is_file()
