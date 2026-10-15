"""Guards of the one-shot holdout script (F4 part B), tested on throwaway git repos only. The real
holdout run is never started here: every case stops in the preflight checks, before any data is read."""
import subprocess

import pytest

from forecast import run_f4_holdout as hold


def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@t",
                           *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    (tmp_path / "a.txt").write_text("x\n")
    (tmp_path / ".gitignore").write_text("data/\n")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-q", "-m", "init")
    return tmp_path


def test_clean_repo_passes_and_records_the_commit(repo):
    h = hold.preflight(repo, repo / "results" / "f4_holdout.json")
    assert h == git(repo, "rev-parse", "HEAD") and len(h) == 40


def test_refuses_a_modified_file(repo):
    (repo / "a.txt").write_text("changed\n")
    with pytest.raises(hold.HoldoutRunError, match="not clean"):
        hold.preflight(repo, repo / "results" / "f4_holdout.json")


def test_refuses_an_untracked_file(repo):
    (repo / "new.py").write_text("print(1)\n")
    with pytest.raises(hold.HoldoutRunError, match="not clean"):
        hold.preflight(repo, repo / "results" / "f4_holdout.json")


def test_ignored_files_do_not_count(repo):
    (repo / "data").mkdir()
    (repo / "data" / "big.parquet").write_text("x")
    hold.preflight(repo, repo / "results" / "f4_holdout.json")


def test_refuses_when_the_holdout_result_already_exists(repo):
    out = repo / "results" / "f4_holdout.json"
    out.parent.mkdir()
    out.write_text("{}\n")
    git(repo, "add", ".")
    git(repo, "commit", "-q", "-m", "result")
    with pytest.raises(hold.HoldoutRunError, match="already exists"):
        hold.preflight(repo, out)


def test_refuses_outside_a_git_repo(tmp_path):
    with pytest.raises(hold.HoldoutRunError):
        hold.preflight(tmp_path, tmp_path / "results" / "f4_holdout.json")


def test_main_stops_before_reading_data_when_dirty(repo, monkeypatch):
    (repo / "a.txt").write_text("changed\n")

    def no_data(*a, **k):
        raise AssertionError("data was read")

    monkeypatch.setattr(hold.model, "load_data", no_data)
    with pytest.raises(hold.HoldoutRunError):
        hold.main(root=repo)
