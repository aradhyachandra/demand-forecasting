import pytest
import yaml

from forecast import raw_files


def test_sha256_of_known_content(tmp_path):
    p = tmp_path / "abc.txt"
    p.write_bytes(b"abc")
    assert raw_files.sha256_file(p) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


def test_count_data_rows_with_and_without_final_newline(tmp_path):
    a = tmp_path / "a.csv"
    a.write_text("h\n1\n2\n")
    b = tmp_path / "b.csv"
    b.write_text("h\n1\n2")
    assert raw_files.count_data_rows(a) == 2
    assert raw_files.count_data_rows(b) == 2


def test_fingerprint_write_and_verify(tmp_path, small_paths):
    fp = raw_files.fingerprint(small_paths)
    assert fp["sales"]["data_rows"] == 5
    assert fp["calendar"]["data_rows"] == 5
    assert fp["prices"]["data_rows"] == 8
    out = tmp_path / "raw_files.yaml"
    raw_files.write_fingerprints(fp, out)
    loaded = yaml.safe_load(out.read_text())
    assert loaded == fp
    raw_files.verify(small_paths, loaded)  # matches: no error


def test_verify_stops_on_a_changed_file(tmp_path, small_paths):
    fp = raw_files.fingerprint(small_paths)
    changed = tmp_path / "sales.csv"
    changed.write_text(small_paths["sales"].read_text().replace(",0,2,1,3", ",9,2,1,3"))
    with pytest.raises(ValueError, match="sha256"):
        raw_files.verify({**small_paths, "sales": changed}, fp)
