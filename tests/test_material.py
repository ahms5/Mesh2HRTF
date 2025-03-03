import mesh2scattering as m2s
import os
import filecmp
import pytest
import pyfar as pf
from pathlib import Path

def test_write_material(tmpdir):
    # test write boundary condition with default values
    filename = os.path.join(tmpdir, "test_material.csv")

    # write data
    m2s.input.write_material(
        filename, "pressure", pf.FrequencyData(0, 0))

    # read and check data
    assert filecmp.cmp(
        os.path.join(os.path.dirname(__file__), 'references', 'SoundSoft.csv'),
        filename,
    )

@pytest.mark.parametrize(('kind', 'check_kind'), [
    ("admittance", ["ADMI", "PRES", "VELO"]),
    ("pressure", ["PRES", "ADMI", "VELO"]),
    ("velocity", ["VELO", "ADMI", "PRES"])])
def test_write_material_kind(kind, check_kind, tmpdir):
    # test if the kind of boundary condition is written correctly

    filename = os.path.join(tmpdir, "test_material.csv")

    # write data
    m2s.input.write_material(
        filename, kind, pf.FrequencyData([1 + 0j, 1.5 + 0.5j], [100, 200]))

    # read and check data
    with open(filename, "r") as f_id:
        file = f_id.readlines()

    assert f"{check_kind[0]}\n" in file
    assert f"{check_kind[1]}\n" not in file
    assert f"{check_kind[2]}\n" not in file


def test_write_material_comment(tmpdir):
    # test if the comment is written

    filename = os.path.join(tmpdir, "test_material.csv")
    comment = "Weird, random data"

    # write data
    m2s.input.write_material(
        filename, "pressure",
        pf.FrequencyData([1 + 0j, 1.5 + 0.5j], [100, 200]), comment)

    # read and check data
    with open(filename, "r") as f_id:
        file = f_id.readlines()

    assert file[0] == "# " + comment + "\n"
    assert file[1] == "#\n"
    assert file[2] == "# Keyword to define the boundary condition:\n"


def test_write_material_wrong_data(tmpdir):
    filename = os.path.join(tmpdir, "test_material.csv")

    with pytest.raises(
            ValueError,
            match="data must be a pyfar.FrequencyData object."):
        m2s.input.write_material(
            filename, "pressure", 'data')


def test_write_material_wrong_filename(tmpdir):
    data = pf.FrequencyData([1], [100])

    with pytest.raises(
            ValueError,
            match="filename must be a string or Path."):
        m2s.input.write_material(
            55, "pressure", data)
    with pytest.raises(
            ValueError,
            match="The filename must end with .csv."):
        m2s.input.write_material(
            Path(os.path.join(tmpdir, 'test.ccc')), "pressure", data)


def test_write_material_wrong_comment(tmpdir):
    filename = os.path.join(tmpdir, "test_material.csv")
    data = pf.FrequencyData([1], [100])

    with pytest.raises(
            ValueError,
            match="comment must be a string or None."):
        m2s.input.write_material(
            filename, "pressure", data, 5)


def test_write_material_wrong_kind(tmpdir):
    # test if the comment is written

    filename = os.path.join(tmpdir, "test_material.csv")
    data = pf.FrequencyData([1], [100])

    with pytest.raises(
            ValueError,
            match="kind must be admittance, pressure, or velocity."):
        m2s.input.write_material(
            filename, "pres", data)
    with pytest.raises(
            ValueError,
            match="kind must be a string."):
        m2s.input.write_material(
            filename, 5, data)

