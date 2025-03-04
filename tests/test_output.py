import os
import mesh2scattering as m2s
import pyfar as pf
import shutil
import pytest
import numpy as np
import sofar as sf


@pytest.mark.parametrize("project_name", [
    'sine', 'reference',
])
def test_write_pattern(project_name, tmpdir):
    base_dir = os.path.dirname(__file__)
    # copy test directory
    test_dir = os.path.join(tmpdir, project_name)
    shutil.copytree(
        os.path.join(
            base_dir, 'resources', 'output', project_name),
        test_dir)

    written_files = m2s.output.write_pressure(test_dir)

    assert len(written_files) == 1
    assert os.path.isfile(written_files[0])
    assert os.path.join(
            test_dir, '..',
            f'{project_name}_gaussian_63.pressure.sofa') in written_files

    sofa = sf.read_sofa(written_files[0])
    pressure, sources, receivers = pf.io.convert_sofa(sofa)

    # check if the sofa file is correct
    assert pressure.cshape[0] == sources.csize
    assert pressure.cshape[1] == receivers.csize
    assert isinstance(sofa.SpeedOfSound, float)
    assert isinstance(sofa.SampleStructuralWavelengthX, float)
    assert isinstance(sofa.SampleStructuralWavelengthY, float)
    assert isinstance(sofa.SampleDiameter, float)
    assert isinstance(sofa.SourceWeights, np.ndarray)
    assert isinstance(sofa.ReceiverWeights, np.ndarray)
    assert sofa.ReceiverWeights.size == receivers.csize
    assert sofa.SourceWeights.size == sources.csize
    if project_name == 'sine':
        assert sofa.SampleSymmetryAzimuth == '90,180'
        assert bool(sofa.SampleSymmetryRotational) is False
    elif project_name == 'reference':
        assert sofa.SampleSymmetryAzimuth == ''
        assert bool(sofa.SampleSymmetryRotational) is True
    else:
        raise AssertionError()


def test_import():
    from mesh2scattering import output
    assert output


@pytest.mark.parametrize(("folders", "issue", "errors", "nots"), [
    # no issues single NC.out filejoin
    (["case_0"], False, [], []),
    # issues in NC.out that are corrected by second file NC1-1.out
    (["case_4"], False, [], []),
    # missing frequencies
    (["case_1"], True,
     ["Frequency steps that were not calculated:\n59, 60"], []),
    # convergence issues
    (["case_2"], True,
     ["Frequency steps that did not converge:\n18, 42"], []),
    # input/mesh issues
    (["case_3"], True,
     ["Frequency steps that were not calculated:\n59, 60",
      "Frequency steps with bad input:\n58"], []),
    # no issues in source 1 but issues in source 2
    (["case_0", "case_1"], True,
     ["Detected issues for source 2",
      "Frequency steps that were not calculated:\n59, 60"],
     ["Detected issues for source 1"]),
])
def test_project_report(folders, issue, errors, nots, tmpdir):
    """Test issues found by the project report."""
    cwd = os.path.dirname(__file__)
    data_nc = os.path.join(cwd, 'resources', 'nc.out')
    # create fake project structure
    os.mkdir(os.path.join(tmpdir, "NumCalc"))
    os.mkdir(os.path.join(tmpdir, "report"))
    shutil.copyfile(os.path.join(data_nc, "parameters.json"),
                    os.path.join(tmpdir, "parameters.json"))
    for ff, folder in enumerate(folders):
        shutil.copytree(os.path.join(data_nc, folder),
                        os.path.join(tmpdir, "NumCalc", f"source_{ff + 1}"))

    # run the project report
    issues, report = m2s.output.write_output_report(tmpdir)

    # test the output
    assert issues is issue
    for error in errors:
        assert error in report
    for no in nots:
        assert no not in report
    if issue:
        assert os.path.isfile(os.path.join(
            tmpdir, "report", "report_issues.txt"))
        assert ("For more information check report/report_source_*.csv "
                "and the NC*.out files located at NumCalc/source_*") in report
    else:
        assert not os.path.isfile(os.path.join(
            tmpdir, "report", "report_issues.txt"))
