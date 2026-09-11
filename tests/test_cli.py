"""lookfx.cli: the console entry point (`lookfx = lookfx.cli:main`).

Everything here runs on the CPU (``--device cpu``); the video test needs ffmpeg
and is gated by the ``needs_ffmpeg`` marker like the IO suite.
"""

from fractions import Fraction

import pytest
import torch

from lookfx_core.io.image import read_image, write_image
from lookfx_core.io.writer import CODECS
from lookfx.cli import main as cli_main, build_parser


@pytest.fixture
def still(tmp_path):
    p = tmp_path / "in.png"
    write_image(p, torch.rand(1, 40, 60, 3)[0])
    return p


def test_presets_and_probe(still, capsys):
    assert cli_main(["presets"]) == 0
    flare = capsys.readouterr().out.splitlines()
    assert any(line.startswith("cine_blue") for line in flare)
    assert cli_main(["presets", "print_look"]) == 0
    print_look = capsys.readouterr().out.splitlines()
    assert any(line.startswith("Vintage Poster") for line in print_look)
    assert cli_main(["probe", str(still)]) == 0
    out = capsys.readouterr().out
    assert '"kind": "still"' in out and '"width": 60' in out


def test_render_still_via_cli(still, tmp_path):
    out = tmp_path / "out.png"
    rc = cli_main(["render", str(still), str(out), "--flare", "cine_blue", "--light", "0.3,0.4",
                   "--print", "Vintage Poster", "--device", "cpu"])
    assert rc == 0 and out.exists()
    assert read_image(out).shape == (1, 40, 60, 3)


def test_no_steps_exits(still, tmp_path):
    with pytest.raises(SystemExit) as ei:
        cli_main(["render", str(still), str(tmp_path / "out.png")])
    assert "nothing to do" in str(ei.value)


def test_bad_preset_message(still, tmp_path, capsys):
    rc = cli_main(["render", str(still), str(tmp_path / "out.png"), "--flare", "no_such_preset", "--device", "cpu"])
    assert rc == 2
    err = capsys.readouterr().err
    assert err.startswith("error:") and "no_such_preset" in err
    assert "Traceback" not in err


def test_help_has_no_phase_notes_and_all_codecs(capsys):
    with pytest.raises(SystemExit):
        cli_main(["--help"])
    assert "Phase" not in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli_main(["render", "--help"])
    out = capsys.readouterr().out.replace("\n", " ")
    for key in CODECS:
        assert key in out
    # unknown codec keys are rejected by argparse, not deep inside the encoder
    with pytest.raises(SystemExit):
        build_parser().parse_args(["render", "a", "b", "--codec", "vp9"])


@pytest.mark.needs_ffmpeg
def test_render_video_range_h264(tmp_path):
    from lookfx_core.io.probe import probe
    from lookfx_core.io.writer import open_sink
    n, h, w = 6, 48, 64
    src = tmp_path / "clip.mkv"
    s = open_sink(src, width=w, height=h, fps=Fraction(24), codec="ffv1")
    s.write(torch.rand(n, h, w, 3)); s.close()
    out = tmp_path / "out.mp4"
    rc = cli_main(["render", str(src), str(out), "--flare", "cine_blue", "--light", "0.5,0.5",
                   "--codec", "h264", "--start", "1", "--end", "4", "--device", "cpu"])
    assert rc == 0 and out.exists()
    assert probe(out).nb_frames == 3
