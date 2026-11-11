"""Resume pieces of the frame-sequence nodes. Needs av/numpy/torch (a worker's venv):  python tests/test_resume_farm.py <__init__.py>"""
import sys, types, os, tempfile, importlib.util, glob, numpy as np, av, torch
from fractions import Fraction
T = tempfile.mkdtemp()
sys.modules["folder_paths"] = types.SimpleNamespace(get_temp_directory=lambda: T + "/tmp", get_output_directory=lambda: T + "/out",
    get_save_image_path=lambda prefix, base, w, h: (os.makedirs(os.path.join(base, os.path.dirname(prefix)), exist_ok=True) or os.path.join(base, os.path.dirname(prefix)), os.path.basename(prefix), 1, os.path.dirname(prefix), prefix),
    get_filename_list=lambda k: [])
sys.modules["nodes"] = types.SimpleNamespace(LoraLoaderModelOnly=object)
spec = importlib.util.spec_from_file_location("eu", sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
W, C, V = m.FrameSeqWrite(), m.FrameSeqCheck(), m.FrameSeqVideo()
img = lambda n, v=0.5: torch.full((n, 32, 48, 3), v)

# where=output survives in the output folder; temp stays the default
d = W.run(img(4), "job", 0, 10, True, where="output")[0]
assert d.startswith(T + "/out/endo_seq/") and len(glob.glob(d + "/img/*.png")) == 4 and not glob.glob(d + "/img/*.tmp")
assert W.run(img(2), "t", 0, 2, True)[0].startswith(T + "/tmp/")
# check: [0,4) done, [4,8) not, clipped at total, resume off = never done
assert C.run("job", "output", 0, 4, 10, True)[0] is True
assert C.run("job", "output", 4, 4, 10, True)[0] is False
W.run(img(6), "job", 4, 10, False, where="output")             # 4..9
assert C.run("job", "output", 8, 4, 10, True)[0] is True        # wants 8..11, only 8..9 exist in a 10-frame clip
assert C.run("job", "output", 0, 4, 10, False)[0] is False
# a different job under the same name stops instead of mixing
for bad in (lambda: C.run("job", "output", 0, 4, 12, True), lambda: W.run(img(2), "job", 0, 10, False, where="output"[:6]) if False else W.run(torch.zeros(2, 64, 48, 3), "job", 0, 10, False, where="output")):
    try: bad(); raise AssertionError("did not stop")
    except ValueError as e: assert "이어서 하기" in str(e), e
# an interrupted write leaves no half file: a .tmp left over is not counted
open(d + "/img/000003.png.tmp", "w").write("x"); os.remove(d + "/img/000003.png")
assert C.run("job", "output", 0, 4, 10, True)[0] is False

# audio from an offset, cut at the video's end; cleanup removes the frames
src = T + "/src.mp4"
with av.open(src, "w") as c:
    vs = c.add_stream("libx264", rate=Fraction(16)); vs.width, vs.height, vs.pix_fmt = 48, 32, "yuv420p"
    a = c.add_stream("aac", rate=44100); a.layout = "stereo"
    for k in range(80):
        for p in vs.encode(av.VideoFrame.from_ndarray(np.full((32, 48, 3), k * 3, np.uint8), format="rgb24")): c.mux(p)
    for p in vs.encode(): c.mux(p)
    t = 0
    for _ in range(int(44100 * 5 / 1024)):
        loud = 0.3 if t >= 2 * 44100 else 0.0                # silent for 2 s, then a tone: the offset must land on the tone
        x = (loud * np.sin(2 * np.pi * 440 * (np.arange(1024) + t) / 44100)).astype(np.float32)
        fr = av.AudioFrame.from_ndarray(np.stack([x, x]), format="fltp", layout="stereo"); fr.sample_rate = 44100; fr.pts = t; t += 1024
        for p in a.encode(fr): c.mux(p)
    for p in a.encode(): c.mux(p)
V_ = types.SimpleNamespace(get_stream_source=lambda: src)
d2 = W.run(img(32), "clip2", 0, 32, True, where="output")[0]      # 2 s at 16 fps
out = V.run(True, 16.0, 32, "x/part", "x/u", "img", True, seq_dir=d2, audio_from=V_, audio_start=2.0, cleanup=True)["result"][0]
with av.open(out) as c:
    v, a = c.streams.video[0], c.streams.audio[0]
    vd = float(v.duration * v.time_base); ad = float(a.duration * a.time_base) if a.duration else None
    first = float(next(p for p in c.demux(a) if p.pts is not None).pts * a.time_base)
    c.seek(0)
    pcm = np.concatenate([f.to_ndarray().reshape(-1) for f in c.decode(audio=0)])
rms_head = float(np.sqrt((pcm[:44100 // 2] ** 2).mean()))
print("video", vd, "s · audio", ad, "s · first audio pts", first, "s · loudness of the first 0.5 s", round(rms_head, 3))
assert rms_head > 0.1, "the sound does not start at the offset"
assert ad is not None and abs(ad - vd) < 0.1 and first < 0.05
assert not os.path.exists(d2), "cleanup left the frames"
print("ALL OK")
