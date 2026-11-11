import sys, types, os, tempfile, importlib.util, numpy as np, av, torch
T = tempfile.mkdtemp()
sys.modules["folder_paths"] = types.SimpleNamespace(get_temp_directory=lambda: T + "/tmp", get_output_directory=lambda: T + "/out",
    get_save_image_path=lambda prefix, base, w, h: (os.makedirs(os.path.join(base, os.path.dirname(prefix)), exist_ok=True) or os.path.join(base, os.path.dirname(prefix)), os.path.basename(prefix), 1, os.path.dirname(prefix), prefix),
    get_filename_list=lambda k: [])
sys.modules["nodes"] = types.SimpleNamespace(LoraLoaderModelOnly=object)
spec = importlib.util.spec_from_file_location("eu", sys.argv[1]); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
def make(path, audio, n=40, fps=16, w=96, h=160):
    from fractions import Fraction
    with av.open(path, "w") as c:
        vs = c.add_stream("libx264", rate=Fraction(fps)); vs.width, vs.height, vs.pix_fmt = w, h, "yuv420p"
        if audio: as_ = c.add_stream("aac", rate=44100); as_.layout = "stereo"
        for k in range(n):
            for p in vs.encode(av.VideoFrame.from_ndarray(np.full((h, w, 3), k * 5, np.uint8), format="rgb24")): c.mux(p)
        for p in vs.encode(): c.mux(p)
        if audio:
            t = 0
            for _ in range(int(44100 * (n / fps + 1.0) / 1024)):    # 1 s longer than the video: must be cut
                fr = av.AudioFrame.from_ndarray(np.zeros((2, 1024), np.float32), format="fltp", layout="stereo"); fr.sample_rate = 44100; fr.pts = t; t += 1024
                for p in as_.encode(fr): c.mux(p)
            for p in as_.encode(): c.mux(p)
V = lambda p: types.SimpleNamespace(get_stream_source=lambda: p)
for audio in (True, False):
    src = f"{T}/src_{audio}.mp4"; make(src, audio)
    print("VideoInfo", audio, m.VideoInfo().run(V(src)))
    imgs = torch.rand(40, 160, 192, 3)                           # upscaled 2x width
    d = m.FrameSeqWrite().run(imgs, f"t{audio}", 0, 40, True)[0]
    r = m.FrameSeqVideo().run(True, 32.0 if False else 16.0, 40, f"x/up{audio}", "x/u", "img", True, seq_dir=d, audio_from=V(src))
    out = r["result"][0]
    with av.open(out) as c:
        v = c.streams.video[0]; a = c.streams.audio
        vd = float(v.duration * v.time_base) if v.duration else c.duration / 1e6
        ad = (float(a[0].duration * a[0].time_base) if a and a[0].duration else None)
        nv = sum(1 for _ in c.decode(video=0))
    print(" out", os.path.basename(out), "frames", nv, "size", (v.codec_context.width, v.codec_context.height), "video s", round(vd, 3), "audio tracks", len(a), "audio s", ad and round(ad, 3))
    assert nv == 40 and (len(a) == 1) == audio
    if audio: assert ad is not None and ad <= vd + 0.1, (ad, vd)
    assert not os.path.exists(out + ".noaudio.mp4")
print("ALL OK")
