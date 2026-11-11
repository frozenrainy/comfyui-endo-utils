"""엔도 연구용 유틸 노드.

FrameContactSheet — 영상 프레임을 번호 붙인 격자 한 장으로 (앵커 프레임 번호 고르기용).

WanLoraPair — WAN 2.2 LoRA 한 쌍(HIGH 파일·LOW 파일 + 강도 각각). None 이나 강도 0 이면 그쪽은 파일도 안 읽음.

VideoInfo — 영상의 원래 크기·fps·프레임 수·소리 유무(디코딩 없이). FrameSeqVideo 의 audio_from — 원본 소리를 그대로 붙임.
FrameSeqCheck · FrameSeqWrite(where=output) — 이어서 하기: 프레임을 출력 폴더에 남기고, 다시 실행하면 이미 있는 조각은 건너뜀.
  프레임은 다 쓴 뒤에만 이름이 붙고(.tmp → 바꿈), 같은 이름에 다른 작업 흔적이 있으면 멈춘다. FrameSeqVideo cleanup — 끝나면 지움.

AnchorPick — 후보 영상에서 번호로 앵커를 고르고 파일로 보관. lock=켜면 보관본을 쓰고 후보 영상 입력을
  아예 요청하지 않는다(지연 입력) → 후보 생성이 다시 돌지 않음. WAN 생성은 같은 시드여도 실행마다 달라지므로
  (실측 PSNR ~21dB) 캐시가 비워진 뒤 재생성하면 같은 번호가 다른 그림이 된다 — 고른 앵커를 고정하는 장치.

DepthInverseNormalize — 추정 depth(raw)를 VACE용 컨트롤로 변환.
  DA3 등의 기본 정규화는 ① 선형 depth라 하늘 같은 먼 영역이 범위를 먹어 인물·근경이 전부 흰색으로 뭉개지고
  ② 프레임마다 따로 정규화해 영상에서 밝기가 출렁인다.
  → 역수(1/z, 가까울수록 밝음)로 바꾸고 **클립 전체** 백분위 하나로 정규화. Blender 프리비즈 depth와 같은 규약.
"""
import math
import numpy as np
import torch


class DepthInverseNormalize:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "depth": ("IMAGE", {"tooltip": "raw depth (DA3Render normalization=raw). 클수록 멂."}),
            "low_pct": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 50.0, "step": 0.5,
                                  "tooltip": "역수 depth 하위 백분위 → 0(검정). 하늘·원경 기준"}),
            "high_pct": ("FLOAT", {"default": 99.5, "min": 50.0, "max": 100.0, "step": 0.1,
                                   "tooltip": "역수 depth 상위 백분위 → 1(흰색). 근경 포화 방지"}),
            "align_scale": ("BOOLEAN", {"default": False,
                                        "tooltip": "프레임별 중앙값으로 스케일 정렬 (단안 depth의 프레임별 스케일 흔들림 보정)"}),
            "gamma": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 3.0, "step": 0.05}),
        }, "optional": {
            "stats": ("DEPTH_STATS", {"tooltip": "연결하면 이 값으로 정규화 (클립 전체 기준, DepthStats) — 파트 단위 처리용"}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "endo/depth"

    def run(self, depth, low_pct, high_pct, align_scale, gamma, stats=None):
        d = depth[..., 0].float().clamp(min=1e-6)            # (B,H,W)
        inv = 1.0 / d
        if align_scale and inv.shape[0] > 1:
            med = inv.flatten(1).median(dim=1).values
            inv = inv * (med.median() / med.clamp(min=1e-9)).view(-1, 1, 1)
        if stats is not None:                                # 클립 전체 기준(파트 루프) — 백분위 대신 미리 구한 값
            lo, hi = torch.tensor(stats["lo"]), torch.tensor(stats["hi"])
        else:
            lo, hi = _inv_quantiles(inv, low_pct, high_pct)
        lo, hi = lo.to(inv.device), hi.to(inv.device)
        out = ((inv - lo) / (hi - lo).clamp(min=1e-9)).clamp(0.0, 1.0)
        if gamma != 1.0:
            out = out ** gamma
        return (out.unsqueeze(-1).repeat(1, 1, 1, 3).contiguous(),)


def _inv_quantiles(inv, low_pct, high_pct):
    flat = inv.flatten()
    if flat.numel() > 4_000_000:                             # torch.quantile 입력 한계 회피, 결정적 표본
        g = torch.Generator(device="cpu").manual_seed(0)
        flat = flat.cpu()[torch.randint(0, flat.numel(), (4_000_000,), generator=g)]
    return torch.quantile(flat.float(), low_pct / 100.0), torch.quantile(flat.float(), high_pct / 100.0)


class DepthNormalizeChained:
    """파트 루프용 역수 depth 정규화 — 첫 파트는 자기 백분위, 이후 파트는 겹치는 앞 K 프레임에서
    '이 파트 역수 depth → 앞 파트가 실제로 쓴 정규화 depth' 1차 변환(a·x+b)을 맞춰 전 프레임에 적용.
    파트마다 따로 백분위를 잡으면 생성 초안처럼 내용이 파트마다 다른 depth 는 경계에서 밝기 단계가 튄다(10-09 실측)."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "depth": ("IMAGE", {"tooltip": "raw depth (DA3Render normalization=raw), 앞 K 프레임 = 앞 파트와 겹치는 구간"}),
            "prev": ("IMAGE", {"tooltip": "앞 파트가 실제로 쓴 정규화 depth 의 끝 K 프레임"}),
            "first": ("BOOLEAN", {"default": True}),
            "overlap": ("INT", {"default": 5, "min": 1, "max": 64}),
            "low_pct": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 50.0, "step": 0.5}),
            "high_pct": ("FLOAT", {"default": 99.5, "min": 50.0, "max": 100.0, "step": 0.1}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("depth", "report")
    FUNCTION = "run"
    CATEGORY = "endo/depth"

    def run(self, depth, prev, first, overlap, low_pct, high_pct):
        inv = 1.0 / depth[..., 0].float().clamp(min=1e-6)
        lo, hi = _inv_quantiles(inv, low_pct, high_pct)
        a, b = 1.0 / float((hi - lo).clamp(min=1e-9)), -float(lo) / float((hi - lo).clamp(min=1e-9))
        rep = "첫 파트: 자기 백분위"
        if not first and prev is not None and inv.shape[0] >= overlap and prev.shape[0] >= overlap:
            x = inv[:overlap].flatten().double().cpu()
            y = prev[-overlap:, ..., 0].float().flatten().double().cpu()
            if y.numel() == x.numel():
                ok = (y > 0.02) & (y < 0.98)                  # 앞 파트에서 잘린(포화) 화소는 맞춤에서 제외
                if ok.sum() > 1000:
                    xs, ys = x[ok], y[ok]
                    xm, ym = xs.mean(), ys.mean()
                    va = ((xs - xm) ** 2).mean()
                    a2 = float(((xs - xm) * (ys - ym)).mean() / va) if va > 0 else 0.0
                    if a2 > 0:
                        a, b = a2, float(ym - a2 * xm)
                        fit = (a * xs + b - ys).abs().mean()
                        rep = f"앞 파트에 맞춤 · 겹침 평균오차 {float(fit):.3f}"
                    else: rep = "맞춤 실패(기울기≤0) → 자기 백분위"
                else: rep = "겹침 유효 화소 부족 → 자기 백분위"
        out = (inv * a + b).clamp(0.0, 1.0)
        return (out.unsqueeze(-1).repeat(1, 1, 1, 3).contiguous(), rep)


class DepthPadExtend:
    """depth 아웃페인트용 패딩 — 가장자리 픽셀을 바깥으로 복제(replicate).
    회색/블러 패딩은 검은 원경과의 경계 그라데이션을 모델이 액자·비네트로 읽는다(10-06 실측 2회) —
    복제 패딩은 경계 단차가 0이라 그 신호 자체가 없다."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "left": ("INT", {"default": 0, "min": 0, "max": 2048}),
            "top": ("INT", {"default": 0, "min": 0, "max": 2048}),
            "right": ("INT", {"default": 0, "min": 0, "max": 2048}),
            "bottom": ("INT", {"default": 0, "min": 0, "max": 2048}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "endo/depth"

    def run(self, image, left, top, right, bottom):
        import torch.nn.functional as F
        x = image.movedim(-1, 1)                           # B,C,H,W
        x = F.pad(x, (left, right, top, bottom), mode="replicate")
        return (x.movedim(1, -1).contiguous(),)


class DepthPadDirectional:
    """방향별 아웃페인트 패딩 — 상/하/좌/우 px 를 지정하면 원본 비율을 유지한 채(무왜곡 균일 축소)
    그 방향 비율대로 배치하고 가장자리 복제로 패딩. 캔버스 크기는 입력과 동일하게 유지.
    두 축의 요청 합이 캔버스 비율과 안 맞으면: 빡빡한 축 = 요청 px 정확히, 느슨한 축 = 요청 비율대로 '이상'."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "enable": ("BOOLEAN", {"default": True}),
            "top": ("INT", {"default": 0, "min": 0, "max": 2048}),
            "bottom": ("INT", {"default": 0, "min": 0, "max": 2048}),
            "left": ("INT", {"default": 0, "min": 0, "max": 2048}),
            "right": ("INT", {"default": 0, "min": 0, "max": 2048}),
        }, "optional": {
            "pad": ("PAD_CFG", {"tooltip": "연결하면 위 값 대신 이 설정 (DepthPadSettings)"}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "endo/depth"

    def run(self, image, enable, top, bottom, left, right, pad=None):
        import torch.nn.functional as F
        if pad is not None:
            enable, top, bottom, left, right = pad["enable"], pad["top"], pad["bottom"], pad["left"], pad["right"]
        if not enable or (top + bottom + left + right) == 0:
            return (image,)
        H, W = image.shape[1], image.shape[2]
        avail_w, avail_h = W - left - right, H - top - bottom
        if avail_w < 32 or avail_h < 32:
            raise ValueError(f"패딩이 너무 큼 — 남는 영역 {avail_w}x{avail_h} (최소 32)")
        s = min(avail_w / W, avail_h / H)                  # 균일 축소 = 비율 유지
        iw, ih = max(32, round(W * s)), max(32, round(H * s))
        x = F.interpolate(image.movedim(-1, 1), size=(ih, iw), mode="bilinear", align_corners=False, antialias=True)
        sx, sy = W - iw, H - ih                            # 실제 패딩 총량 (요청 이상일 수 있음)
        pl = round(sx * left / (left + right)) if (left + right) > 0 else sx // 2
        pt = round(sy * top / (top + bottom)) if (top + bottom) > 0 else sy // 2
        x = F.pad(x, (pl, sx - pl, pt, sy - pt), mode="replicate")
        return (x.movedim(1, -1).contiguous(),)


# ---------- 영상 로드 엔진: 구간 단위 디코딩 (긴 클립도 RAM 이 길이에 비례하지 않게) ----------
# 프레임별 실제 시각(pts)으로 목표 fps 격자를 채운다. 복원 끔 = 각 시각의 직전 실제 프레임(sample & hold),
# 복원 켬 = 중복 제거 → 앞뒤 실제 프레임 사이를 보간 모델로 정확한 비율 위치에 생성(공백·컷은 유지).
# 구간 로드는 '계획(축소본·시각만) → 필요한 프레임만 원본 크기로 다시 디코딩' 2단계라 원본 fps 와 무관하게 출력 수만큼만 메모리를 쓴다.
RETIME_DEFAULT = {"retime": False, "dup_threshold": 0.8, "gap_max_sec": 0.5, "cut_threshold": 40.0, "dup_center": False, "interp_model": None}


def _small(a):                                              # 차이 계산용 축소 흑백
    g = a[..., :3].astype(np.float32).mean(-1)
    h, w = g.shape; sy, sx = max(1, h // 72), max(1, w // 72)
    return np.ascontiguousarray(g[::sy, ::sx])              # 🪤 슬라이스 뷰는 원본 크기 g 전체를 붙잡음 → 1분 1080p60 이면 ~17GB


def _load_size(src, height):
    import av
    with av.open(src, mode="r") as c:
        cc = c.streams.video[0].codec_context
        return max(8, int(round(cc.width * height / cc.height / 8)) * 8), height


def _scan(src, height, lo, hi, smalls, want=None):
    """절대 시각 [lo, hi](hi=None 이면 끝까지) 프레임을 디코딩 → (시각들, pts들, 축소본들|None, {pts: 원본 크기 uint8}).
    smalls=True 면 각 프레임 축소 흑백(중복·컷 판정용), want 에 든 pts 만 원본 크기로 보관."""
    import av
    want = want or set()
    W, H = _load_size(src, height)

    def run(seek_t):
        times, pts, sm, full, first_t = [], [], [] if smalls else None, {}, None
        with av.open(src, mode="r") as c:
            st = c.streams.video[0]; st.thread_type = "AUTO"; tb = st.time_base
            if seek_t > 0:
                c.seek(int(seek_t / tb), stream=st, backward=True, any_frame=False)
            for fr in c.decode(st):
                if fr.pts is None: continue
                t = float(fr.pts * tb)
                if first_t is None: first_t = t
                if t < lo - 1e-6: continue
                if hi is not None and t > hi + 1e-6: break
                times.append(t); pts.append(fr.pts)
                if smalls or fr.pts in want:
                    a = fr.to_ndarray(width=W, height=H, format="rgb24")
                    if smalls: sm.append(_small(a))
                    if fr.pts in want: full[fr.pts] = a
        return times, pts, sm, full, first_t

    seek_t = lo - 1.0
    r = run(seek_t)
    if seek_t > 0 and (r[4] is None or r[4] > lo + 1e-6):  # 키프레임 색인이 부정확해 지나쳐 찾은 경우 → 처음부터
        r = run(0)
    return np.array(r[0]), r[1], r[2], r[3]


def _plan(times, sm, tgt, cfg):
    """목표 시각마다 (프레임 a, 프레임 b|None, 비율). b=None 이면 a 그대로. times·tgt 는 같은 기준의 초."""
    n = len(times)
    if not cfg["retime"]:
        idx = np.clip(np.searchsorted(times, tgt + 1e-6, side="right") - 1, 0, n - 1)
        return [(int(i), None, 0.0) for i in idx], 0, 0
    dif = np.array([np.abs(sm[i] - sm[i - 1]).mean() for i in range(1, n)]) if n > 1 else np.zeros(0)
    keep = [0] + [i for i in range(1, n) if not dif[i - 1] < cfg["dup_threshold"]]
    if cfg["dup_center"]:                                   # 중복 구간(같은 그림이 이어진 시간)의 가운데를 실제 시각으로
        ends = keep[1:] + [n]
        kt = np.array([(times[a] + times[b - 1]) / 2 for a, b in zip(keep, ends)])
    else:
        kt = times[keep]
    j = np.clip(np.searchsorted(kt, tgt + 1e-6, side="right") - 1, 0, len(keep) - 1)
    plan, ni, nh = [None] * len(tgt), 0, 0
    for i in np.unique(j):
        sel = np.where(j == i)[0]
        if i + 1 >= len(keep):
            for s_ in sel: plan[s_] = (keep[i], None, 0.0)
            continue
        gap = kt[i + 1] - kt[i]
        cut = np.abs(sm[keep[i + 1]] - sm[keep[i]]).mean() > cfg["cut_threshold"]
        for s_ in sel:
            al = (tgt[s_] - kt[i]) / gap
            if al >= 0.98: plan[s_] = (keep[i + 1], None, 0.0)
            elif al <= 0.02: plan[s_] = (keep[i], None, 0.0)
            elif gap > cfg["gap_max_sec"] or cut: plan[s_] = (keep[i], None, 0.0); nh += 1   # 공백·컷: 앞 프레임 유지
            else: plan[s_] = (keep[i], keep[i + 1], float(al)); ni += 1
    return plan, ni, nh


def _interp(model, a, b, ts):
    from comfy.ldm.common_dit import pad_to_patch_size
    dev, dt = model.load_device, model.model_dtype()
    im = model.model
    align = getattr(im, "pad_align", 1)
    def prep(x):
        t = torch.from_numpy(x).to(dev, dt).permute(2, 0, 1)[None] / 255.0
        return pad_to_patch_size(t, (align, align), padding_mode="reflect") if align > 1 else t
    A, B = prep(a), prep(b); H, W = a.shape[:2]
    with torch.no_grad():
        if hasattr(im, "forward_multi_timestep"):
            outs = im.forward_multi_timestep(A, B, list(ts))
        else:
            outs = torch.cat([im(A, B, timestep=torch.full((1, 1, A.shape[2], A.shape[3]), float(t), device=dev, dtype=dt)) for t in ts])
    return [(o[:, :H, :W].permute(1, 2, 0).float().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8) for o in outs]


def _render(src, height, times_abs, pts, plan, cfg):
    need = sorted({i for a, b, _ in plan for i in (a, b) if i is not None})
    _, _, _, full = _scan(src, height, times_abs[need[0]], times_abs[need[-1]], False, {pts[i] for i in need})
    if cfg["retime"] and any(b is not None for _, b, _ in plan):
        if cfg["interp_model"] is None: raise ValueError("복원 켬 — 보간 모델(FrameInterpolationModelLoader) 연결 필요")
        import comfy.model_management as mm
        mm.load_models_gpu([cfg["interp_model"]])
    H, W = full[pts[need[0]]].shape[:2]
    out = torch.empty((len(plan), H, W, 3), dtype=torch.float32)
    pairs = {}
    for k, (a, b, al) in enumerate(plan):
        if b is None: out[k] = torch.from_numpy(full[pts[a]]).float() / 255.0
        else: pairs.setdefault((a, b), []).append((k, al))
    for (a, b), items in pairs.items():
        for (k, _), r in zip(items, _interp(cfg["interp_model"], full[pts[a]], full[pts[b]], [al for _, al in items])):
            out[k] = torch.from_numpy(r).float() / 255.0
    return out


def _clip_range(start_sec, dur_sec):
    return start_sec, (start_sec + dur_sec if dur_sec > 0 else None)


def _clip_scan(src, height, start_sec, dur_sec, fps, cfg):
    """클립 전체 스캔(축소본만) → 시각·pts·축소본·목표 프레임 수·진단 문자열."""
    t0, t1 = _clip_range(start_sec, dur_sec)
    times, pts, sm, _ = _scan(src, height, t0, t1, True)
    if not len(times): raise ValueError("구간에 프레임이 없음")
    rel = times - t0; n = len(rel)
    iv = np.diff(rel) if n > 1 else np.zeros(0)
    dif = np.array([np.abs(sm[i] - sm[i - 1]).mean() for i in range(1, n)]) if n > 1 else np.zeros(0)
    dup = dif < cfg["dup_threshold"]
    span = rel[-1] - rel[0] if n > 1 else 0.0
    src_fps = (n - 1) / span if span > 0 else fps
    T = dur_sec if dur_sec > 0 else rel[-1] + (np.median(iv) if len(iv) else 1 / fps)
    N = len(np.arange(0, T - 1e-6, 1.0 / fps))
    rep = (f"원본 {n}f · 평균 {src_fps:.2f}fps · 간격 중앙 {np.median(iv) * 1000 if len(iv) else 0:.0f}ms "
           f"최대 {iv.max() * 1000 if len(iv) else 0:.0f}ms · 중복 {dup.mean() * 100 if len(dup) else 0:.0f}% ({int(dup.sum())}f) → "
           f"{N}f @ {fps:g}fps")
    return times, pts, sm, N, rep, float(src_fps)


def _load_targets(src, height, start_sec, dur_sec, tgt, cfg):
    """목표 시각(클립 시작 기준 초)들만 로드. 앞뒤 여유(복원 켬 = 공백 한도+0.25초)를 두고 그 창만 디코딩 —
    여유가 공백 한도보다 길어서 창 밖 프레임이 결과를 바꾸는 경우는 '공백 유지'로 같은 결과가 된다."""
    t0, t1 = _clip_range(start_sec, dur_sec)
    margin = (cfg["gap_max_sec"] + 0.25) if cfg["retime"] else 0.25
    lo, hi = max(t0, t0 + tgt.min() - margin), t0 + tgt.max() + margin
    if t1 is not None: hi = min(hi, t1)
    while True:
        times, pts, sm, _ = _scan(src, height, lo, hi, cfg["retime"])
        if lo <= t0 + 1e-9 or (len(times) and times[0] <= t0 + tgt.min() + 1e-6): break
        lo = max(t0, lo - 4 * margin - 1.0)                # 창 앞이 공백이라 직전 프레임이 없음 → 더 앞에서
    if not len(times): raise ValueError("구간에 프레임이 없음")
    plan, ni, nh = _plan(times - t0, sm, tgt, cfg)
    return _render(src, height, times, pts, plan, cfg), ni, nh


class VideoRetimeSettings:
    """원본 프레임 복원 설정 한 벌 — 클립 스캔·구간 로드 노드들이 같은 값을 쓰도록 묶음."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "retime": ("BOOLEAN", {"default": False, "label_on": "복원 (중복 제거 + 보간)", "label_off": "그대로 (직전 프레임)"}),
            "dup_threshold": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 20.0, "step": 0.1, "tooltip": "앞 프레임과 평균 차이(0~255)가 이보다 작으면 중복"}),
            "gap_max_sec": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 5.0, "step": 0.05, "tooltip": "이보다 긴 공백은 보간 안 함"}),
            "cut_threshold": ("FLOAT", {"default": 40.0, "min": 1.0, "max": 255.0, "step": 1.0, "tooltip": "두 프레임 평균 차이가 이보다 크면 장면 전환으로 보고 보간 안 함"}),
            "dup_time": (["처음 (캡처 장치·폰)", "가운데 (fps 변환된 영상)"], {"tooltip": "중복이 연달아 나올 때 실제 찍힌 시각을 어디로 볼지. 캡처 장치는 처음, 편집기·트랜스코더로 fps 를 올린 영상은 가운데"}),
        }, "optional": {
            "interp_model": ("INTERP_MODEL", {"tooltip": "복원 켤 때 필요 (FrameInterpolationModelLoader, FILM 권장)"}),
        }}

    RETURN_TYPES = ("RETIME_CFG",)
    RETURN_NAMES = ("settings",)
    FUNCTION = "run"
    CATEGORY = "endo/video"

    def run(self, retime, dup_threshold, gap_max_sec, cut_threshold, dup_time, interp_model=None):
        return ({"retime": bool(retime), "dup_threshold": dup_threshold, "gap_max_sec": gap_max_sec, "cut_threshold": cut_threshold,
                 "dup_center": dup_time.startswith("가운데"), "interp_model": interp_model},)


class VideoInfo:
    """영상의 원래 크기·fps·프레임 수·소리 유무 — 디코딩 없이 컨테이너 정보만."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video": ("VIDEO",)}}

    RETURN_TYPES = ("INT", "INT", "FLOAT", "INT", "BOOLEAN")
    RETURN_NAMES = ("width", "height", "fps", "frames", "has_audio")
    FUNCTION = "run"
    CATEGORY = "endo/video"

    def run(self, video):
        import av
        src = video.get_stream_source()
        if hasattr(src, "seek"): src.seek(0)
        with av.open(src, mode="r") as c:
            vs = c.streams.video[0]
            fps = float(vs.average_rate or vs.guessed_rate or 16)
            n = vs.frames or (int(round(float(vs.duration * vs.time_base) * fps)) if vs.duration else
                              int(round(c.duration / 1e6 * fps)) if c.duration else 0)
            return (vs.codec_context.width, vs.codec_context.height, fps, int(n), len(c.streams.audio) > 0)


class VideoClipInfo:
    """클립 전체를 축소본으로만 훑어 목표 프레임 수·로드 크기·진단을 낸다(원본 크기 프레임은 보관 안 함)."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video": ("VIDEO",),
            "settings": ("RETIME_CFG",),
            "fps": ("FLOAT", {"default": 16.0, "min": 1.0, "max": 120.0, "step": 0.001}),
            "height": ("INT", {"default": 832, "min": 64, "max": 4096, "step": 8, "tooltip": "비율 유지 로드 높이"}),
            "start_sec": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 36000.0, "step": 0.01}),
            "dur_sec": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 36000.0, "step": 0.01, "tooltip": "0 = 끝까지"}),
        }}

    RETURN_TYPES = ("INT", "INT", "INT", "STRING", "FLOAT")
    RETURN_NAMES = ("frames", "width", "height", "report", "source_fps")
    FUNCTION = "run"
    CATEGORY = "endo/video"

    def run(self, video, settings, fps, height, start_sec, dur_sec):
        src = video.get_stream_source()
        _, _, _, N, rep, src_fps = _clip_scan(src, height, start_sec, dur_sec, fps, settings)
        W, H = _load_size(src, height)
        rep += " · 복원 켬 (중복 제거 + 보간)" if settings["retime"] else " · 복원 꺼짐 (직전 프레임)"
        return (N, W, H, rep, src_fps)


class VideoLoadFrames:
    """클립의 출력 프레임 번호 [first, first+count) (step 간격)만 로드 — 파트 루프 안에서 그 파트만 디코딩.
    번호 k 의 시각 = k / fps (클립 시작 기준). 클립 끝을 넘는 번호는 마지막 프레임 유지."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "video": ("VIDEO",),
            "settings": ("RETIME_CFG",),
            "fps": ("FLOAT", {"default": 16.0, "min": 1.0, "max": 120.0, "step": 0.001}),
            "height": ("INT", {"default": 832, "min": 64, "max": 4096, "step": 8}),
            "start_sec": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 36000.0, "step": 0.01}),
            "dur_sec": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 36000.0, "step": 0.01}),
            "first_frame": ("INT", {"default": 0, "min": 0, "max": 1_000_000}),
            "frame_count": ("INT", {"default": 81, "min": 1, "max": 100_000}),
            "frame_step": ("INT", {"default": 1, "min": 1, "max": 10_000}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "endo/video"

    def run(self, video, settings, fps, height, start_sec, dur_sec, first_frame, frame_count, frame_step):
        tgt = (first_frame + np.arange(frame_count) * frame_step) / fps
        return (_load_targets(video.get_stream_source(), height, start_sec, dur_sec, tgt, settings)[0],)


class VideoLoadRetime:
    """영상 → 일정한 fps 프레임 (선택: 저fps·가변fps·중복 프레임 복원). 클립 전체를 한 번에 — 긴 클립은 VideoLoadFrames 로 구간씩.
    retime 끔 = 목표 시각마다 그 직전 실제 프레임. retime 켬 = 중복 제거 → 앞뒤 실제 프레임 사이 정확한 비율 위치 보간.
      공백이 gap_max 초보다 길거나 두 프레임 차이가 cut 보다 크면(장면 전환) 보간하지 않고 앞 프레임 유지."""
    @classmethod
    def INPUT_TYPES(cls):
        req = dict(VideoRetimeSettings.INPUT_TYPES()["required"])
        return {"required": {
            "video": ("VIDEO",),
            "fps": ("FLOAT", {"default": 16.0, "min": 1.0, "max": 120.0, "step": 0.001}),
            "height": ("INT", {"default": 832, "min": 64, "max": 4096, "step": 8, "tooltip": "비율 유지 로드 높이 (RAM 절약)"}),
            "start_sec": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 36000.0, "step": 0.01}),
            "dur_sec": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 36000.0, "step": 0.01, "tooltip": "0 = 끝까지"}),
            **req,
        }, "optional": VideoRetimeSettings.INPUT_TYPES()["optional"]}

    RETURN_TYPES = ("IMAGE", "STRING", "FLOAT")
    RETURN_NAMES = ("images", "report", "source_fps")
    FUNCTION = "run"
    CATEGORY = "endo/video"

    def run(self, video, fps, height, start_sec, dur_sec, retime, dup_threshold, gap_max_sec, cut_threshold, dup_time="처음 (캡처 장치·폰)", interp_model=None):
        cfg = VideoRetimeSettings().run(retime, dup_threshold, gap_max_sec, cut_threshold, dup_time, interp_model)[0]
        src = video.get_stream_source()
        times, pts, sm, N, rep, src_fps = _clip_scan(src, height, start_sec, dur_sec, fps, cfg)
        tgt = np.arange(N) / fps
        plan, ni, nh = _plan(times - start_sec, sm, tgt, cfg)
        imgs = _render(src, height, times, pts, plan, cfg)
        rep += f" · 보간 {ni}f · 공백/컷 유지 {nh}f" if retime else " · 복원 꺼짐(직전 프레임)"
        return (imgs, rep, src_fps)


class DepthStats:
    """raw depth 표본(클립 전체에서 고르게 뽑은 프레임) → 역수 depth 백분위(lo, hi). 파트마다 따로 정규화하면
    파트 경계에서 depth 밝기가 튀므로, 클립 전체 기준 하나를 미리 구해 모든 파트에 같은 값을 쓴다."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "depth": ("IMAGE", {"tooltip": "raw depth (DA3Render normalization=raw)"}),
            "low_pct": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 50.0, "step": 0.5}),
            "high_pct": ("FLOAT", {"default": 99.5, "min": 50.0, "max": 100.0, "step": 0.1}),
        }}

    RETURN_TYPES = ("DEPTH_STATS", "STRING")
    RETURN_NAMES = ("stats", "report")
    FUNCTION = "run"
    CATEGORY = "endo/depth"

    def run(self, depth, low_pct, high_pct):
        lo, hi = _inv_quantiles(1.0 / depth[..., 0].float().clamp(min=1e-6), low_pct, high_pct)
        return ({"lo": float(lo), "hi": float(hi)}, f"역수 depth {depth.shape[0]}f 기준 lo {float(lo):.4g} · hi {float(hi):.4g}")


SEQ_WHERE = ["temp", "output"]


def _seq_dir(name, where="temp"):
    """temp = ComfyUI 임시 폴더(재시작 때 비워짐) · output = 출력 폴더 아래 endo_seq/ (남음 — 이어서 하기용)."""
    import os, folder_paths
    safe = "".join(c for c in name if c.isalnum() or c in "-_") or "clip"
    base = folder_paths.get_output_directory() if where == "output" else folder_paths.get_temp_directory()
    return os.path.join(base, "endo_seq", safe)


def _seq_meta(d):
    import json, os
    try:
        with open(os.path.join(d, "meta.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _frames_present(d, lo, hi, sub="img"):
    import os
    return all(os.path.exists(os.path.join(d, sub, f"{k:06d}.png")) for k in range(lo, hi))


class FrameSeqCheck:
    """이어서 하기: 이 조각 [offset, offset+count) 프레임이 이미 디스크에 다 있나(클립 끝 total 에서 자름).
    resume 끔이면 늘 '없음'. 같은 이름 폴더에 프레임 수가 다른 작업 흔적이 있으면 멈춘다(설정을 바꾼 채 이어 붙이는 사고 방지)."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "name": ("STRING", {"default": "clip"}),
            "where": (SEQ_WHERE, {"default": "output"}),
            "offset": ("INT", {"default": 0, "min": 0, "max": 1_000_000}),
            "count": ("INT", {"default": 1, "min": 1, "max": 1_000_000}),
            "total": ("INT", {"default": 0, "min": 0, "max": 1_000_000}),
            "resume": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("BOOLEAN", "STRING")
    RETURN_NAMES = ("done", "seq_dir")
    FUNCTION = "run"
    CATEGORY = "endo/video"

    @classmethod
    def IS_CHANGED(cls, **kw):
        return float("nan")                     # 디스크 상태를 묻는 노드 — 캐시하면 안 됨

    def run(self, name, where, offset, count, total, resume):
        d = _seq_dir(name, where)
        if not resume:
            return (False, d)
        meta = _seq_meta(d)
        if meta is not None and total and meta.get("total") != total:
            raise ValueError(f"'{name}' 작업 폴더에 다른 작업 흔적이 있습니다(프레임 수 {meta.get('total')} ≠ 이번 {total}) — "
                             f"설정·영상을 바꿨다면 '이어서 하기' 를 끄거나 출력 이름을 바꾸세요: {d}")
        hi = min(offset + count, total) if total else offset + count
        return (meta is not None and _frames_present(d, offset, hi), d)


class FrameSeqWrite:
    """파트 결과를 디스크에 프레임 파일로 — 결과를 RAM 에 모으지 않으므로 클립 길이와 무관. reset(첫 파트)이면 비우고 시작."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "name": ("STRING", {"default": "clip"}),
            "offset": ("INT", {"default": 0, "min": 0, "max": 1_000_000, "tooltip": "이 조각 첫 프레임의 클립 내 번호"}),
            "total": ("INT", {"default": 0, "min": 0, "max": 1_000_000, "tooltip": "클립 프레임 수 — 넘는 프레임은 버림"}),
            "reset": ("BOOLEAN", {"default": False}),
        }, "optional": {
            "depth": ("IMAGE", {"tooltip": "같은 번호로 depth 도 저장 (미리보기용)"}),
            "where": (SEQ_WHERE, {"default": "temp", "tooltip": "temp = 임시(재시작 때 비워짐) · output = 남김(이어서 하기)"}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("seq_dir",)
    FUNCTION = "run"
    CATEGORY = "endo/video"

    def run(self, images, name, offset, total, reset, depth=None, where="temp"):
        import json, os, shutil
        from PIL import Image
        d = _seq_dir(name, where)
        if reset and os.path.isdir(d): shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
        meta = _seq_meta(d)
        if meta is None:
            with open(os.path.join(d, "meta.json"), "w") as f:
                json.dump({"total": total, "width": int(images.shape[2]), "height": int(images.shape[1])}, f)
        elif (meta.get("total"), meta.get("width"), meta.get("height")) != (total, int(images.shape[2]), int(images.shape[1])):
            raise ValueError(f"'{name}' 작업 폴더의 프레임({meta.get('width')}×{meta.get('height')}, {meta.get('total')}장)과 이번 조각"
                             f"({images.shape[2]}×{images.shape[1]}, {total}장)이 다릅니다 — 처음부터 하려면 '이어서 하기' 를 끄세요: {d}")
        for sub, t in (("img", images), ("depth", depth)):
            if t is None: continue
            os.makedirs(os.path.join(d, sub), exist_ok=True)
            for k in range(t.shape[0]):
                if total and offset + k >= total: break
                a = (t[k].cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)
                path = os.path.join(d, sub, f"{offset + k:06d}.png")
                Image.fromarray(a).save(path + ".tmp", format="PNG", compress_level=1)
                os.replace(path + ".tmp", path)          # 다 쓴 뒤에만 이름이 붙음 — 끊기면 반쪽 파일이 안 남는다
        return (d,)


def _encode_mp4(frames, n, fps, path):
    """frames(k) → uint8 HxWx3. libx264 crf19 yuv420p (VHS h264-mp4 기본과 같은 설정)."""
    import av
    from fractions import Fraction
    with av.open(path, mode="w") as c:
        s = c.add_stream("libx264", rate=Fraction(fps).limit_denominator(1001))
        s.pix_fmt = "yuv420p"; s.options = {"crf": "19"}
        for k in range(n):
            a = frames(k)
            if k == 0: s.height, s.width = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
            vf = av.VideoFrame.from_ndarray(np.ascontiguousarray(a[:s.height, :s.width]), format="rgb24")
            for p in s.encode(vf): c.mux(p)
        for p in s.encode(): c.mux(p)


def _mux_audio(path, src, start=0.0):
    """path(영상만) 에 src 영상의 첫 소리 트랙을 그대로(재인코딩 없이) 붙인다. start 초부터, 영상 길이를 넘는 소리는 버림.
    소리가 없으면 아무것도 안 하고 False."""
    import av, heapq, os
    if hasattr(src, "seek"): src.seek(0)
    with av.open(src, mode="r") as a:
        if not a.streams.audio: return False
    if hasattr(src, "seek"): src.seek(0)
    tmp = path + ".noaudio.mp4"
    os.replace(path, tmp)
    with av.open(tmp, mode="r") as v, av.open(src, mode="r") as a, av.open(path, mode="w") as out:
        vi, ai = v.streams.video[0], a.streams.audio[0]
        vo, ao = out.add_stream_from_template(vi), out.add_stream_from_template(ai)
        end = float(vi.duration * vi.time_base) if vi.duration else (v.duration / 1e6 if v.duration else None)
        shift = int(round(start / ai.time_base)) if start else 0
        if start:
            a.seek(int(start * 1e6), any_frame=False, backward=True)   # av.time_base(µs) — 그 앞 키 패킷부터
        def packets(c, st, o):
            for p in c.demux(st):
                if p.dts is None: continue
                if o is ao and shift:
                    if p.pts is not None and p.pts < shift: continue
                    p.pts -= shift; p.dts -= shift
                t = float(p.dts * st.time_base)
                if o is ao and end is not None and p.pts is not None and float(p.pts * st.time_base) >= end: return
                yield t, id(o), p, o
        for _, _, p, o in heapq.merge(packets(v, vi, vo), packets(a, ai, ao), key=lambda x: (x[0], x[1])):
            p.stream = o
            out.mux(p)
    os.remove(tmp)
    return True


class FrameSeqVideo:
    """영상 저장 + 미리보기. lock 켬 = 파트 루프가 디스크에 쓴 프레임(seq_dir)을 이어 인코딩 / lock 끔 = images(앵커 후보 등).
    지연 입력이라 선택 안 된 쪽은 실행 자체가 안 됨 → lock 끔이면 본 루프가, 켬이면 후보 생성이 돌지 않는다.
    audio_from 을 이으면 그 영상의 소리 트랙을 재인코딩 없이 붙인다(영상 길이까지)."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "lock": ("BOOLEAN", {"default": False}),
            "fps": ("FLOAT", {"default": 16.0, "min": 1.0, "max": 120.0, "step": 0.001}),
            "total": ("INT", {"default": 0, "min": 0, "max": 1_000_000, "tooltip": "lock 켬: 이어붙일 프레임 수"}),
            "filename_prefix": ("STRING", {"default": "clip/vace"}),
            "filename_prefix_unlocked": ("STRING", {"default": "clip/anchor_try"}),
            "kind": (["img", "depth"],),
            "save_output": ("BOOLEAN", {"default": True}),
        }, "optional": {
            "images": ("IMAGE", {"lazy": True}),
            "seq_dir": ("STRING", {"lazy": True, "forceInput": True}),
            "audio_from": ("VIDEO", {"tooltip": "이 영상의 소리를 그대로 붙임 (없으면 소리 없이)"}),
            "audio_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 36000.0, "step": 0.01, "tooltip": "소리를 원본의 몇 초부터"}),
            "cleanup": ("BOOLEAN", {"default": False, "tooltip": "lock 켬: 영상을 다 만든 뒤 프레임 폴더(seq_dir)를 지움"}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "endo/video"

    def check_lazy_status(self, lock, fps, total, filename_prefix, filename_prefix_unlocked, kind, save_output, images=None, seq_dir=None,
                          audio_from=None, audio_start=0.0, cleanup=False):
        return ["seq_dir"] if lock else ["images"]

    def run(self, lock, fps, total, filename_prefix, filename_prefix_unlocked, kind, save_output, images=None, seq_dir=None,
            audio_from=None, audio_start=0.0, cleanup=False):
        import os, folder_paths
        from PIL import Image
        if lock:
            d = os.path.join(seq_dir, kind)
            missing = [k for k in range(total) if not os.path.exists(os.path.join(d, f"{k:06d}.png"))]
            if missing: raise ValueError(f"프레임 누락 {len(missing)}개 (처음 {missing[:5]}) — {d}")
            n, frames, prefix = total, (lambda k: np.asarray(Image.open(os.path.join(d, f"{k:06d}.png")).convert("RGB"))), filename_prefix
            h, w = frames(0).shape[:2]
        else:
            n, prefix = images.shape[0], filename_prefix_unlocked
            frames = lambda k: (images[k].cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)
            h, w = images.shape[1], images.shape[2]
        base = folder_paths.get_output_directory() if save_output else folder_paths.get_temp_directory()
        out_dir, fname, counter, sub, _ = folder_paths.get_save_image_path(prefix, base, w, h)
        file = f"{fname}_{counter:05}.mp4"
        path = os.path.join(out_dir, file)
        _encode_mp4(frames, n, fps, path)
        if audio_from is not None:
            _mux_audio(path, audio_from.get_stream_source(), audio_start)
        if lock and cleanup:
            import shutil
            shutil.rmtree(seq_dir, ignore_errors=True)
        return {"ui": {"images": [{"filename": file, "subfolder": sub, "type": "output" if save_output else "temp"}], "animated": (True,)},
                "result": (path,)}


class DepthPadSettings:
    """테두리 연장 설정 한 벌 — 같은 값을 앵커 후보·파트 루프의 패딩 노드가 함께 쓰도록."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "enable": ("BOOLEAN", {"default": True}),
            "top": ("INT", {"default": 0, "min": 0, "max": 2048}),
            "bottom": ("INT", {"default": 0, "min": 0, "max": 2048}),
            "left": ("INT", {"default": 0, "min": 0, "max": 2048}),
            "right": ("INT", {"default": 0, "min": 0, "max": 2048}),
        }}

    RETURN_TYPES = ("PAD_CFG",)
    RETURN_NAMES = ("pad",)
    FUNCTION = "run"
    CATEGORY = "endo/depth"

    def run(self, enable, top, bottom, left, right):
        return ({"enable": bool(enable), "top": top, "bottom": bottom, "left": left, "right": right},)


class FrameContactSheet:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "columns": ("INT", {"default": 9, "min": 1, "max": 32}),
            "thumb_width": ("INT", {"default": 160, "min": 48, "max": 1024, "step": 8}),
        }}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "endo/video"

    def run(self, images, columns, thumb_width):
        from PIL import Image, ImageDraw, ImageFont
        b, h, w, _ = images.shape
        tw, th = thumb_width, max(1, round(h * thumb_width / w))
        rows = math.ceil(b / columns)
        sheet = Image.new("RGB", (columns * tw, rows * th), (255, 255, 255))
        size = max(12, tw // 5)
        try:
            font = ImageFont.truetype("arial.ttf", size)
        except OSError:
            font = ImageFont.load_default(size=size)
        draw = ImageDraw.Draw(sheet)
        for i in range(b):
            im = Image.fromarray((images[i].cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8))
            x, y = (i % columns) * tw, (i // columns) * th
            sheet.paste(im.resize((tw, th), Image.LANCZOS), (x, y))
            box = draw.textbbox((x + 4, y + 2), str(i), font=font)
            draw.rectangle((box[0] - 3, box[1] - 2, box[2] + 3, box[3] + 2), fill=(0, 0, 0))
            draw.text((x + 4, y + 2), str(i), fill=(255, 255, 0), font=font)
        return (torch.from_numpy(np.asarray(sheet, dtype=np.float32) / 255.0)[None],)


class AnchorPick:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "frame": ("INT", {"default": 60, "min": 0, "max": 4095, "tooltip": "후보 영상 프레임 번호 (접촉 시트 숫자)"}),
            "lock": ("BOOLEAN", {"default": False, "label_on": "고정 (보관본 사용·후보 생성 안 함)", "label_off": "고르기 (후보에서 뽑아 보관)"}),
            "slot": ("STRING", {"default": "clip_anchor", "tooltip": "보관 이름 (output/anchors/<slot>.png)"}),
        }, "optional": {
            "images": ("IMAGE", {"lazy": True, "tooltip": "후보 영상 (lock 이면 요청하지 않음)"}),
        }}

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("anchor", "candidates")
    FUNCTION = "run"
    CATEGORY = "endo/video"

    @staticmethod
    def _path(slot, kind):
        import os, folder_paths
        d = os.path.join(folder_paths.get_output_directory(), "anchors")
        os.makedirs(d, exist_ok=True)
        safe = "".join(c for c in slot if c.isalnum() or c in "-_") or "anchor"
        return os.path.join(d, f"{safe}{kind}.png")

    @classmethod
    def _locked_ok(cls, lock, slot):
        import os
        return bool(lock) and os.path.exists(cls._path(slot, "")) and os.path.exists(cls._path(slot, "_sheet"))

    def check_lazy_status(self, frame, lock, slot, images=None):
        return [] if self._locked_ok(lock, slot) else ["images"]

    @classmethod
    def IS_CHANGED(cls, frame, lock, slot, images=None):
        import os
        if cls._locked_ok(lock, slot):                     # 보관본이 바뀌면 다시 읽기
            return f"locked:{os.path.getmtime(cls._path(slot, ''))}"
        return f"pick:{frame}:{slot}"

    def run(self, frame, lock, slot, images=None):
        from PIL import Image
        load = lambda p: torch.from_numpy(np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0)[None]
        if self._locked_ok(lock, slot):
            return (load(self._path(slot, "")), load(self._path(slot, "_sheet")))
        if images is None:
            raise ValueError("후보 영상이 없음 — lock 을 끄고 실행하세요 (보관본도 없음)")
        i = max(0, min(frame, images.shape[0] - 1))
        anchor = images[i:i + 1]
        save = lambda t, p: Image.fromarray((t[0].cpu().float().numpy() * 255).clip(0, 255).astype(np.uint8)).save(p)
        save(anchor, self._path(slot, ""))
        sheet = FrameContactSheet().run(images, 9, 160)[0]
        save(sheet, self._path(slot, "_sheet"))
        return (anchor, sheet)


class WanLoraPair:
    """WAN 2.2 LoRA 한 쌍 — HIGH 모델엔 HIGH 파일, LOW 모델엔 LOW 파일을 각자 강도로. WAN 2.2 LoRA 는 대부분
    HIGH/LOW 두 파일이라 한 줄에 같이 둔다. 파일이 None 이거나 강도 0 이면 그쪽은 건드리지 않음(파일도 안 읽음).
    텍스트 인코더 쪽 강도는 없다 — WAN LoRA 는 거의 umt5 가중치가 없고, ② 같은 단계엔 인코더가 아예 없다."""
    NONE = "None"

    def __init__(self):
        import nodes
        self._h, self._l = nodes.LoraLoaderModelOnly(), nodes.LoraLoaderModelOnly()   # 파일 캐시가 칸마다 따로

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths
        names = [cls.NONE] + folder_paths.get_filename_list("loras")
        strength = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05})
        return {"required": {"model_high": ("MODEL",), "model_low": ("MODEL",),
                             "lora_high": (names, {"tooltip": "HIGH(노이즈 큰 앞 단계) 모델에 걸 파일. None = 안 씀"}),
                             "strength_high": strength,
                             "lora_low": (names, {"tooltip": "LOW(마무리 단계) 모델에 걸 파일. None = 안 씀"}),
                             "strength_low": strength}}

    RETURN_TYPES = ("MODEL", "MODEL")
    RETURN_NAMES = ("model_high", "model_low")
    FUNCTION = "run"
    CATEGORY = "endo/wan"

    def _one(self, loader, model, name, strength):
        if name == self.NONE or not name or strength == 0:
            return model
        return loader.load_lora_model_only(model, name, strength)[0]

    def run(self, model_high, model_low, lora_high, strength_high, lora_low, strength_low):
        return (self._one(self._h, model_high, lora_high, strength_high), self._one(self._l, model_low, lora_low, strength_low))


NODE_CLASS_MAPPINGS = {"DepthInverseNormalize": DepthInverseNormalize, "WanLoraPair": WanLoraPair, "FrameContactSheet": FrameContactSheet,
                       "AnchorPick": AnchorPick, "DepthPadExtend": DepthPadExtend,
                       "DepthPadDirectional": DepthPadDirectional, "VideoLoadRetime": VideoLoadRetime,
                       "VideoRetimeSettings": VideoRetimeSettings, "VideoClipInfo": VideoClipInfo, "VideoInfo": VideoInfo, "FrameSeqCheck": FrameSeqCheck, "VideoLoadFrames": VideoLoadFrames,
                       "DepthStats": DepthStats, "FrameSeqWrite": FrameSeqWrite, "FrameSeqVideo": FrameSeqVideo,
                       "DepthPadSettings": DepthPadSettings, "DepthNormalizeChained": DepthNormalizeChained}
NODE_DISPLAY_NAME_MAPPINGS = {"DepthInverseNormalize": "Depth → 역수 정규화 (클립 전체, VACE용)",
                              "WanLoraPair": "WAN LoRA 쌍 (HIGH·LOW 파일 + 강도 각각)",
                              "FrameContactSheet": "프레임 접촉 시트 (번호 표시)",
                              "AnchorPick": "앵커 고르기·고정 (번호 → 보관, lock 이면 후보 생성 생략)",
                              "DepthPadExtend": "Depth 패딩 (가장자리 복제, 아웃페인트용)",
                              "DepthPadDirectional": "Depth 방향별 패딩 (상/하/좌/우 px · 비율 유지)",
                              "VideoLoadRetime": "영상 → 일정 fps (저fps·가변fps·중복 복원)",
                              "VideoRetimeSettings": "원본 프레임 복원 설정",
                              "VideoClipInfo": "클립 스캔 (프레임 수·로드 크기·진단)",
                              "VideoInfo": "영상 정보 (원래 크기·fps·프레임 수·소리 유무)",
                              "FrameSeqCheck": "이어서 하기 — 이 조각 프레임이 이미 있나",
                              "VideoLoadFrames": "클립 구간 로드 (프레임 번호 범위 — 파트 루프용)",
                              "DepthStats": "Depth 정규화 기준 (클립 전체 표본)",
                              "FrameSeqWrite": "프레임 파일로 쓰기 (파트 결과, RAM 누적 없음)",
                              "FrameSeqVideo": "영상 저장 (lock 켬 = 프레임 파일 이어붙임 / 끔 = 입력 이미지)",
                              "DepthPadSettings": "테두리 연장 설정 (상/하/좌/우 px)",
                              "DepthNormalizeChained": "Depth 역수 정규화 (파트 이어 맞춤)"}
