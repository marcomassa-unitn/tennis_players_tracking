"""
motionEstimation/block_matching.py

Classical block-matching motion estimation, implemented from scratch (no
cv2 motion functions), as covered in the course labs.

Two search strategies over a SAD (sum of absolute differences) criterion:
  - full  : exhaustive full search over a (2R+1) x (2R+1) window
  - tss   : three-step search (logarithmic, ~25 SAD evaluations per block)

For every processed frame pair the script:
  - estimates a motion-vector field on non-overlapping blocks,
  - builds the motion-compensated prediction of the current frame and
    reports its PSNR vs the "no motion" prediction (plain previous frame),
  - saves a visualization (vectors of moving blocks drawn on the frame).

Motion-vector convention: (u, v) maps the previous frame to the current
one, i.e.  cur(x, y) ~ prev(x - u, y - v).

Usage (from project root):
    python motionEstimation/block_matching.py
    python motionEstimation/block_matching.py --method full --frames 3
    python motionEstimation/block_matching.py --video data/Input_video2.mp4 \\
        --method tss --block 16 --range 12 --scale 0.5 --start 40 \\
        --frames 5 --step 2 --output outputs/motion_estimation
"""

import argparse
import os
import time

import cv2
import numpy as np


# ── SAD helpers ────────────────────────────────────────────────────────────────

def _block_sad_field(prev_pad, cur, dx, dy, block, search):
    """
    SAD between every non-overlapping block of `cur` and the block of the
    (padded) previous frame displaced by (dx, dy).
    Returns an (n_blocks_y, n_blocks_x) float32 array.
    """
    H, W = cur.shape
    shifted = prev_pad[search + dy: search + dy + H,
                       search + dx: search + dx + W]
    diff = np.abs(cur.astype(np.float32) - shifted.astype(np.float32))
    nby, nbx = H // block, W // block
    diff = diff[: nby * block, : nbx * block]
    return diff.reshape(nby, block, nbx, block).sum(axis=(1, 3))


def full_search(prev, cur, block=16, search=8):
    """
    Exhaustive full-search block matching.

    Vectorized over blocks: for each of the (2R+1)^2 candidate displacements
    one whole-frame absolute difference + per-block reduction is computed,
    and the per-block argmin over displacements is tracked.

    Returns (u, v): two (n_blocks_y, n_blocks_x) int arrays, motion in pixels.
    """
    prev_pad = cv2.copyMakeBorder(prev, search, search, search, search,
                                  cv2.BORDER_REPLICATE)
    H, W = cur.shape
    nby, nbx = H // block, W // block

    best_sad = np.full((nby, nbx), np.inf, dtype=np.float32)
    u = np.zeros((nby, nbx), dtype=np.int32)
    v = np.zeros((nby, nbx), dtype=np.int32)

    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            sad = _block_sad_field(prev_pad, cur, dx, dy, block, search)
            better = sad < best_sad
            best_sad[better] = sad[better]
            # cur(x) ~ prev(x + d)  →  the content moved by -d
            u[better] = -dx
            v[better] = -dy
    return u, v


def _sad_one(prev_pad, cur_block, bx, by, dx, dy, block, search):
    """SAD of a single block of `cur` vs prev displaced by (dx, dy)."""
    y0 = by * block + search + dy
    x0 = bx * block + search + dx
    ref = prev_pad[y0: y0 + block, x0: x0 + block]
    return float(np.abs(cur_block.astype(np.float32)
                        - ref.astype(np.float32)).sum())


def three_step_search(prev, cur, block=16, search=8):
    """
    Three-step (logarithmic) search: starts with step = ceil(R/2), evaluates
    the 8 neighbours + centre, recentres on the best and halves the step.
    ~25 SAD evaluations per block instead of (2R+1)^2.

    Returns (u, v) like full_search.
    """
    prev_pad = cv2.copyMakeBorder(prev, search, search, search, search,
                                  cv2.BORDER_REPLICATE)
    H, W = cur.shape
    nby, nbx = H // block, W // block

    u = np.zeros((nby, nbx), dtype=np.int32)
    v = np.zeros((nby, nbx), dtype=np.int32)

    offsets = [(-1, -1), (0, -1), (1, -1),
               (-1, 0),  (0, 0),  (1, 0),
               (-1, 1),  (0, 1),  (1, 1)]

    for by in range(nby):
        for bx in range(nbx):
            cur_block = cur[by * block: (by + 1) * block,
                            bx * block: (bx + 1) * block]
            cx = cy = 0                      # current search centre
            best = _sad_one(prev_pad, cur_block, bx, by, 0, 0, block, search)
            step = max(1, int(np.ceil(search / 2)))
            while step >= 1:
                best_off = (0, 0)
                for ox, oy in offsets:
                    dx, dy = cx + ox * step, cy + oy * step
                    if abs(dx) > search or abs(dy) > search or (ox, oy) == (0, 0):
                        continue
                    sad = _sad_one(prev_pad, cur_block, bx, by, dx, dy,
                                   block, search)
                    if sad < best:
                        best = sad
                        best_off = (ox, oy)
                cx += best_off[0] * step
                cy += best_off[1] * step
                step //= 2
            u[by, bx] = -cx
            v[by, bx] = -cy
    return u, v


# ── motion compensation & PSNR ─────────────────────────────────────────────────

def motion_compensate(prev, u, v, block):
    """Predict the current frame moving each block of `prev` by its MV."""
    H, W = prev.shape
    pred = prev.copy()
    nby, nbx = u.shape
    for by in range(nby):
        for bx in range(nbx):
            y0, x0 = by * block, bx * block
            # cur(x) ~ prev(x - u): source block sits at destination - (u, v)
            sy = np.clip(y0 - v[by, bx], 0, H - block)
            sx = np.clip(x0 - u[by, bx], 0, W - block)
            pred[y0: y0 + block, x0: x0 + block] = \
                prev[sy: sy + block, sx: sx + block]
    return pred


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0 ** 2 / mse)


# ── visualization ──────────────────────────────────────────────────────────────

def draw_motion_field(frame, u, v, block, min_mag=1.0):
    """Draw MV arrows (green) on moving blocks; static blocks are skipped."""
    vis = frame.copy()
    nby, nbx = u.shape
    for by in range(nby):
        for bx in range(nbx):
            uu, vv = int(u[by, bx]), int(v[by, bx])
            if np.hypot(uu, vv) < min_mag:
                continue
            cx = bx * block + block // 2
            cy = by * block + block // 2
            cv2.arrowedLine(vis, (cx - uu, cy - vv), (cx, cy),
                            (0, 255, 0), 1, tipLength=0.35)
            cv2.circle(vis, (cx, cy), 1, (0, 0, 255), -1)
    return vis


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Block-matching motion estimation (full search / TSS)")
    parser.add_argument("--video",  default="data/Input_video2.mp4")
    parser.add_argument("--method", choices=["tss", "full", "both"],
                        default="tss",
                        help="Search strategy (both = compare on each pair)")
    parser.add_argument("--block",  type=int, default=16,
                        help="Block size in pixels (default: 16)")
    parser.add_argument("--range",  type=int, default=12, dest="search",
                        help="Search range R in pixels (default: 12)")
    parser.add_argument("--scale",  type=float, default=0.5,
                        help="Frame downscale factor before matching "
                             "(default: 0.5)")
    parser.add_argument("--start",  type=int, default=40,
                        help="First frame index of the first pair (default: 40)")
    parser.add_argument("--frames", type=int, default=5,
                        help="Number of frame pairs to process (default: 5)")
    parser.add_argument("--step",   type=int, default=2,
                        help="Distance between processed pairs (default: 2)")
    parser.add_argument("--output", default="outputs/motion_estimation")
    parser.add_argument("--display", action="store_true",
                        help="Also show results in a window")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    methods = ["tss", "full"] if args.method == "both" else [args.method]
    runner = {"tss": three_step_search, "full": full_search}

    print(f"Block matching  ({args.video})")
    print(f"  block={args.block}px  range=±{args.search}px  "
          f"scale={args.scale}  pairs={args.frames}")
    header = f"{'pair':>11s} | {'method':>6s} | {'time':>7s} | " \
             f"{'PSNR comp.':>10s} | {'PSNR no-MC':>10s} | {'moving blocks':>13s}"
    print(header)
    print("-" * len(header))

    for k in range(args.frames):
        i = args.start + k * args.step
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok1, f1 = cap.read()
        ok2, f2 = cap.read()
        if not (ok1 and ok2):
            print(f"  frame {i}: out of video, stopping.")
            break

        if args.scale != 1.0:
            f1 = cv2.resize(f1, None, fx=args.scale, fy=args.scale)
            f2 = cv2.resize(f2, None, fx=args.scale, fy=args.scale)
        g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

        for method in methods:
            t0 = time.perf_counter()
            u, v = runner[method](g1, g2, args.block, args.search)
            dt = time.perf_counter() - t0

            pred = motion_compensate(g1, u, v, args.block)
            h_crop = (g2.shape[0] // args.block) * args.block
            w_crop = (g2.shape[1] // args.block) * args.block
            p_mc = psnr(g2[:h_crop, :w_crop], pred[:h_crop, :w_crop])
            p_no = psnr(g2[:h_crop, :w_crop], g1[:h_crop, :w_crop])
            moving = int(np.sum(np.hypot(u, v) >= 1.0))

            print(f"{i:>5d}-{i + 1:<5d} | {method:>6s} | {dt:6.2f}s | "
                  f"{p_mc:9.2f}dB | {p_no:9.2f}dB | {moving:>13d}")

            vis = draw_motion_field(f2, u, v, args.block)
            out_png = os.path.join(
                args.output, f"bm_{method}_frame{i:05d}.png")
            cv2.imwrite(out_png, vis)
            if args.display:
                cv2.imshow("Block matching", vis)
                cv2.waitKey(300)

    cap.release()
    if args.display:
        cv2.destroyAllWindows()
    print(f"\nVisualizations saved in: {args.output}")


if __name__ == "__main__":
    main()
