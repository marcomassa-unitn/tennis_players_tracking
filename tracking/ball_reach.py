"""Can the ball get from A to B? The physical check of a jump between two positions.

One camera does not see how high the ball is: a point of the image is anywhere on
a ray, from the ground up. A speed limit alone therefore stops nothing (the jump
of Input_video2 from the top of a lob, frame 407, onto the near player's head,
frame 411, needs only 10 m/s if the ball is taken 8 m up at the start). What does
pin the ball down is where it was going, or the racket that just struck it. A
move from A (frame t) to B (frame t + gap) is allowed if:

  (a) it carries on the motion the ball had at A: B is near A + velocity * gap,
      within a margin growing with the gap (gravity, drag, detection noise);
  (b) it is at most MAX_BALL_SPEED_MS along the ground, the points taken where
      they are seen (right for a ball near the ground), and keeps the way the
      ball was going over the ground: a bounce turns it upwards, never aside;
  (c) A is on a player, and some point of the racket above the player's feet
      (RACKET_HEIGHTS_M) reaches some height of B at MAX_BALL_SPEED_MS: a hit,
      after which the ball may go anywhere it can reach, a lob as well;
  (d) with no motion known at A (a path starts there), some heights of A and B
      put them within MAX_BALL_SPEED_MS: only the impossible is refused.

On the 22 real-time clips (about 8800 moves of the final tracks) (a)-(c) refused
both jumps of Input_video2 and about 25 others, most of them tracker errors (a
ball kid, a jump from one baseline to the other net in 4 frames).

Heights use the scale of the ground below the ball for the vertical too, as the
flight heights of the shot analysis do.
"""

import numpy as np

from utils.court_geometry import L_m, W_m

MAX_BALL_SPEED_MS = 240 / 3.6      # the fastest serves; nothing in play goes faster
CONTINUITY_PX = 40.0               # (a) margin at 1080p, plus...
CONTINUITY_GROWTH_PX = 2.0         # ...this times gap^2: the longer unseen, the wider
                                   # (gravity is at most ~0.5 px/frame^2 of drift)
BOUNCE_MAX_TURN_COS = 0.5          # (b) at most 60 degrees off its way over the ground
RACKET_HEIGHTS_M = (0.3, 3.5)      # (c) the racket, from a low volley to a smash
RACKET_SLACK_M = 1.5               # ...reaching out from the feet, and the box error
PLAYER_SIDE_MARGIN = 0.6           # (c) the player's box grown by this on the sides (racket and arm)
PLAYER_TOP_MARGIN = 0.6            # ...and this above: the toss and the smash are over it
BALL_HEIGHTS_M = np.linspace(0.0, 12.0, 25)   # where a seen ball can be, up to a high lob
PLAY_MARGIN_M = 8.0                # ...over the court and this far around it


class BallReach:
    """The check above, for one camera (court homography) and frame rate."""

    def __init__(self, conv, fps, frame_height=1080):
        self.conv, self.fps = conv, fps
        self.s = frame_height / 1080
        self.m_to_px = np.linalg.inv(conv._H)
        self.racket = np.linspace(*RACKET_HEIGHTS_M, 8)

    def ground(self, x, y):
        """Ground point (m) seen at pixel (x, y)."""
        return np.asarray(self.conv.to_meters(float(x), float(y)), float)

    def _px_per_metre(self, g):
        """Image scale (px per metre) at ground points g, shape (n, 2)."""
        ends = np.concatenate([g - [0.5, 0.0], g + [0.5, 0.0]])
        ends = np.c_[ends, np.ones(len(ends))] @ self.m_to_px.T
        ends = ends[:, :2] / ends[:, 2:]
        n = len(g)
        return np.hypot(*(ends[n:] - ends[:n]).T)

    def points3d(self, x, y):
        """(x_m, y_m, h) of the points seen at pixel (x, y), for BALL_HEIGHTS_M."""
        h = BALL_HEIGHTS_M
        g = np.repeat(self.ground(x, y)[None], len(h), axis=0)
        for _ in range(4):                     # the ground point h * scale below it
            px = np.c_[np.full(len(h), float(x)), y + h * self._px_per_metre(g)]
            g = np.asarray(self.conv.to_meters_batch(px), float)
        # Only over the playing area: a ground point beyond it (towards the horizon
        # of the ground plane) is no place for a ball in play.
        inside = ((g[:, 0] >= -PLAY_MARGIN_M) & (g[:, 0] <= W_m + PLAY_MARGIN_M)
                  & (g[:, 1] >= -PLAY_MARGIN_M) & (g[:, 1] <= L_m + PLAY_MARGIN_M))
        inside[0] = True                       # the point itself, on the ground
        return np.c_[g, h][inside]

    def on_player(self, a, box):
        x, y, w, h = box
        return (x - PLAYER_SIDE_MARGIN * w <= a[0] <= x + w + PLAYER_SIDE_MARGIN * w
                and y - PLAYER_TOP_MARGIN * h <= a[1] <= y + h + PLAYER_SIDE_MARGIN * h)

    def allowed(self, a, b, gap, velocity=None, players=()):
        """True if the ball can be at pixel b `gap` frames after pixel a.

        velocity: (vx, vy) px/frame of the ball at a, or None if unknown;
        players: (x, y, w, h) boxes at a's frame.
        """
        a, b = np.asarray(a, float), np.asarray(b, float)
        gap = max(1, gap)
        if velocity is not None and np.all(np.isfinite(velocity)):           # (a)
            off = np.hypot(*(b - (a + np.asarray(velocity, float) * gap)))
            if off <= (CONTINUITY_PX + CONTINUITY_GROWTH_PX * gap * gap) * self.s:
                return True
        reach = MAX_BALL_SPEED_MS * gap / self.fps
        known = velocity is not None and np.all(np.isfinite(velocity))
        ga, gb = self.ground(*a), self.ground(*b)
        if np.hypot(*(gb - ga)) <= reach:                                    # (b)
            if not known:
                return True
            # A bounce keeps the ball's way over the ground; only a racket turns it.
            way = ga - self.ground(*(a - np.asarray(velocity, float)))
            move = gb - ga
            norms = np.hypot(*way) * np.hypot(*move)
            if norms < 1e-9 or np.dot(way, move) >= BOUNCE_MAX_TURN_COS * norms:
                return True
        B = None
        for box in players:                                                   # (c)
            if not self.on_player(a, box):
                continue
            B = self.points3d(*b) if B is None else B
            fx, fy = self.ground(box[0] + box[2] / 2, box[1] + box[3])
            R = np.c_[np.full(len(self.racket), fx), np.full(len(self.racket), fy), self.racket]
            d = np.sqrt(((R[:, None, :] - B[None, :, :]) ** 2).sum(-1)).min()
            if d - RACKET_SLACK_M <= reach:
                return True
        if velocity is None or not np.all(np.isfinite(velocity)):            # (d)
            A, B = self.points3d(*a), (self.points3d(*b) if B is None else B)
            if np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)).min() <= reach:
                return True
        return False
