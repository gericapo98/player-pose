import numpy as np

from player_pose.pose import JOINT_NAMES, PoseDetection
from player_pose.sort import SORTTracker, iou_matrix


def det(frame, x1, y1, x2, y2, joints=len(JOINT_NAMES)):
    """A detection with `joints` confident joints spread over the box (corners included)."""
    pts = np.full((len(JOINT_NAMES), 2), np.nan, dtype=np.float32)
    conf = np.zeros(len(JOINT_NAMES), dtype=np.float32)
    xs = np.linspace(x1, x2, joints)
    ys = np.linspace(y1, y2, joints)
    pts[:joints] = np.stack([xs, ys], axis=1)
    conf[:joints] = 1.0
    return PoseDetection(frame, pts, conf, np.array([x1, y1, x2, y2], dtype=np.float32), 1.0)


def ids_per_frame(outputs):
    return [sorted(p.track_id for p in out) for out in outputs]


def test_iou_matrix():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[0, 0, 10, 10], [5, 5, 15, 15], [20, 20, 30, 30]], dtype=np.float32)
    np.testing.assert_allclose(iou_matrix(a, b)[0], [1.0, 25 / 175, 0.0], atol=1e-6)


def test_empty_input_yields_no_tracks():
    assert SORTTracker().update([]) == []


def test_two_moving_players_keep_their_ids():
    tracker = SORTTracker(min_hits=3)
    outputs = []
    for f in range(40):
        near = det(f, 100 + 3 * f, 400, 180 + 3 * f, 600)   # drifting right
        far = det(f, 600, 200 - f, 660, 320 - f)            # drifting up
        outputs.append(tracker.update([far, near]))
    assert all(ids == [1, 2] for ids in ids_per_frame(outputs)[3:])
    # identity follows the box, not the input order
    last = {p.track_id: p.detection.box[0] for p in outputs[-1]}
    assert last[1] > last[2]  # id 1 was the far player (x=600), id 2 the near one


def test_short_gap_keeps_id_and_long_gap_does_not():
    tracker = SORTTracker(max_age=5, min_hits=1)
    box = lambda f: det(f, 100, 100, 200, 300)
    for f in range(5):
        tracker.update([box(f)])
    for f in range(5, 8):          # 3 missed frames < max_age
        assert tracker.update([]) == []
    assert [p.track_id for p in tracker.update([box(8)])] == [1]
    for f in range(9, 20):         # 11 missed frames > max_age
        tracker.update([])
    assert [p.track_id for p in tracker.update([box(20)])] == [2]


def test_confirmed_track_is_reported_right_after_a_gap():
    tracker = SORTTracker(min_hits=3)
    box = lambda f: det(f, 100, 100, 200, 300)
    for f in range(5):
        tracker.update([box(f)])
    tracker.update([])
    assert [p.track_id for p in tracker.update([box(6)])] == [1]  # no re-confirmation delay


def test_ghost_track_cannot_steal_a_confirmed_players_detection():
    tracker = SORTTracker(min_hits=3)
    player = lambda f: det(f, 600, 200, 660, 320)
    for f in range(5):
        tracker.update([player(f)])
    # one-frame partial detection right next to the player spawns a tentative track
    tracker.update([player(5), det(5, 605, 205, 640, 300, joints=8)])
    # next frames the player's box is closer to where the ghost was: player still wins
    for f in range(6, 12):
        out = tracker.update([det(f, 604, 204, 642, 305)])
        assert [p.track_id for p in out] == [1]
    assert len(tracker.tracks) == 1  # and the ghost has been dropped


def test_intermittent_detections_still_confirm_a_track():
    tracker = SORTTracker(min_hits=3, tentative_max_age=3)
    box = lambda f: det(f, 600, 200, 660, 320)
    outputs = [tracker.update([box(f)] if f % 2 == 0 else []) for f in range(8)]  # seen every other frame
    assert ids_per_frame(outputs)[4] == [1]


def test_redetect_recovers_missed_player():
    tracker = SORTTracker(min_hits=3)
    far_box = lambda f: det(f, 600, 200, 660, 320)
    near_box = lambda f: det(f, 100, 400, 180, 600)
    calls = []

    def redetect(search):
        calls.append(search.copy())
        return [far_box(99)]

    for f in range(5):
        tracker.update([far_box(f), near_box(f)], redetect=redetect)
    assert calls == []
    out = tracker.update([near_box(5)], redetect=redetect)   # Vision misses the far player
    assert len(calls) == 1
    assert iou_matrix(calls[0][None], far_box(0).box[None])[0, 0] > 0.8
    assert sorted(p.track_id for p in out) == [1, 2]


def test_redetect_rejects_partial_and_wrong_sized_candidates():
    tracker = SORTTracker(min_hits=3)
    far_box = lambda f: det(f, 600, 200, 660, 320)
    for f in range(5):
        tracker.update([far_box(f)])
    sliver = det(5, 600, 250, 666, 274)                    # 66x24 px, full joints but wrong height
    partial = det(5, 600, 200, 660, 320, joints=6)         # right box, too few joints
    out = tracker.update([], redetect=lambda _: [sliver, partial])
    assert out == []
    assert tracker.tracks[0].last_box[3] - tracker.tracks[0].last_box[1] == 120  # box untouched


def test_redetect_ignores_candidates_already_matched():
    tracker = SORTTracker(min_hits=1)
    near_box = lambda f: det(f, 100, 400, 180, 600)
    far_box = lambda f: det(f, 600, 200, 660, 320)
    for f in range(3):
        tracker.update([near_box(f), far_box(f)])
    # far player missing; the crop "finds" only the near player again
    out = tracker.update([near_box(3)], redetect=lambda _: [near_box(3)])
    assert [p.track_id for p in out] == [1]
