"""Turns a stream of per-frame predictions into letters, words and sentences.

How signing works for the user
------------------------------
* HOLD a sign steady for ~0.8 s  -> the letter is added to the current word.
* Same letter twice in a row ("LL")  -> lower your hand out of view for a moment
  (>0.35 s) and sign it again. Changing to a different sign needs no pause.
* Lower your hands out of frame for ~1.8 s -> the word is finished (a space).
* Stay out of frame ~5 s after a word -> the sentence is finished (optional
  auto-translate).

No camera / ML code in here, so it is easy to unit-test and tune.
"""
from __future__ import annotations

from collections import Counter, deque


class SignComposer:
    def __init__(self, conf_thresh=0.70, hold_s=0.8, vote_frac=0.70, release_s=0.35,
                 word_gap_s=1.8, sentence_gap_s=5.0, min_frames=4):
        self.conf_thresh = conf_thresh      # ignore predictions less confident than this
        self.hold_s = hold_s                # how long a sign must be held
        self.vote_frac = vote_frac          # fraction of frames in the hold window that must agree
        self.release_s = release_s          # hand absent this long -> "released" (allows repeats)
        self.word_gap_s = word_gap_s        # hand absent this long -> end of word
        self.sentence_gap_s = sentence_gap_s
        self.min_frames = min_frames
        self.reset()

    # ----------------------------------------------------------------- state
    def reset(self):
        self.current_word = ""
        self.win: deque = deque()           # (time, label-or-None) over the last hold_s seconds
        self.win_start = None
        self.locked = False                 # True right after a commit until release / new sign
        self.last_committed = None
        self.absent_since = None
        self.sentence_pending = False
        # for the UI
        self.candidate = None
        self.progress = 0.0
        self.status = "no_hand"             # no_hand | reading | locked

    def _clear_window(self):
        self.win.clear()
        self.win_start = None

    # ----------------------------------------------------------------- manual controls
    def break_word(self):
        events = []
        if self.current_word:
            events.append(("word", self.current_word))
            self.current_word = ""
            self.sentence_pending = True
        self.locked = False
        self.last_committed = None
        self._clear_window()
        return events

    def backspace(self):
        self.current_word = self.current_word[:-1]

    # ----------------------------------------------------------------- main update
    def update(self, label, conf, hand_present, now):
        """Feed one frame. Returns a list of events:
        ("letter", "A"), ("word", "HELLO"), ("sentence", None)."""
        events = []

        if not hand_present:
            if self.absent_since is None:
                self.absent_since = now
            gap = now - self.absent_since
            self.candidate, self.progress, self.status = None, 0.0, "no_hand"
            if gap >= self.release_s:
                self._clear_window()
                self.locked = False
            if gap >= self.word_gap_s and self.current_word:
                events += self.break_word()
            if gap >= self.sentence_gap_s and self.sentence_pending and not self.current_word:
                self.sentence_pending = False
                events.append(("sentence", None))
            return events

        self.absent_since = None
        lab = label if (label is not None and conf >= self.conf_thresh) else None
        if self.win_start is None:
            self.win_start = now
        self.win.append((now, lab))
        while self.win and now - self.win[0][0] > self.hold_s:
            self.win.popleft()

        counts = Counter(l for _, l in self.win if l is not None)
        cand, frac = None, 0.0
        if counts:
            cand, c = counts.most_common(1)[0]
            frac = c / len(self.win)
        elapsed = now - self.win_start
        self.candidate = cand

        repeat_blocked = self.locked and cand == self.last_committed
        if cand is None:
            self.progress, self.status = 0.0, "reading"
        elif repeat_blocked:
            self.progress, self.status = 1.0, "locked"
        else:
            self.progress, self.status = min(1.0, elapsed / self.hold_s), "reading"

        if (cand is not None and not repeat_blocked and elapsed >= self.hold_s
                and len(self.win) >= self.min_frames and frac >= self.vote_frac):
            self.current_word += cand
            self.last_committed = cand
            self.locked = True
            self._clear_window()
            self.progress, self.status = 1.0, "locked"
            events.append(("letter", cand))
        return events
