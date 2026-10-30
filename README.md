# FramePath / Vision Tracking Lab

![Tests](https://github.com/Hemanth-TS/framepath-vision-lab/actions/workflows/tests.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-3776AB)
![License](https://img.shields.io/badge/license-MIT-22c55e)

A synthetic image-to-track pipeline with connected components, IoU matching and line-crossing counts.

> **New portfolio demo · created 30 October 2026.** This is an original,
> AI-assisted implementation created for this portfolio. It is not a past
> client delivery, production deployment, or evidence of commercial results.
> All supplied data is synthetic. No popular project was copied or forked.

## Run in under a minute

Only Python 3.11 or later is required. No API keys, model downloads,
paid services or third-party Python packages are needed.

```sh
git clone https://github.com/Hemanth-TS/framepath-vision-lab.git
cd framepath-vision-lab
python framepath.py
python -m unittest discover -s tests -v
```

## Pipeline

`Synthetic grayscale frames → threshold → 4-connected components → greedy IoU assignment → crossing counter`

The CLI renders eight tiny images as portable grayscale PGM files in `output/`
and prints a JSON track trace. The moving rectangle crosses x=20 once; a
second rectangle remains stationary. The final count should be **1**.
Coordinates are half-open `(x1,y1,x2,y2)`. IDs persist while overlap is above
the configured threshold; unmatched tracks expire after `max_missed` frames.

## Input and extension

`samples/sequence.json` specifies image dimensions, a vertical counting line,
and rectangle coordinates per frame. Use `--input` and `--output` to change
the fixture and destination. `components()` also accepts any rectangular
2D list of 8-bit grayscale pixels. A real detector adapter can provide boxes
to `Tracker.update()` without changing the counter.

## Limits

**This is classical image processing on synthetic frames, not YOLO, BoT-SORT,
face recognition or a learned vision model.** Greedy IoU matching has no
appearance model, motion prediction or optimal assignment. Fast motion,
merged components and long occlusions can switch IDs. Only left-to-right
crossings count, once per track; fragmented tracks can overcount. PGM files
can be opened in image tools supporting Netpbm. No real video or biometric
data is included, and no accuracy claims are made.


## Repository guide

- `framepath.py`: implementation and command-line interface.
- `samples/`: fictional input fixtures; safe to edit and reuse.
- `tests/`: regression tests, including boundary cases.
- `.github/workflows/tests.yml`: automated Windows/Linux checks.
- [CHANGELOG.md](CHANGELOG.md): release notes.
- [CONTRIBUTING.md](CONTRIBUTING.md): development and bug-report guidance.

## Maintenance and provenance

Version **0.1.0** is an educational demo, with no production support commitment.
Changes should include a regression test and an entry in the changelog.
Algorithms are implemented directly in this repository; there are no vendored
models, third-party source files, proprietary assets or external datasets.
Badges show GitHub's current CI status; sample outputs are not independent benchmarks.

[Portfolio](https://hemanth-ts.github.io/) · [MIT license](LICENSE)
