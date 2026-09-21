# Benchmark

```bash
python -m benchmarks.run
```

`--markdown` for a table to paste into the paper, `--json` for machine-readable
output. Exit code is non-zero if anything fails, so CI can gate on it.

No API key, no GPU, no model weights. It runs in under a second.

---

## What this measures

The paper's contribution is explicitly **system level**: automatic routing,
modular orchestration and label refinement over pretrained models used as-is,
with no fine-tuning. That is what this measures, and all of it is
deterministic.

| Benchmark | Question |
|---|---|
| Wildlife gate | Does the animal gate fire for the labels the classifier actually emits? |
| Label refinement | Are coarse COCO labels promoted correctly, and is everything else left alone? |
| Model routing | Does the agent pick the right model sequence for each input type? |
| Routing selectivity | How many model invocations does routing avoid? |
| Determinism | Does the same input always produce the same decision? |
| Routing overhead | How much time does the routing layer itself cost? |

## What this does **not** measure

It does not measure how accurately ResNet50 classifies, how accurately YOLOv8
detects, or how good a caption BLIP writes. Those are properties of the
pretrained models, not of this project's code.

Measuring those needs a labelled image corpus and the `full` profile. Nothing
here should be quoted as a model-accuracy figure.

## Current results

```
Benchmark                   Passed   Total     Rate
--------------------------------------------------------------------------
Wildlife gate                   11      11   100.0%
Label refinement                51      51   100.0%
Model routing                   15      15   100.0%
Routing selectivity              9       9   100.0%
Determinism                     33      33   100.0%
Routing overhead                 1       1   100.0%
--------------------------------------------------------------------------
OVERALL                        120     120   100.0%
```

| Measure | Value |
|---|---|
| Taxonomy entries | 791 |
| ImageNet classes routed to the wildlife model | 127 of 1000 |
| Coarse COCO labels with a refinement map | 14 |
| Secondary subjects preserved in multi-animal scenes | 100% |
| Model invocations avoided by routing | 66.7% (21 instead of 63) |
| Routing overhead, mean | 0.64 ms |
| Routing overhead, max | 2.52 ms |

## Reconciling this with the paper

Two numbers in the paper need care when quoted alongside these results.

**"132 species".** The shipped taxonomy holds **791** unique names, of which
**127** correspond to ImageNet classes the classifier can actually emit. 127 is
the defensible figure for end-to-end species coverage: a name in the taxonomy
that ResNet50 cannot produce never affects a result. Whichever figure the paper
means, it should say which.

**"95% wildlife accuracy".** This benchmark cannot produce that number, because
it does not run models. It does establish the mechanism the claim rests on:
refinement is correct on 51 of 51 cases, and secondary subjects are never
overwritten. To substantiate the accuracy figure, run the `full` profile over a
labelled wildlife set and report that separately.

## Bugs this found

Written down because it is the argument for having the benchmark at all.

1. **Both elephant classes never reached the wildlife detector.** The taxonomy
   stored `"African elephant"` capitalised while lookups lowercased the input,
   so the match always failed. Also affected Arctic fox and American black
   bear. ImageNet classes reaching the wildlife path: 123 before, 127 after.

2. **Refinement claimed a change it had not made.** When the detector already
   named the species the classifier reported (`zebra` + `zebra`), the detection
   was still flagged `corrected: true`, and the interface displayed
   "refined from 'zebra'".

## Adding cases

Edit `benchmarks/dataset.py`. Each case states its expected outcome, so a case
that disagrees with the code fails loudly rather than being absorbed. Cases are
generated from the project's own taxonomy where possible, so the suite cannot
drift away from what ships.
