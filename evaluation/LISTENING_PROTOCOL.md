# AudioForge listening protocol

This protocol is the human-perception gate for DSP changes whose correctness
cannot be established by unit tests or objective metrics alone. Test audio,
answers, and aggregate results are local evaluation artifacts and are never
release assets.

## Panel and playback

- Use at least three listeners when a change could alter release defaults; one
  trained listener is sufficient for an optional expert control.
- Use the same neutral headphones or monitors, device, sample rate, and fixed
  playback gain for the entire session.
- Loudness-match A and B to within 0.2 LU before randomization. Do not normalize
  true peaks independently.
- Randomize and blind the incumbent/candidate labels. Include a hidden repeated
  reference in every ten trials; discard a session if repeated-reference scores
  differ by more than two points on a seven-point scale.

## Corpus

Use 12-20 clips spanning:

- Clean close-mic speech, room noise at low and moderate SNR, and speech after a
  long pause.
- At least four speakers and more than one language.
- Bright, dark, proximity-heavy, plosive-heavy, and sibilant speech.
- Real child/high-pitched speech from the local SLR98 edge corpus.
- Both bundled DeepFilterNet modes where noise suppression is being judged.

Clips should be 8-20 seconds. Do not use the same utterance to tune a threshold
and judge it.

## Questions

Rate each blinded sample from 1 (unacceptable) to 7 (excellent):

1. Speech naturalness.
2. Intelligibility.
3. Noise reduction without musical noise.
4. Spectral balance without hollow, harsh, or muffled coloration.
5. Dynamics stability without pumping, clicks, or words changing level after
   pauses.
6. Overall preference.

Also mark binary defects: clipped consonant, speech dropout, false gate opening,
de-esser lisp, transient click, or obvious pumping.

## Retention rule

Retain a candidate only when all conditions hold:

- Median overall preference is not worse than the incumbent.
- The lower 95% bootstrap confidence bound for mean preference difference is
  above -0.25 points.
- No defect category increases by more than one event per five minutes.
- Clean-speech naturalness is not worse by more than 0.5 median points.
- Any claimed improvement is supported by both objective evidence and at least
  60% listener preference on the affected condition.

If the rule fails, restore the incumbent behavior. Record the corpus manifest
hash, randomized trial map, listener count, playback chain, raw ratings,
bootstrap seed, and decision under `evaluation/`.
