# Gaming highlight selection — 25 September 2026

## Research and interpretation

The central distinction is between a well-formed conversation and an entertaining
event. A horror stream can contain coherent pet-care talk with no clip-worthy
reaction. Conversely a short exclamation can be the payoff to a strong event.

Primary sources reviewed:

1. [YouTube: Shorts deep dive, 2025](https://blog.youtube/creator-and-artist-stories/youtube-shorts-deep-dive/).
   The interview emphasizes an immediate hook and a satisfying miniature story.
   This is creator guidance, not a universal experimentally established one-second
   threshold. Implementation: score the opening separately from the payoff; do not
   spend the opening on greetings or unnecessary exposition.
2. [Ringer & Nicolaou, 2018: game-stream highlight detection](https://arxiv.org/abs/1807.09715).
   Stream highlights depend on social signals from the player as well as gameplay;
   their approach combines gameplay, facial expression and speech. Implementation
   inference: prioritize streamer reactions and departures from routine instead of
   limiting highlights to kills, victories, or well-articulated topics.
3. [Fu et al., EMNLP 2017: audience chat reactions](https://aclanthology.org/D17-1102/).
   Work on English and Traditional Chinese League of Legends streams uses visual
   features and audience language, including slang. Implementation inference:
   retain multilingual/slang understanding; a viewer prompt plus streamer payoff
   can qualify. This project does not currently ingest audience chat, so it must
   not invent chat surges or chat sentiment.
4. [Hasan et al., EMNLP 2019: UR-FUNNY](https://aclanthology.org/D19-1211/).
   Humor is expressed through language, gesture and prosody. The accompanying
   research examines context and punchlines. Implementation inference: identify a
   joke's setup and payoff rather than counting laughter words; transcript-only
   detection has limits for physical humor and delivery.
5. [Xie et al., ACL 2021: uncertainty and surprisal](https://aclanthology.org/2021.acl-short.6/).
   The work models a setup creating uncertainty and a punchline disrupting
   expectations. Implementation inference: dad jokes, absurd dialogue and
   expectation-to-failure reversals deserve their own comedy route.
6. [Castro et al., ACL 2019: MUStARD](https://aclanthology.org/P19-1455/).
   Sarcasm detection considers multimodal utterances and preceding dialogue.
   Implementation inference: deadpan and disappointed understatement require their
   triggering context; low vocal energy is not proof of low highlight value.
7. [Lei et al., NeurIPS 2021: QVHighlights](https://arxiv.org/abs/2107.09609).
   The task distinguishes locating relevant moments from scoring their saliency.
   Implementation inference: keep a peak anchor and a surrounding contiguous span,
   then suppress repeated selections of the same event.
8. [Mundnich et al.: Audiovisual Highlight Detection, 2021](https://arxiv.org/abs/2102.05811).
   The work combines image, affect-related and audio representations. It supports
   treating audiovisual information as useful evidence; a text-only prompt does
   not acquire those capabilities merely by asking about reactions.

These sources support the design principles, not the exact weights, categories,
duration defaults, or a promise of increased views. The rubric below is an
editorial heuristic for this gaming workflow and needs calibration against clips
the user actually accepts and audience retention measurements.

## Categories and evidence rules

Pre/post durations below are editorial starting points around the peak, not
automatic padding. The implementation asks the model for segment boundaries and
preserves the resulting full setup, outcome and immediate aftermath.

| Category | Required event/contrast | Common false positive | Suggested pre / post seconds |
|---|---|---|---|
| jumpscare | Sudden fright with a startled response | Mentioning a ghost or saying a game is scary | 2–8 / 3–12 |
| scream_panic | Fear escalating or loss of control | Profanity alone, quoting a scream | 2–8 / 3–12 |
| funny_reaction | A response that is itself comic | Saying “funny” without a comic beat | 2–6 / 3–10 |
| dad_joke | Setup plus pun/receh punchline; a groan can be payoff | Unrelated facts or a question without its answer | 3–12 / 2–8 |
| disappointment | Expectation followed by visible/verbal deflation | Routine negative commentary | 3–10 / 3–10 |
| rage | Concrete trigger and escalating frustration | Ambient angry language | 3–10 / 3–12 |
| npc_moment | Absurd/repetitive scripted behavior or NPC-like response | Merely mentioning an NPC | 3–10 / 3–10 |
| bug_glitch | Evidenced broken/unexpected game behavior and consequence | Calling something a bug without showing/describing it | 3–10 / 3–10 |
| unexpected_fail | Confidence/plan overturned by failure | Repeated routine death without a new beat | 3–12 / 3–10 |
| clutch | Stakes, difficult reversal and resolved outcome | Routine kill or an unresolved attempt | 5–20 / 3–10 |
| absurd_dialogue | Comic mismatch or escalating nonsensical exchange | Random but unfunny topic changes | 3–12 / 2–8 |
| deadpan_reaction | Understatement contrasting with a preceding event | Flat ordinary speech | 3–10 / 2–8 |
| awkward_silence | A social/comic setup and a meaningful pause/payoff | ASR gap, loading screen, muted microphone | 3–10 / 2–8 |
| plot_twist_reveal | Established expectation overturned by a reveal | Exposition without a reversal | 5–20 / 3–12 |
| streamer_chat_interaction | Viewer input answered with a reaction or comic payoff | Greetings, thanking subscribers, reading messages | 3–12 / 3–10 |
| payoff | Anticipation/callback resolved in a distinct beat | Setup promising something that never happens | 5–20 / 3–12 |

The current analyzer can use only transcribed evidence for these categories.
It cannot establish a visual event, loudness, facial expression or chat response
that is absent from its input. “Awkward silence” needs context establishing its
meaning; an empty time interval alone is insufficient.

## Deterministic rubric

Every component uses 0–10: 0 absent, 3 weak, 5 clear/moderate, 8 strong, 10 exceptional.

```
impact = max(reaction, surprise, comedy, skill)
score = .35*impact + .25*payoff + .20*hook + .10*event_clarity
        + .05*standalone_coherence + .05*novelty
        - .35*filler - .20*missing_context - .15*repetition
```

Clamp final score to 0–10. Using the strongest impact route allows a quiet joke
or skilled clutch to compete with a scream without earning all four components.
Payoff may be the reaction itself; it does not require a long narrative.

Hard caps at 3.9 apply to filler/unknown categories, impact below 5, payoff below
4, event clarity below 4, filler at least 6, or missing context at least 7.
Missing/unverifiable quoted evidence also caps the score. Ollama returns an
`evidence_segment` ID; the backend copies the actual source line as `evidence`
rather than letting the model compose a quotation. Missing, non-numeric or
non-finite component scores fail inference instead of becoming a fake highlight.
The model's own overall score cannot bypass this calculation.

Gaming candidates must meet `MIN_VIRAL_SCORE` (currently 6); the UI/CLI minimum
may further tighten selection. Coherent mundane talk cannot be rescued by a high
coherence score. There is no forced top-three fallback in the gaming analyzer or
the web/CLI paths. Returning no clips is a valid result.

## Boundaries, metadata and compatibility

- `ANALYSIS_PROFILE` defaults to `gaming`. Set it to `general` before starting
  the app, or pass `profile='general'` to `ViralMomentAnalyzer`, to retain the
  previous general multilingual rubric and selection behavior.
- Gaming outputs require valid ordered start/peak/end segment IDs, an exact
  evidence quote inside the selected span, and duration 3–60 seconds. Three
  seconds is an implementation minimum, not a recommendation to make every clip
  that short. Short complete reactions are not padded to the generic 15-second
  minimum. Overlong/invalid spans are rejected, never replaced by an entire window.
- `peak_timestamp` is the start of the selected peak segment, not a frame-accurate
  acoustic peak. `peak_end`, `pre_context_seconds`, `post_context_seconds`,
  `evidence`, and `score_breakdown` remain in analysis results and web clip data.
- Sentence refinement preserves the selected gaming arc and its pauses. No
  cold-open rearrangement, silence removal or invented hook is performed.
- Highest score wins duplicate suppression: intersecting peak intervals or at
  least 65% overlap relative to the shorter clip. Distinct moments survive.
- All transcript windows still reach the LLM. Keyword ranking only changes call
  order and does not filter out non-English or quiet moments. Current defaults
  remain 60-second windows with 30-second overlap. One candidate per window is
  assessed; nearby multiple events may still need manual review.
- Cache keys include profile, analyzer version, gaming prompt, weights and
  penalties in addition to existing model/digest/language/settings information.
  Old general scores are not reused for gaming runs. Failed/partial inference is
  not cached as a complete analysis.
- Qwen 3.5 9B Q4_K_M, 8192 context, local inference, multilingual/code-switching
  instructions and transcription/subtitle/render settings are retained.

## Validation and next calibration

Regression tests cover quiet alternatives, filler rejection, invalid scores,
evidence grounding, peak and aftermath preservation, cache separation,
multilingual prompts, duplicate events and the general-profile compatibility path.
Live model smoke tests use explicit synthetic positive/negative examples and
three excerpts of the existing Windah transcript. Synthetic assertions exercise
the intended behavior; they are not a blind benchmark of video highlight quality.

For a stronger evaluation, manually label peak timestamps and clip boundaries on
several held-out horror/comedy/competitive streams, then measure accepted clips
among the top candidates, peak recall, duplicates, missing payoff and filler rate.
Measure quiet comedy and loud reactions separately. The next capability gap is
audio/visual candidate generation for screams, silent expressions and game events
ASR omits. Audio spikes alone would still need confirmation: game sound effects
and music are not automatically streamer reactions.
