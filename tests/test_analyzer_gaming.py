import json
import pytest
from modules.analyzer import ViralMomentAnalyzer
from modules.gaming_rubric import score_assessment, CATEGORIES


@pytest.fixture
def analyzer(monkeypatch, tmp_path):
    monkeypatch.setattr(ViralMomentAnalyzer, '_check_ollama_connection', lambda self: None)
    a = ViralMomentAnalyzer(profile='gaming')
    a.cache_dir = tmp_path
    return a


def assessment(**changes):
    result = dict(reaction=9, surprise=9, comedy=2, skill=0, payoff=9,
                  hook=8, event_clarity=9, standalone_coherence=5, novelty=8,
                  filler=0, missing_context=0, repetition=0, reason='Reaksi mendadak.',
                  moment_type='jumpscare', evidence='AAAA! Kaget gue!',
                  start_segment=0, peak_segment=1, end_segment=2)
    return {**result, **changes}


def transcript():
    return {'duration': 18, 'language': 'id', 'segments': [
        {'start': 0, 'end': 5, 'text': 'Aman kan? Buka pintunya.'},
        {'start': 5, 'end': 7, 'text': 'AAAA! Kaget gue!'},
        {'start': 7, 'end': 11, 'text': 'Jantung gue copot!'},
        {'start': 11, 'end': 18, 'text': 'Sekarang kita bahas makanan kucing.'}]}


@pytest.mark.parametrize('category', [c for c in CATEGORIES if c != 'filler'])
def test_categories_can_qualify_without_loudness(category):
    quiet = assessment(moment_type=category, reaction=0, surprise=2, comedy=9)
    assert score_assessment(quiet)[0] > 7


def test_filler_cannot_be_rescued_by_coherence_or_model_total():
    filler = assessment(moment_type='filler', reaction=0, surprise=0, comedy=0,
                        payoff=0, filler=9, standalone_coherence=10, overall_score=10)
    assert score_assessment(filler)[0] < 4 < score_assessment(assessment())[0]


@pytest.mark.parametrize('changes', [dict(payoff=0), dict(event_clarity=0),
    dict(missing_context=9), dict(filler=7), dict(moment_type='invented'),
    dict(reaction=0, surprise=0, comedy=0, skill=0)])
def test_eligibility_gates(changes):
    assert score_assessment(assessment(**changes))[0] <= 3.9


@pytest.mark.parametrize('value', [True, '9', None, float('nan'), float('inf')])
def test_malformed_scores_fail(value):
    with pytest.raises((ValueError, TypeError)):
        score_assessment(assessment(reaction=value))


def test_peak_context_short_span_and_cache(analyzer, monkeypatch):
    monkeypatch.setattr(analyzer, '_call_ollama', lambda _: json.dumps(assessment()))
    moments = analyzer.analyze_transcript(transcript())
    assert len(moments) == 1
    m = moments[0]
    assert (m['start'], m['end'], m['peak_timestamp']) == (0, 11, 5)
    assert (m['pre_context_seconds'], m['post_context_seconds']) == (5, 4)
    assert 'kucing' not in m['text']
    assert m['score_breakdown']['impact'] == 9
    monkeypatch.setattr(analyzer, '_find_sentence_boundaries', lambda *args: pytest.fail('Padded gaming clip'))
    assert analyzer.refine_moments(moments, transcript())[0]['end'] == 11
    monkeypatch.setattr(analyzer, '_call_ollama', lambda _: pytest.fail('Cache miss'))
    assert analyzer.analyze_transcript(transcript())[0]['peak_timestamp'] == 5


@pytest.mark.parametrize('changes', [dict(start_segment=-1), dict(peak_segment=80),
    dict(end_segment=0), dict(peak_segment=True), dict(peak_segment=None),
    dict(evidence='Invented screaming'), dict(start_segment=2, peak_segment=2)])
def test_invalid_or_ungrounded_spans_do_not_fall_back(analyzer, monkeypatch, changes):
    monkeypatch.setattr(analyzer, '_call_ollama', lambda _: json.dumps(assessment(**changes)))
    assert analyzer.analyze_transcript(transcript()) == []


def test_filler_returns_empty_even_when_every_window_is_filler(analyzer, monkeypatch):
    monkeypatch.setattr(analyzer, '_call_ollama', lambda _: json.dumps(assessment(
        moment_type='filler', filler=10, payoff=0, reaction=0, surprise=0)))
    assert analyzer.analyze_transcript(transcript()) == []
    assert json.loads(next(analyzer.cache_dir.glob('*.json')).read_text()) == []


def test_high_score_with_invalid_peak_is_not_cached(analyzer, monkeypatch):
    monkeypatch.setattr(analyzer, '_call_ollama', lambda _: json.dumps(assessment(peak_segment=800)))
    assert analyzer.analyze_transcript(transcript()) == []
    assert not list(analyzer.cache_dir.glob('*.json'))


def test_overlapping_windows_keep_best_peak_but_distinct_events_survive(analyzer):
    def moment(start, end, peak, score):
        return dict(start=start, end=end, duration=end-start, peak_timestamp=peak,
                    peak_end=peak+2, score=score)
    a, b, c = moment(0, 20, 8, 7), moment(2, 22, 8, 9), moment(35, 50, 40, 8)
    assert analyzer._deduplicate_gaming_moments([a, b, c]) == [b, c]


def test_profiles_have_separate_cache(analyzer):
    key = analyzer._generate_cache_key(transcript(), 30, 'sliding', 6)
    analyzer.profile = 'general'
    assert key != analyzer._generate_cache_key(transcript(), 30, 'sliding', 6)


@pytest.mark.parametrize('language', ['id', 'en', 'fr', 'ja', None])
def test_gaming_language_preserved(analyzer, monkeypatch, language):
    def call(prompt):
        assert f'Language: {language or "auto-detect"}' in prompt
        return json.dumps(assessment())
    monkeypatch.setattr(analyzer, '_call_ollama', call)
    assert analyzer._evaluate_chunk('AAAA! Kaget gue!', language)['overall_score'] > 7


def test_evidence_across_formatted_segments_is_grounded(analyzer, monkeypatch):
    text = '[0] 0.00-5.00s: Hello\n[1] 5.00-10.00s: world!'
    monkeypatch.setattr(analyzer, '_call_ollama', lambda _: json.dumps(assessment(evidence='Hello world!')))
    assert analyzer._evaluate_chunk(text)['overall_score'] > 7


def test_gaming_ollama_schema_bounds_ids(analyzer, monkeypatch):
    import ollama
    captured = {}
    def chat(**kwargs):
        captured.update(kwargs)
        return {'message': {'content': json.dumps(assessment())}}
    monkeypatch.setattr(ollama, 'chat', chat)
    analyzer._evaluate_chunk('[0] 0.00-5.00s: AAAA! Kaget gue!\n[1] 5.00-10.00s: Wah')
    schema = captured['format']
    assert schema['properties']['peak_segment']['anyOf'][0]['maximum'] == 1
    assert 'evidence_segment' in schema['required']
    analyzer._call_ollama('Generate metadata')
    assert captured['format'] == 'json'


def test_evidence_is_copied_from_source_not_generated(analyzer, monkeypatch):
    monkeypatch.setattr(analyzer, '_call_ollama', lambda _: json.dumps(assessment(
        evidence_segment=1, evidence='Invented quote')))
    result = analyzer._evaluate_chunk('[0] 0.00-5.00s: Aman?\n[1] 5.00-10.00s: AAAA! Kaget gue!')
    assert result['evidence'] == 'AAAA! Kaget gue!'
    assert result['overall_score'] > 7


def test_plain_text_scoring_compatibility(analyzer, monkeypatch):
    monkeypatch.setattr(analyzer, '_call_ollama', lambda _: json.dumps(assessment(
        evidence_segment=None, start_segment=None, peak_segment=None, end_segment=None)))
    score, _ = analyzer._analyze_chunk('AAAA! Kaget gue!', 'id')
    assert score > 7


def test_web_minimum_score_never_falls_back_to_lower_clips(analyzer, monkeypatch, tmp_path):
    import app as web
    source = tmp_path / 'source.mp4'
    source.touch()
    monkeypatch.setattr(web, 'DOWNLOADS_DIR', tmp_path)
    monkeypatch.setattr(web.VideoProcessor, 'get_video_info', lambda *a: {'duration': 18})
    monkeypatch.setattr(web, 'transcribe_with_engine', lambda *a, **k: transcript())
    monkeypatch.setattr(web, 'VideoTranscriber', lambda: object())
    monkeypatch.setattr(web, 'ViralMomentAnalyzer', lambda **k: analyzer)
    monkeypatch.setattr(analyzer, 'analyze_transcript', lambda *a, **k: [{'score': 6.5}])
    monkeypatch.setattr(analyzer, 'generate_clip_metadata', lambda *a, **k: pytest.fail('Below-threshold clip accepted'))
    response = web.app.test_client().post('/api/process', json={
        'source': 'local', 'local_file': source.name, 'min_score': 8})
    assert 'No gaming highlights met your minimum score' in response.data.decode()
