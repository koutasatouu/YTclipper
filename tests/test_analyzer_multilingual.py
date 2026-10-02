import json
import pytest
import ollama
from modules import analyzer as module
from modules.analyzer import ViralMomentAnalyzer


@pytest.fixture
def analyzer(monkeypatch, tmp_path):
    monkeypatch.setattr(ViralMomentAnalyzer, '_check_ollama_connection', lambda self: None)
    instance = ViralMomentAnalyzer(provider='ollama', profile='general')
    instance.cache_dir = tmp_path
    return instance


def transcript(language='id'):
    lines = ['Aku tanya kenapa dia selalu datang terlambat.',
             'Dia bilang sedang belajar manajemen waktu.',
             'Ternyata kelasnya juga dia datangi terlambat! Semua tertawa.']
    return {'duration': 30, 'language': language, 'segments': [
        {'start': i * 10, 'end': (i + 1) * 10, 'text': text}
        for i, text in enumerate(lines)]}


def response(**kwargs):
    return json.dumps(dict(overall_score=8, reason='Setup dan punchline lengkap.',
                           moment_type='funny', start_segment=0, end_segment=2, **kwargs))


@pytest.mark.parametrize('language', ['id', 'en', 'ja', 'id-ID', None])
def test_pipeline_retains_category_and_full_arc(analyzer, monkeypatch, language):
    monkeypatch.setattr(analyzer, '_call_ollama', lambda prompt: response())
    result = analyzer.analyze_transcript(transcript(language))
    assert result[0]['moment_type'] == 'funny'
    assert (result[0]['start'], result[0]['end']) == (0, 30)
    monkeypatch.setattr(analyzer, '_call_ollama', lambda prompt: pytest.fail('Cache miss'))
    assert analyzer.analyze_transcript(transcript(language)) == result


def test_cache_tracks_language_prompt_model_and_version(analyzer):
    def key():
        return analyzer._generate_cache_key(transcript(), 30, 'sliding', 5)
    original = key()
    analyzer.model_name = 'different-model'
    changed = key()
    assert changed != original
    analyzer.ANALYSIS_PROMPT += '\nNew rubric'
    assert key() != changed
    changed = key()
    analyzer.ANALYZER_VERSION = 'next'
    assert key() != changed
    assert analyzer._generate_cache_key(transcript('en'), 30, 'sliding', 5) != key()


@pytest.mark.parametrize('enabled', [True, False])
def test_ranking_never_skips_windows(analyzer, monkeypatch, enabled):
    monkeypatch.setattr(module, 'ANALYSIS_PREFILTER_ENABLED', enabled)
    chunks = [{'start': i * 30, 'end': (i + 1) * 30,
               'text': '静かな話' if i % 2 else 'shocking secret!', 'segments': []}
              for i in range(60)]
    ranked, count = analyzer._rank_chunks_for_analysis(chunks, 'ja')
    assert count == len(chunks) == len(ranked)
    assert {id(c) for c in ranked} == {id(c) for c in chunks}


def test_unknown_language_does_not_use_english_keywords(analyzer):
    chunk = {'start': 0, 'end': 30, 'text': 'shocking secret', 'segments': []}
    assert analyzer._score_chunk_for_prefilter(chunk, 'en') > analyzer._score_chunk_for_prefilter(chunk, 'ja')


def test_failed_inference_never_returns_fake_clip_or_cache(analyzer, monkeypatch):
    monkeypatch.setattr(analyzer, '_call_ollama', lambda prompt: 'not valid JSON')
    with pytest.raises(RuntimeError, match='All transcript analysis calls failed'):
        analyzer.analyze_transcript(transcript())
    assert not list(analyzer.cache_dir.glob('*.json'))


def test_ollama_request_uses_bounded_json_mode(analyzer, monkeypatch):
    calls = []
    def chat(**kwargs):
        calls.append(kwargs)
        return {'message': {'content': response()}}
    monkeypatch.setattr(ollama, 'chat', chat)
    analyzer._call_ollama('Analyze this transcript.')
    assert calls[0]['options']['num_ctx'] == 8192
    assert calls[0]['think'] is False
    assert calls[0]['format'] == 'json'
    with pytest.raises(ValueError, match='context budget'):
        analyzer._call_ollama('字' * 8192)


def test_invalid_span_is_not_used(analyzer):
    chunk = analyzer._create_sliding_chunks(transcript()['segments'])[0]
    for first, last in [(-1, 2), (0, 500), (True, 2), (2, 0)]:
        assert analyzer._select_assessed_span(chunk, {'start_segment': first, 'end_segment': last}) == chunk


def test_cloud_provider_request_stays_compatible(analyzer):
    from types import SimpleNamespace
    from unittest.mock import Mock
    create = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content='{}'))]))
    analyzer.openai_client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create)))
    analyzer._call_openai('test')
    assert set(create.call_args.kwargs) == {'model', 'messages', 'max_completion_tokens', 'temperature'}


def test_refinement_cannot_trim_payoff(analyzer, monkeypatch):
    monkeypatch.setattr(analyzer, '_find_sentence_boundaries', lambda *args: (0, 20))
    result = analyzer.refine_moments([{'start': 0, 'end': 30}], transcript())
    assert result[0]['end'] == 30


def test_modern_ollama_model_listing(monkeypatch, capsys):
    analyzer = object.__new__(ViralMomentAnalyzer)
    analyzer.model_name = 'qwen3.5-9b-q4km'
    monkeypatch.setattr(ollama, 'list', lambda: ollama.ListResponse(models=[
        {'model': analyzer.model_name + ':latest', 'digest': 'abc'}]))
    analyzer._check_ollama_connection()
    assert analyzer.model_digest == 'abc'
    assert 'Warning' not in capsys.readouterr().out
