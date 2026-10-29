"""TTS HTTP contract tests; no external service or GPU calls."""
import asyncio
import base64
import io
import json
from pathlib import Path
import sys
import wave
import httpx
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flex_web.server import create_app, Settings
from flex_web.tts import DEFAULT_VOICES, TTSOptions, TTSService, TTSError, decode_audio


def wav_bytes(rate=24000, seconds=.1):
    out = io.BytesIO()
    with wave.open(out, 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b'\0\0' * int(rate * seconds))
    return out.getvalue()


@pytest.fixture
def assets(tmp_path):
    for voice in DEFAULT_VOICES.values():
        (tmp_path / voice['audio']).write_bytes(wav_bytes(seconds=2))
    return tmp_path


def options(assets, **overrides):
    return TTSOptions(url='http://tts.example:9100', assets_dir=assets, **overrides)


def test_voice_files_and_transcriptions_are_paired(assets):
    service = TTSService(options(assets))
    for lang, voice in DEFAULT_VOICES.items():
        assert service.voices[lang]['ref_text'] == voice['text']
        assert base64.b64decode(service.voices[lang]['ref_audio_b64']) == (assets / voice['audio']).read_bytes()
    info = service.public_config()
    assert info['languages'] == ['en', 'zh'] and info['enabled']
    assert info['max_concurrent'] == 2
    assert 'audio' not in json.dumps(info) and 'tts.example' not in json.dumps(info)


def test_custom_voice_json_and_missing_files(assets):
    config = assets / 'voices.json'
    config.write_text(json.dumps({'en': {'audio':'minimax_en_1_f.wav', 'text':'Custom reference.'}}))
    service = TTSService(TTSOptions(url='http://tts.example', voices_config=config))
    assert list(service.voices) == ['en']
    assert service.voices['en']['ref_text'] == 'Custom reference.'
    config.write_text('{}')
    assert not TTSService(TTSOptions(url='http://tts.example', voices_config=config)).public_config()['enabled']
    service = TTSService(options(assets / 'missing'))
    assert service.public_config()['message'] and not service.voices


@pytest.mark.parametrize('kwargs', [
    {'url':'file:///tmp/voice'}, {'url':'http://a@host'}, {'url':'http://host?q=x'},
    {'max_text_chars':0}, {'max_concurrent':0}, {'timeout_seconds':float('nan')},
])
def test_bad_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        TTSOptions(**kwargs).validate()


def test_http_payload_wav_authority_and_explicit_false(assets):
    async def run():
        captured = []
        def handle(request):
            captured.append(json.loads(request.content))
            assert str(request.url) == 'http://tts.example:9100/api/tts/clone'
            return httpx.Response(200, json={'audio_b64':base64.b64encode(wav_bytes()).decode(),
                                            'sample_rate':999, 'duration_s':999})
        service = TTSService(options(assets, x_vector_only=False))
        service.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        try:
            diagnostics = {}
            raw, rate, duration = await service.synthesize('Hello there.', 'en', diagnostics=diagnostics)
            assert (raw, rate, duration) == (wav_bytes(), 24000, .1)
            assert captured[0]['x_vector_only'] is False
            assert captured[0]['language'] == 'en'
            assert captured[0]['ref_text'] == DEFAULT_VOICES['en']['text']
            assert captured[0]['request_id'] and service.active == 0
            assert diagnostics['request_id'] == captured[0]['request_id']
            assert 0 <= diagnostics['upstream_headers_seconds'] <= diagnostics['upstream_body_seconds']
            assert diagnostics['elapsed_seconds'] >= diagnostics['upstream_body_seconds']
            assert diagnostics['decode_seconds'] >= 0 and diagnostics['upstream_status'] == 200
        finally:
            await service.close()
    asyncio.run(run())


def test_busy_timeout_and_cancellation_release_admission(assets):
    async def run():
        entered = asyncio.Event()
        async def handle(request):
            entered.set()
            await asyncio.sleep(10)
        service = TTSService(options(assets, max_concurrent=1, timeout_seconds=.03))
        service.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        try:
            first = asyncio.create_task(service.synthesize('Hello.', 'en'))
            await entered.wait()
            with pytest.raises(TTSError, match='服务忙'):
                await service.synthesize('第二条', 'zh')
            with pytest.raises(TTSError) as error:
                await first
            assert error.value.code == 'tts_timeout' and service.active == 0
            entered.clear()
            second = asyncio.create_task(service.synthesize('Hello again.', 'en'))
            await entered.wait()
            second.cancel()
            with pytest.raises(asyncio.CancelledError):
                await second
            assert service.active == 0
        finally:
            await service.close()
    asyncio.run(run())


def test_two_upstream_requests_can_run_concurrently(assets):
    async def run():
        both_entered = asyncio.Event()
        release = asyncio.Event()
        entered = 0

        async def handle(request):
            nonlocal entered
            entered += 1
            if entered == 2:
                both_entered.set()
            await release.wait()
            return httpx.Response(200, json={'audio_b64':base64.b64encode(wav_bytes()).decode()})

        service = TTSService(options(assets, max_concurrent=2))
        service.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        requests = [asyncio.create_task(service.synthesize(text, 'en'))
                    for text in ('Segment B.', 'Segment C.')]
        try:
            await asyncio.wait_for(both_entered.wait(), 1)
            assert service.active == 2
            with pytest.raises(TTSError) as error:
                await service.synthesize('Segment D.', 'en')
            assert error.value.status == 503 and error.value.code == 'tts_busy'
            assert entered == 2 and service.active == 2
            release.set()
            assert len(await asyncio.gather(*requests)) == 2
            assert service.active == 0
        finally:
            release.set()
            await asyncio.gather(*requests, return_exceptions=True)
            await service.close()
    asyncio.run(run())


@pytest.mark.parametrize('data', [{}, {'audio_b64':'not-base64'}, {'audio_b64':12},
    {'audio_b64':base64.b64encode(b'not a wave').decode()},
    {'audio_b64':base64.b64encode(wav_bytes()[:-20]).decode()}])
def test_bad_audio_not_sent_to_browser(data):
    with pytest.raises(TTSError) as error:
        decode_audio(data)
    assert error.value.code == 'tts_audio'


def test_route_auth_contract_and_text_translation_survive_tts_failure(assets, monkeypatch):
    captured = []
    async def start(self):
        def handle(request):
            payload = json.loads(request.content)
            captured.append(payload)
            assert 'x_vector_only' not in payload
            if payload['text'] == 'Fail.':
                return httpx.Response(503, json={'error': {'code':'ENGINE_NOT_READY'}})
            return httpx.Response(200, json={'audio_b64':base64.b64encode(wav_bytes()).decode()})
        self.client = httpx.AsyncClient(transport=httpx.MockTransport(handle), trust_env=False)
    monkeypatch.setattr(TTSService, 'start', start)
    settings = Settings(backend='mock', log_dir=None, tts=options(assets), token='test-token')
    with TestClient(create_app(settings)) as client:
        headers = {'Authorization':'Bearer test-token'}
        payload = {'text':'Hello there.', 'language':'en'}
        assert client.post('/api/tts/synthesize', json=payload).status_code == 401
        assert client.post('/api/tts/synthesize', json=payload,
                           headers={**headers, 'Origin':'https://another.example'}).status_code == 403
        for bad in ({'text':'…', 'language':'zh'}, {'text':'<|WAIT|>', 'language':'en'},
                    {'text':'x'*201, 'language':'en'}, {'text':'hello', 'language':'ja'},
                    {**payload, 'url':'http://other.example'}, []):
            assert client.post('/api/tts/synthesize', json=bad, headers=headers).status_code == 400
        assert not captured
        response = client.post('/api/tts/synthesize', json=payload, headers=headers)
        assert response.status_code == 200 and response.content == wav_bytes()
        assert response.headers['content-type'] == 'audio/wav'
        assert response.headers['cache-control'] == 'no-store'
        assert response.headers['x-audio-sample-rate'] == '24000'
        assert response.headers['x-tts-request-id'] == captured[-1]['request_id']
        assert float(response.headers['x-tts-service-seconds']) >= 0
        assert 0 <= float(response.headers['x-tts-upstream-headers-seconds']) <= float(response.headers['x-tts-upstream-body-seconds'])
        assert float(response.headers['x-tts-decode-seconds']) >= 0
        assert captured[-1]['language'] == 'en'
        response = client.post('/api/tts/synthesize', json={'text':'你好。', 'language':'zh'}, headers=headers)
        assert response.status_code == 200
        assert captured[-1]['ref_text'] == DEFAULT_VOICES['zh']['text']
        failed = client.post('/api/tts/synthesize', json={'text':'Fail.', 'language':'en'}, headers=headers)
        assert failed.status_code == 502 and failed.json()['detail']['code'] == 'tts_upstream'
        assert failed.headers['x-tts-request-id'] == captured[-1]['request_id']
        assert float(failed.headers['x-tts-service-seconds']) >= 0
        assert client.get('/api/config').json()['status'] == 'ready'
        assert client.post('/api/tts/synthesize', content=b'x'*8193, headers=headers).status_code == 413


def test_unconfigured_or_bad_tts_does_not_block_text_web(assets):
    for tts in (TTSOptions(), options(assets / 'missing')):
        with TestClient(create_app(Settings(backend='mock', log_dir=None, tts=tts))) as client:
            config = client.get('/api/config').json()
            assert config['status'] == 'ready' and not config['tts']['enabled']
            response = client.post('/api/tts/synthesize', json={'text':'你好', 'language':'zh'})
            assert response.status_code == 503
            assert response.json()['detail']['code'] == 'tts_unavailable'
            assert len(response.headers['x-tts-request-id']) == 32
            assert float(response.headers['x-tts-service-seconds']) >= 0


def test_http_disconnect_cancels_synthesis_and_releases_slot(assets, monkeypatch):
    """Exercise an ASGI disconnect through the route and HTTP middleware."""
    async def run():
        entered = asyncio.Event()
        upstream_cancelled = asyncio.Event()
        disconnect = asyncio.Event()

        async def start(self):
            async def handle(request):
                if json.loads(request.content)['text'] == 'Wait for disconnect.':
                    entered.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        upstream_cancelled.set()
                return httpx.Response(200, json={'audio_b64':base64.b64encode(wav_bytes()).decode()})
            self.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))

        monkeypatch.setattr(TTSService, 'start', start)
        app = create_app(Settings(backend='mock', log_dir=None,
                                  tts=options(assets, max_concurrent=1, timeout_seconds=30)))
        body = json.dumps({'text':'Wait for disconnect.', 'language':'en'}).encode()
        scope = {'type':'http', 'asgi':{'version':'3.0', 'spec_version':'2.3'},
                 'http_version':'1.1', 'method':'POST', 'scheme':'http',
                 'path':'/api/tts/synthesize', 'raw_path':b'/api/tts/synthesize',
                 'query_string':b'', 'root_path':'',
                 'headers':[(b'host', b'testserver'), (b'content-type', b'application/json')],
                 'client':('127.0.0.1', 1234), 'server':('testserver', 80)}
        sent_body = False

        async def receive():
            nonlocal sent_body
            if not sent_body:
                sent_body = True
                return {'type':'http.request', 'body':body, 'more_body':False}
            await disconnect.wait()
            return {'type':'http.disconnect'}

        messages = []
        async def send(message):
            messages.append(message)

        async with app.router.lifespan_context(app):
            request_task = asyncio.create_task(app(scope, receive, send))
            try:
                await asyncio.wait_for(entered.wait(), 1)
                assert app.state.tts.active == 1
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                             base_url='http://testserver') as client:
                    busy = await client.post('/api/tts/synthesize', json={'text':'Another request.', 'language':'en'})
                    assert busy.status_code == 503 and busy.json()['detail']['code'] == 'tts_busy'
                    assert len(busy.headers['x-tts-request-id']) == 32
                    disconnect.set()
                    await asyncio.wait_for(request_task, 1)
                    assert upstream_cancelled.is_set() and app.state.tts.active == 0
                    resumed = await client.post('/api/tts/synthesize', json={'text':'Slot is available.', 'language':'en'})
                    assert resumed.status_code == 200 and app.state.tts.active == 0
            finally:
                disconnect.set()
                if not request_task.done():
                    request_task.cancel()
                await asyncio.gather(request_task, return_exceptions=True)
    asyncio.run(run())
