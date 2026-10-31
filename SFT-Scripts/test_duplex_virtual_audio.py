"""Offline virtual-chunk and template-hook tests; no SWIFT/model/GPU required."""
import base64
import importlib.util
import io
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import duplex_audio as audio


@pytest.fixture(autouse=True)
def clear_chunk_cache():
    audio._read_chunk.cache_clear()
    yield
    audio._read_chunk.cache_clear()


def write_source(path, rate=16000, seconds=2, stereo=False):
    # Exact FLOAT source values make sample equality independent of quantization.
    values = ((np.arange(round(rate * seconds)) % 1024) - 512).astype(np.float32) / 1024
    if stereo:
        values = np.stack((values, -values / 2), axis=1)
    sf.write(path, values, rate, subtype="FLOAT")
    return values


def read_memory_wav(data):
    assert isinstance(data, bytes)
    with sf.SoundFile(io.BytesIO(data)) as stream:
        assert stream.samplerate == 16000
        assert stream.channels == 1
        return stream.read(dtype="float32")


@pytest.mark.parametrize("stereo", [False, True])
def test_interval_contains_exact_source_samples(tmp_path, stereo):
    source = tmp_path / "source.wav"
    original = write_source(source, stereo=stereo)
    ref = audio.encode_chunk_ref(source, 0.25, 0.75, 0.5)
    got = read_memory_wav(audio.chunk_wav_bytes(ref))
    expected = original[4000:12000]
    if stereo:
        expected = expected.mean(axis=1, dtype=np.float32)
    np.testing.assert_array_equal(got, expected)


def test_subsecond_tail_is_padded_only_with_silence(tmp_path):
    source = tmp_path / "source.wav"
    original = write_source(source, seconds=1.25)
    ref = audio.encode_chunk_ref(source, 1, 1.25, 1)
    got = read_memory_wav(audio.chunk_wav_bytes(ref))
    assert got.shape == (16000,)
    np.testing.assert_array_equal(got[:4000], original[16000:20000])
    np.testing.assert_array_equal(got[4000:], np.zeros(12000, dtype=np.float32))


@pytest.mark.parametrize("rate", [16000, 44100])
def test_source_reader_seeks_and_never_requests_future_frames(tmp_path, monkeypatch, rate):
    source = tmp_path / "source.wav"
    write_source(source, rate)
    native_soundfile = sf.SoundFile
    operations = []

    class BoundedReader:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def __len__(self):
            return len(self.stream)

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def seek(self, offset, *args, **kwargs):
            operations.append(("seek", offset))
            return self.stream.seek(offset, *args, **kwargs)

        def read(self, frames=-1, *args, **kwargs):
            start = self.stream.tell()
            operations.append(("read", start, frames))
            assert start == round(0.25 * rate)
            assert frames == round(0.75 * rate) - start
            assert start + frames <= round(0.75 * rate)
            return self.stream.read(frames, *args, **kwargs)

    def soundfile(file, *args, **kwargs):
        stream = native_soundfile(file, *args, **kwargs)
        return BoundedReader(stream) if isinstance(file, str) and file == str(source) else stream

    monkeypatch.setattr(sf, "SoundFile", soundfile)
    ref = audio.encode_chunk_ref(source, 0.25, 0.75, 0.5)
    got = read_memory_wav(audio.chunk_wav_bytes(ref))
    assert got.shape == (8000,)
    assert operations == [("seek", round(0.25 * rate)),
                          ("read", round(0.25 * rate), round(0.5 * rate))]


@pytest.mark.parametrize("rate", [16000, 44100])
def test_changing_future_audio_cannot_change_earlier_chunk(tmp_path, rate):
    source = tmp_path / "source.wav"
    original = write_source(source, rate)
    ref = audio.encode_chunk_ref(source, 0.25, 0.75, 0.5)
    before = audio.chunk_wav_bytes(ref)
    # Change every sample from the exclusive end onward, including the boundary
    # neighborhood that would contaminate an acausal full-recording resampler.
    changed = original.copy()
    changed[round(0.75 * rate):] = 0.99
    sf.write(source, changed, rate, subtype="FLOAT")
    # Force actual rereading so a stale cached chunk cannot make this pass.
    audio._read_chunk.cache_clear()
    after = audio.chunk_wav_bytes(ref)
    np.testing.assert_array_equal(read_memory_wav(before), read_memory_wav(after))


def raw_ref(**changes):
    data = {"path": "/source.wav", "start": 0, "end": 1, "input_seconds": 1}
    data.update(changes)
    encoded = base64.urlsafe_b64encode(json.dumps(data).encode("utf-8")).decode("ascii")
    return audio.PREFIX + encoded


@pytest.mark.parametrize("changes", [
    {"start": -1}, {"start": 1, "end": 1}, {"start": 2, "end": 1},
    {"input_seconds": 0}, {"input_seconds": 9}, {"input_seconds": 0.5},
    {"start": True}, {"end": "1"}, {"end": float("nan")},
    {"input_seconds": float("inf")}, {"path": ""},
])
def test_invalid_reference_values_are_rejected(changes):
    with pytest.raises(ValueError):
        audio.decode_chunk_ref(raw_ref(**changes))


@pytest.mark.parametrize("value", ["/full/source.wav", "flexaudio:v1:!", None, "flexaudio:v1:e30="])
def test_non_virtual_or_malformed_reference_is_rejected(value):
    with pytest.raises(ValueError):
        audio.decode_chunk_ref(value)


@pytest.mark.parametrize("start,end,pad", [(0, 2.001, 3), (2, 2.1, 1), (0, 0.001, 0.001)])
def test_out_of_recording_or_unusable_interval_is_rejected(tmp_path, start, end, pad):
    source = tmp_path / "source.wav"
    write_source(source)
    with pytest.raises(ValueError):
        audio.chunk_wav_bytes(audio.encode_chunk_ref(source, start, end, pad))


def test_relative_audio_path_is_rejected():
    with pytest.raises(ValueError, match="absolute"):
        audio.chunk_wav_bytes(audio.encode_chunk_ref("source.wav", 0, 1, 1))


def load_plugin_with_swift_stub(monkeypatch, layout, native_hook):
    """Provide just one SWIFT layout; importing the other must really fail."""
    for name in list(sys.modules):
        if name == "swift" or name.startswith("swift."):
            monkeypatch.delitem(sys.modules, name)
    paths = (["swift", "swift.template", "swift.template.templates"] if layout == "new"
             else ["swift", "swift.llm", "swift.llm.template", "swift.llm.template.template"])
    for name in paths:
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)

    class NativeTemplate:
        sampling_rate = 16000
        _preprocess_inputs = native_hook

    class Meta:
        def __init__(self, template_type, **kwargs):
            self.template_type = template_type
            self.__dict__.update(kwargs)

    registered = []
    prefix = "swift.template" if layout == "new" else "swift.llm.template"
    register_module = ModuleType(prefix + ".register")
    native_meta = Meta("qwen3_omni", template_cls=NativeTemplate,
                       thinking_prefix="<think>\n", suffix=["native-suffix"],
                       prompt=["native-user-wrapper"], settings={"packing": True})
    register_module.TEMPLATE_MAPPING = {"qwen3_omni": native_meta}
    register_module.register_template = registered.append
    monkeypatch.setitem(sys.modules, prefix + ".register", register_module)
    sys.modules[prefix].register_template = registered.append
    qwen_name = prefix + (".templates.qwen" if layout == "new" else ".template.qwen")
    qwen = ModuleType(qwen_name)
    qwen.Qwen3OmniTemplate, qwen.QwenTemplateMeta = NativeTemplate, Meta
    monkeypatch.setitem(sys.modules, qwen_name, qwen)
    spec = importlib.util.spec_from_file_location("_duplex_sft_plugin_test", SCRIPTS / "duplex_sft_plugin.py")
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)
    assert len(registered) == 1
    assert registered[0].template_type == "flex_duplex_qwen3_omni"
    assert registered[0].template_cls is plugin.FlexDuplexQwen3OmniTemplate
    assert native_meta.template_type == "qwen3_omni"
    assert native_meta.template_cls is NativeTemplate
    for key in ("thinking_prefix", "suffix", "prompt", "settings"):
        assert getattr(registered[0], key) == getattr(native_meta, key)
    assert registered[0].settings is not native_meta.settings
    assert registered[0].suffix is not native_meta.suffix
    return plugin


@pytest.mark.parametrize("layout", ["new", "old"])
def test_plugin_passes_only_selected_bytes_to_native_hook_and_creates_no_wavs(tmp_path, monkeypatch, layout):
    source = tmp_path / "complete_source.wav"
    original = write_source(source)
    refs = [audio.encode_chunk_ref(source, 0.25, 0.75, 0.5),
            audio.encode_chunk_ref(source, 1, 1.25, 1)]
    files_before = set(tmp_path.rglob("*"))
    seen = []
    native_write = sf.write

    def memory_only_write(file, *args, **kwargs):
        assert isinstance(file, io.BytesIO), "Virtual materialization wrote an audio file"
        return native_write(file, *args, **kwargs)

    def native_hook(self, inputs):
        assert len(inputs.audios) == 2
        assert all(isinstance(value, bytes) for value in inputs.audios)
        seen.extend(read_memory_wav(value) for value in inputs.audios)
        return "native-hook-reached"

    monkeypatch.setattr(sf, "write", memory_only_write)
    plugin = load_plugin_with_swift_stub(monkeypatch, layout, native_hook)
    inputs = SimpleNamespace(audios=refs)
    assert plugin.FlexDuplexQwen3OmniTemplate()._preprocess_inputs(inputs) == "native-hook-reached"
    np.testing.assert_array_equal(seen[0], original[4000:12000])
    assert seen[1].shape == (16000,)
    np.testing.assert_array_equal(seen[1][:4000], original[16000:20000])
    np.testing.assert_array_equal(seen[1][4000:], np.zeros(12000, dtype=np.float32))
    assert set(tmp_path.rglob("*")) == files_before
    assert all(len(chunk) < len(original) for chunk in seen)


@pytest.mark.parametrize("layout", ["new", "old"])
def test_plugin_rejects_full_recording_mixed_with_virtual_chunk(tmp_path, monkeypatch, layout):
    def native_hook(self, inputs):
        pytest.fail("Mixed recording reached the native processor")

    plugin = load_plugin_with_swift_stub(monkeypatch, layout, native_hook)
    ref = audio.encode_chunk_ref(tmp_path / "unused.wav", 0, 1, 1)
    with pytest.raises(ValueError, match="mix"):
        plugin.FlexDuplexQwen3OmniTemplate()._preprocess_inputs(SimpleNamespace(audios=[ref, "/full.wav"]))


def test_plugin_rejects_incompatible_processor_rate(monkeypatch):
    plugin = load_plugin_with_swift_stub(monkeypatch, "old", lambda self, inputs: pytest.fail("native called"))
    template = plugin.FlexDuplexQwen3OmniTemplate()
    template.sampling_rate = 44100
    with pytest.raises(ValueError, match="16000"):
        template._preprocess_inputs(SimpleNamespace(audios=[raw_ref()]))
