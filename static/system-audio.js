/**
 * Explicit browser-authorized computer audio capture. This module never reads
 * screen frames, plays captured audio, or sends any stream over the network.
 * getDisplayMedia requires video; keep that track locally until capture stops.
 */
export class SystemAudioError extends Error {
  constructor(code, message, cause) {
    super(message);
    this.name = 'SystemAudioError';
    this.code = code;
    if (cause !== undefined) this.cause = cause;
  }
}

function cancelled(reason) {
  const error = new SystemAudioError('system_audio_cancelled', '已取消电脑声音采集。', reason);
  error.name = 'AbortError';
  return error;
}

function captureError(error, microphone) {
  if (error instanceof SystemAudioError) return error;
  if (microphone) {
    return new SystemAudioError('microphone_capture_failed',
      error?.name === 'NotAllowedError'
        ? '未获得麦克风权限。已停止电脑声音采集；请允许麦克风，或取消勾选“同时收录我的讲话”后重试。'
        : '无法采集麦克风。已停止电脑声音采集；请检查麦克风，或取消勾选“同时收录我的讲话”后重试。',
      error);
  }
  const messages = {
    NotAllowedError: '未获得声音共享权限，或已取消共享。请重新点击开始，在浏览器弹窗中选择来源并勾选共享音频。',
    InvalidStateError: '请回到当前页面，直接点击开始，再选择共享声音；浏览器不允许后台或自动发起共享。',
    NotReadableError: '浏览器无法读取所选共享来源。请检查系统权限，或改为共享正在播放音视频的浏览器标签页。',
    NotFoundError: '浏览器没有找到可共享的来源。请使用桌面版 Chrome / Edge，并打开要播放的音视频。',
    AbortError: '浏览器中断了声音共享请求。请重新点击开始并选择共享来源。',
  };
  return new SystemAudioError('system_audio_capture_failed',
    messages[error?.name] || '无法开始电脑声音采集。请使用桌面版 Chrome / Edge，通过 HTTPS 或 localhost 访问，并重新选择共享来源。',
    error);
}

function stopTracks(stream, stoppedTracks) {
  for (const track of stream?.getTracks?.() || []) {
    if (stoppedTracks.has(track)) continue;
    stoppedTracks.add(track);
    try { track.stop(); } catch { /* One bad track must not leave the others live. */ }
  }
}

/**
 * Call directly from a user-click handler, BEFORE awaiting server readiness.
 * Capture hints never guarantee system audio: validate the actual audio tracks.
 * Optional microphone capture is requested only after explicit opt-in.
 *
 * Cancellation rejects promptly even while the browser picker is pending; the
 * picker itself is browser-owned. Any stream that arrives late is stopped.
 * onEnded(handler) reports external stop/abort; explicit stop() is silent.
 */
export async function requestSystemAudio({
  includeMicrophone = false,
  signal,
  mediaDevices = globalThis.navigator?.mediaDevices,
  isSecureContext = globalThis.isSecureContext !== false,
} = {}) {
  if (signal?.aborted) throw cancelled(signal.reason);
  if (typeof includeMicrophone !== 'boolean') {
    throw new TypeError('includeMicrophone must be an explicit boolean.');
  }
  if (!isSecureContext) {
    throw new SystemAudioError('system_audio_insecure',
      '电脑声音采集需要 HTTPS 或 localhost。请使用当前服务的 HTTPS 地址访问。');
  }
  if (typeof mediaDevices?.getDisplayMedia !== 'function') {
    throw new SystemAudioError('system_audio_unsupported',
      '当前浏览器不支持电脑声音共享。请使用桌面版 Chrome / Edge，并通过 HTTPS 或 localhost 访问。');
  }
  if (includeMicrophone && typeof mediaDevices.getUserMedia !== 'function') {
    throw new SystemAudioError('microphone_unsupported',
      '当前浏览器不能请求麦克风。请更换浏览器，或取消勾选“同时收录我的讲话”后重试。');
  }

  const ownedStreams = new Set(), stoppedTracks = new Set();
  const trackListeners = [], endListeners = new Set();
  let stopped = false, endReason = null, rejectInterrupted;
  const interrupted = new Promise((_, reject) => { rejectInterrupted = reject; });
  // An API implementation can throw synchronously after an abort; retain a
  // rejection handler even when no acquisition race has been constructed yet.
  interrupted.catch(() => {});

  function release(reason = null) {
    if (stopped) return;
    stopped = true;
    endReason = reason;
    signal?.removeEventListener('abort', onAbort);
    for (const [track, handler] of trackListeners) track.removeEventListener('ended', handler);
    trackListeners.length = 0;
    for (const stream of ownedStreams) stopTracks(stream, stoppedTracks);
    ownedStreams.clear();
    if (reason) {
      rejectInterrupted(reason);
      for (const handler of endListeners) {
        try { handler(reason); } catch { /* Cleanup must survive caller errors. */ }
      }
    }
    endListeners.clear();
  }
  function onAbort() { release(cancelled(signal?.reason)); }

  function takeStream(promise, microphone = false) {
    const obtained = Promise.resolve(promise).then(stream => {
      if (stopped) {
        stopTracks(stream, stoppedTracks);
        throw endReason || cancelled();
      }
      ownedStreams.add(stream);
      const audioTracks = stream?.getAudioTracks?.() || [];
      if (!audioTracks.some(track => track.readyState === 'live')) {
        throw new SystemAudioError(microphone ? 'microphone_missing_audio' : 'system_audio_missing_audio',
          microphone
            ? '麦克风没有返回可用音轨。已停止采集，请检查麦克风权限和设备。'
            : '没有收到共享音频。请重新点击开始，选择正在播放的浏览器标签页或整个屏幕，并勾选“共享标签页音频／共享系统音频”。窗口或当前系统可能不提供音频；不会自动改用麦克风。');
      }
      for (const track of stream.getTracks()) {
        const ended = () => release(new SystemAudioError(
          microphone ? 'microphone_ended' : 'system_audio_ended',
          microphone ? '麦克风采集已结束，电脑声音采集也已停止。' : '电脑声音共享已结束。'));
        track.addEventListener('ended', ended);
        trackListeners.push([track, ended]);
        // A video track ending means the shared source was closed even if its
        // audio track has not yet dispatched its own ended event.
        if (track.readyState === 'ended') { ended(); break; }
      }
      if (stopped) throw endReason || cancelled();
      return stream;
    }, error => { throw captureError(error, microphone); });
    return Promise.race([obtained, interrupted]);
  }

  signal?.addEventListener('abort', onAbort, {once:true});
  try {
    // MUST precede every await: browser display-capture permission needs the
    // transient activation from the click that starts this function.
    let displayPromise;
    try {
      displayPromise = mediaDevices.getDisplayMedia({
        video: true,
        audio: {suppressLocalAudioPlayback:false},
        systemAudio: 'include',
        selfBrowserSurface: 'exclude',
        surfaceSwitching: 'exclude',
        monitorTypeSurfaces: 'include',
      });
    } catch (error) { throw captureError(error, false); }
    const displayStream = await takeStream(displayPromise);
    if (stopped) throw endReason || cancelled();

    let microphoneStream = null;
    if (includeMicrophone) {
      let microphonePromise;
      try {
        microphonePromise = mediaDevices.getUserMedia({
          audio: {echoCancellation:true, noiseSuppression:true, autoGainControl:true},
          video: false,
        });
      } catch (error) { throw captureError(error, true); }
      microphoneStream = await takeStream(microphonePromise, true);
    }
    if (stopped) throw endReason || cancelled();

    return {
      displayStream,
      microphoneStream,
      audioStreams: microphoneStream ? [displayStream, microphoneStream] : [displayStream],
      get stopped() { return stopped; },
      stop() { release(); },
      onEnded(handler) {
        if (typeof handler !== 'function') throw new TypeError('onEnded needs a callback.');
        if (endReason) { handler(endReason); return () => {}; }
        if (!stopped) endListeners.add(handler);
        return () => endListeners.delete(handler);
      },
    };
  } catch (error) {
    release();
    throw endReason || error;
  }
}
