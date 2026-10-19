// File audio is read on the server. This clock only releases the prefix that
// has actually played; delayed browser callbacks catch up without losing PCM.
export class FilePlaybackClock {
  constructor({offset, duration, browserDuration}) {
    if (![offset,duration,browserDuration].every(Number.isFinite) || offset < 0 || duration <= 0 || browserDuration <= 0 || offset >= duration) {
      throw new Error('媒体时长或开始位置无效，请重新选择文件。');
    }
    // Container duration may differ slightly from the decoded audio duration.
    // Larger differences must not silently truncate or invent an audio tail.
    this.tolerance = 0.5;
    if (Math.abs(duration-browserDuration) > this.tolerance) {
      throw new Error(`浏览器播放时长（${browserDuration.toFixed(2)} 秒）与服务器音轨时长（${duration.toFixed(2)} 秒）不一致。请将文件转为 WAV 或音视频等长的 MP4 后重试。`);
    }
    this.offset = offset;
    this.duration = duration;
    this.browserDuration = browserDuration;
    this.position = offset;
  }

  read(currentTime, ended = false) {
    if (!Number.isFinite(currentTime)) throw new Error('无法读取当前播放位置。');
    let next = Math.min(this.duration, Math.max(this.offset, currentTime));
    // Only natural EOF, never a manual finish/pause, may reconcile the small
    // container/audio duration discrepancy to the exact server sample count.
    if (ended && Math.abs(currentTime-this.browserDuration) <= this.tolerance) next = this.duration;
    this.position = Math.max(this.position, next);
    return this.position;
  }
}
