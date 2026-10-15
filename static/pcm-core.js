/** CPU-only streaming downsampler. Input positions remain continuous across blocks. */
export class StreamingResampler {
  constructor(inputRate, outputRate = 16000) {
    if (!(inputRate >= outputRate && outputRate > 0)) throw new Error('Unsupported audio sample rate');
    this.inputRate = inputRate;
    this.outputRate = outputRate;
    this.step = inputRate / outputRate;
    this.index = -1;
    this.outputCount = 0;
    this.previous = 0;
    this.finished = false;
    // Anti-alias before downsampling. A causal 63-tap Hann-windowed low-pass
    // adds < 2 ms of fixed filter delay at common browser sample rates.
    this.coefficients = null;
    if (inputRate > outputRate) {
      const n = 63, cutoff = 0.46 * outputRate / inputRate;
      this.coefficients = new Float64Array(n);
      let sum = 0;
      for (let i = 0; i < n; i++) {
        const x = i - (n - 1) / 2;
        const sinc = x === 0 ? 2 * cutoff : Math.sin(2 * Math.PI * cutoff * x) / (Math.PI * x);
        const window = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (n - 1));
        this.coefficients[i] = sinc * window;
        sum += this.coefficients[i];
      }
      for (let i = 0; i < n; i++) this.coefficients[i] /= sum;
      this.ring = new Float64Array(n);
      this.cursor = 0;
    }
  }
  filter(value) {
    if (!this.coefficients) return value;
    this.ring[this.cursor] = value;
    let result = 0, pos = this.cursor;
    for (let i = 0; i < this.coefficients.length; i++) {
      result += this.coefficients[i] * this.ring[pos];
      if (--pos < 0) pos = this.ring.length - 1;
    }
    this.cursor = (this.cursor + 1) % this.ring.length;
    return result;
  }
  push(input) {
    if (this.finished) throw new Error('Resampler already finished');
    const output = [];
    for (let i = 0; i < input.length; i++) {
      const value = this.filter(Number.isFinite(input[i]) ? input[i] : 0);
      this.index++;
      const position = this.outputCount * this.step;
      if (position <= this.index + 1e-8) {
        const fraction = this.index === 0 ? 1 : Math.max(0, Math.min(1, position - (this.index - 1)));
        output.push(this.previous + (value - this.previous) * fraction);
        this.outputCount++;
      }
      this.previous = value;
    }
    return output;
  }
  flush() {
    if (this.finished) return [];
    this.finished = true;
    const output = [];
    // At most one output position remains between the final input sample and
    // its exclusive end. Hold that last value; do not invent extra duration.
    const total = Math.ceil((this.index + 1) * this.outputRate / this.inputRate - 1e-9);
    while (this.outputCount < total) { output.push(this.previous); this.outputCount++; }
    return output;
  }
}

export function floatToPCM16(value) {
  const bounded = Math.max(-1, Math.min(1, Number.isFinite(value) ? value : 0));
  return Math.round(bounded * (bounded < 0 ? 32768 : 32767));
}

export function makeAudioPacket(sequence, startSample, samples) {
  if (!Number.isInteger(sequence) || sequence < 0 || sequence > 0xffffffff || !Number.isInteger(startSample) || startSample < 0 || startSample > 0xffffffff) throw new Error('Audio sequence exceeds protocol bounds');
  const bytes = new ArrayBuffer(8 + samples.length * 2), view = new DataView(bytes);
  view.setUint32(0, sequence, true);
  view.setUint32(4, startSample, true);
  for (let i = 0; i < samples.length; i++) view.setInt16(8 + i * 2, samples[i], true);
  return bytes;
}
