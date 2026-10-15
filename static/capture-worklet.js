import { StreamingResampler, floatToPCM16 } from './pcm-core.js';

class FlexCaptureProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.active = false;
    this.finalized = false;
    this.resampler = new StreamingResampler(sampleRate, options.processorOptions?.sampleRate || 16000);
    this.frameSize = options.processorOptions?.frameSamples || 1600;
    this.frame = new Int16Array(this.frameSize);
    this.used = 0;
    this.meterCount = 0;
    this.port.onmessage = ({ data }) => {
      if (data.op === 'active' && !this.finalized) this.active = true;
      if (data.op === 'pause') { this.active = false; this.flushFrame(); }
      if (data.op === 'finish') {
        this.active = false;
        if (!this.finalized) { this.append(this.resampler.flush()); this.flushFrame(); this.finalized = true; }
      }
      if (data.op === 'cancel') { this.active = false; this.finalized = true; this.used = 0; }
      this.port.postMessage({ type: 'ack', requestId: data.requestId });
    };
  }
  append(samples) {
    for (const value of samples) {
      this.frame[this.used++] = floatToPCM16(value);
      if (this.used === this.frameSize) this.flushFrame();
    }
  }
  flushFrame() {
    if (!this.used) return;
    const buffer = this.frame.slice(0, this.used).buffer;
    this.port.postMessage({ type: 'pcm', buffer }, [buffer]);
    this.used = 0;
  }
  process(inputs, outputs) {
    const input = inputs[0] || [], output = outputs[0] || [];
    for (let c = 0; c < output.length; c++) {
      const source = input[Math.min(c, input.length - 1)];
      if (source) output[c].set(source); else output[c].fill(0);
    }
    if (!this.active || !input.length || !input[0].length) return true;
    const mono = new Float32Array(input[0].length);
    for (let c = 0; c < input.length; c++) for (let i = 0; i < mono.length; i++) mono[i] += input[c][i] / input.length;
    this.append(this.resampler.push(mono));
    if (++this.meterCount % 20 === 0) {
      let power = 0;
      for (const value of mono) power += value * value;
      this.port.postMessage({ type: 'level', value: Math.sqrt(power / mono.length) });
    }
    return true;
  }
}
registerProcessor('flex-capture', FlexCaptureProcessor);
