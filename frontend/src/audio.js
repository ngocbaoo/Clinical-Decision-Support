// Mic capture → 16-bit PCM WAV Blob.
//
// We deliberately do NOT use MediaRecorder: it emits webm/ogg(opus), which the backend's
// soundfile reader can't decode without ffmpeg. Capturing raw PCM via the Web Audio API and
// encoding a WAV ourselves keeps the backend dependency-free (soundfile reads WAV natively) and
// the codec predictable. The backend resamples to 16 kHz, so we send at the device's native rate.

export function createRecorder() {
  let ctx, source, processor, stream;
  let sampleRate = 16000;
  const buffers = [];

  async function start() {
    stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    ctx = new (window.AudioContext || window.webkitAudioContext)();
    sampleRate = ctx.sampleRate;
    source = ctx.createMediaStreamSource(stream);
    // ScriptProcessorNode is deprecated but universally supported and simplest for a demo capture.
    processor = ctx.createScriptProcessor(4096, 1, 1);
    processor.onaudioprocess = (e) =>
      buffers.push(new Float32Array(e.inputBuffer.getChannelData(0)));
    source.connect(processor);
    processor.connect(ctx.destination);
  }

  async function stop() {
    processor?.disconnect();
    source?.disconnect();
    stream?.getTracks().forEach((t) => t.stop());
    await ctx?.close();

    const length = buffers.reduce((n, b) => n + b.length, 0);
    const pcm = new Float32Array(length);
    let off = 0;
    for (const b of buffers) { pcm.set(b, off); off += b.length; }
    return encodeWav(pcm, sampleRate);
  }

  return { start, stop };
}

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeStr = (off, s) => {
    for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i));
  };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);     // fmt chunk size
  view.setUint16(20, 1, true);      // PCM
  view.setUint16(22, 1, true);      // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);  // byte rate (mono, 16-bit)
  view.setUint16(32, 2, true);      // block align
  view.setUint16(34, 16, true);     // bits per sample
  writeStr(36, "data");
  view.setUint32(40, samples.length * 2, true);

  let off = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    off += 2;
  }
  return new Blob([view], { type: "audio/wav" });
}
