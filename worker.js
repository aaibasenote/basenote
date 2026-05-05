import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3/dist/transformers.min.js';

let pipe = null;

async function loadModel(payload, id) {
  try {
    self.postMessage({ type: 'status', kind: 'warn', text: '라이브러리 로딩 중...' });
    env.allowRemoteModels = true;
    env.backends.onnx.wasm.numThreads = Math.min(navigator.hardwareConcurrency || 4, 4);
    self.postMessage({ type: 'status', kind: 'warn', text: '모델 다운로드 중...' });
    pipe = await pipeline('text-generation', payload.modelId, {
      dtype: 'q4',
      device: 'wasm',
      progress_callback: (() => {
        const fileLoaded = {};
        const fileTotal  = {};
        return (p) => {
          if (!p.name) {
            if (p.status === 'loading')
              self.postMessage({ type: 'status', kind: 'warn', text: '모델 초기화 중...' });
            return;
          }
          if (p.status === 'initiate') {
            fileLoaded[p.name] = 0;
            fileTotal[p.name]  = p.total || 0;
          } else if (p.status === 'progress') {
            if (typeof p.loaded === 'number') fileLoaded[p.name] = p.loaded;
            if (typeof p.total  === 'number' && p.total > 0) fileTotal[p.name] = p.total;
          } else if (p.status === 'done') {
            fileLoaded[p.name] = fileTotal[p.name] || fileLoaded[p.name] || 0;
          } else { return; }
          const totalBytes  = Object.values(fileTotal).reduce((a, b) => a + b, 0);
          const loadedBytes = Object.values(fileLoaded).reduce((a, b) => a + b, 0);
          if (totalBytes <= 0) {
            self.postMessage({ type: 'status', kind: 'warn', text: '다운로드 중...' });
            return;
          }
          const pct = Math.min(99, Math.round(loadedBytes / totalBytes * 100));
          const loadedMB = (loadedBytes / 1024 / 1024).toFixed(0);
          const totalMB  = (totalBytes  / 1024 / 1024).toFixed(0);
          self.postMessage({ type: 'status', kind: 'warn', text: `다운로드 ${pct}% (${loadedMB}/${totalMB}MB)` });
        };
      })(),
    });
    self.postMessage({ type: 'status', kind: 'ok', text: '모델 준비 완료' });
    self.postMessage({ type: 'load_ok', id });
  } catch (err) {
    self.postMessage({ type: 'load_err', id, error: (err && (err.message || String(err))) || '모델 로딩 실패' });
  }
}

async function runInfer(payload, id) {
  if (!pipe) {
    self.postMessage({ type: 'infer_err', id, error: '모델이 아직 준비되지 않았어요.' });
    return;
  }
  try {
    const output = await pipe(payload.messages, {
      max_new_tokens: payload.maxTokens || 256,
      temperature: payload.temperature != null ? payload.temperature : 0.4,
      do_sample: true,
    });
    self.postMessage({ type: 'infer_ok', id, output });
  } catch (err) {
    self.postMessage({ type: 'infer_err', id, error: (err && (err.message || String(err))) || '추론 실패' });
  }
}

self.onmessage = (e) => {
  const { type, payload, id } = e.data;
  if (type === 'load')  loadModel(payload, id);
  if (type === 'infer') runInfer(payload, id);
};
