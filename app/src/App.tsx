import { useState } from 'react';
import { TopBar } from './components/top-bar';
import { SoloView } from './components/solo-view';
import { CompareView } from './components/compare-view';
import { useModelWorker } from './hooks/use-model-worker';

type Mode = 'solo' | 'compare';

export default function App() {
  const [mode, setMode] = useState<Mode>('solo');
  const modelA = useModelWorker();

  return (
    <div
      className="min-h-screen"
      style={{ background: 'var(--surface-0)', color: 'var(--text-primary)' }}
    >
      <TopBar mode={mode} onModeChange={setMode} />
      <main className="p-4">
        {mode === 'solo' ? (
          <SoloView workerHandle={modelA} />
        ) : (
          <CompareView workerHandleA={modelA} />
        )}
      </main>
    </div>
  );
}
