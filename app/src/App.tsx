import { useState } from 'react';
import { TopBar } from './components/top-bar';
import { SoloView } from './components/solo-view';
import { CompareView } from './components/compare-view';
import { useModelWorker } from './hooks/use-model-worker';
import type { Mode } from './lib/types';

export function App() {
  const [mode, setMode] = useState<Mode>('solo');
  const modelA = useModelWorker();

  return (
    <div className="min-h-screen bg-surface-0 text-text-primary">
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
