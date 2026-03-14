import { PRESETS } from '../data/presets';

export async function loadDatasetText(datasetId: string): Promise<string | null> {
  const preset = PRESETS.find((p) => p.id === datasetId);
  if (!preset) return null;
  const data = await preset.load();
  return data.join('\n');
}
