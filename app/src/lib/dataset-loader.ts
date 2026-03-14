import { PRESETS } from '../data/presets';

export async function loadDatasetWords(datasetId: string): Promise<string[] | null> {
  const preset = PRESETS.find((p) => p.id === datasetId);
  if (!preset) return null;
  return preset.load();
}
