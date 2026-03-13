import type { Preset } from '../lib/types';

export const PRESETS: Preset[] = [
  {
    id: 'prenoms-simple',
    name: 'Prénoms FR (50)',
    description: '50 prénoms français courants',
    load: () => import('./prenoms-simple').then((m) => m.prenomsSimple),
  },
  {
    id: 'prenoms',
    name: 'Prénoms FR (1000)',
    description: '1000 prénoms français (INSEE 2024)',
    load: () => import('./prenoms').then((m) => m.prenoms),
  },
  {
    id: 'prenoms-insee',
    name: 'Prénoms FR (33k)',
    description: '33 235 prénoms français (INSEE, data.gouv.fr)',
    load: () => import('./prenoms-insee').then((m) => m.prenomsInsee),
  },
  {
    id: 'baby-names',
    name: 'Baby Names EN (1000)',
    description: '1000 English baby names',
    load: () => import('./baby-names').then((m) => m.babyNames),
  },
  {
    id: 'names-en',
    name: 'Names EN (8000)',
    description: '8000 English names (Karpathy/makemore)',
    load: () => import('./names-en').then((m) => m.namesEn),
  },
  {
    id: 'dinosaures',
    name: 'Dinosaures (1530)',
    description: '1530 noms de dinosaures',
    load: () => import('./dinosaures').then((m) => m.dinosaures),
  },
  {
    id: 'pokemon-fr',
    name: 'Pokémon FR (1022)',
    description: '1022 noms de Pokémon en français',
    load: () => import('./pokemon-fr').then((m) => m.pokemonFr),
  },
];
