export type Preset = {
  id: string;
  name: string;
  description: string;
  load: () => Promise<string[]>;
};

export const PRESETS: Preset[] = [
  {
    id: 'prenoms-simple',
    name: 'Prénoms FR',
    description: '~50 prénoms français courants',
    load: () => import('./prenoms-simple').then((m) => m.prenomsSimple),
  },
  {
    id: 'baby-names',
    name: 'Baby Names EN',
    description: '~4500 English baby names',
    load: () => import('./baby-names').then((m) => m.babyNames),
  },
  {
    id: 'dinosaures',
    name: 'Dinosaures',
    description: '~100 noms de dinosaures',
    load: () => import('./dinosaures').then((m) => m.dinosaures),
  },
  {
    id: 'pokemon-fr',
    name: 'Pokémon FR',
    description: '~150 noms de Pokémon en français',
    load: () => import('./pokemon-fr').then((m) => m.pokemonFr),
  },
];
