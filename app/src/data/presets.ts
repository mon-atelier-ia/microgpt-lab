import { prenomsSimple } from './prenoms-simple';
import { babyNames } from './baby-names';
import { dinosaures } from './dinosaures';
import { pokemonFr } from './pokemon-fr';

export type Preset = {
  id: string;
  name: string;
  description: string;
  data: string[];
};

export const PRESETS: Preset[] = [
  {
    id: 'prenoms-simple',
    name: 'Prénoms FR',
    description: '~50 prénoms français courants',
    data: prenomsSimple,
  },
  {
    id: 'baby-names',
    name: 'Baby Names EN',
    description: '~4500 English baby names',
    data: babyNames,
  },
  {
    id: 'dinosaures',
    name: 'Dinosaures',
    description: '~100 noms de dinosaures',
    data: dinosaures,
  },
  {
    id: 'pokemon-fr',
    name: 'Pokémon FR',
    description: '~150 noms de Pokémon en français',
    data: pokemonFr,
  },
];
