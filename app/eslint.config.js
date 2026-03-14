import js from '@eslint/js';
import globals from 'globals';
import reactHooks from 'eslint-plugin-react-hooks';
import reactRefresh from 'eslint-plugin-react-refresh';
import sonarjs from 'eslint-plugin-sonarjs';
import tsParser from '@typescript-eslint/parser';
import tsPlugin from '@typescript-eslint/eslint-plugin';
import { defineConfig, globalIgnores } from 'eslint/config';

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      js.configs.recommended,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
    ],
    plugins: {
      sonarjs,
      '@typescript-eslint': tsPlugin,
    },
    languageOptions: {
      ecmaVersion: 2020,
      globals: globals.browser,
      parser: tsParser,
      parserOptions: {
        ecmaVersion: 'latest',
        ecmaFeatures: { jsx: true },
        sourceType: 'module',
      },
    },
    rules: {
      'no-unused-vars': 'off',
      '@typescript-eslint/no-unused-vars': ['error', { varsIgnorePattern: '^[A-Z_]', argsIgnorePattern: '^_' }],
      /* ── Complexity gates ──────────────────────────────── */
      complexity: ['error', { max: 10 }],
      'max-depth': ['error', { max: 4 }],
      'max-lines-per-function': ['error', { max: 100, skipBlankLines: true, skipComments: true }],
      'max-lines': ['error', { max: 300, skipBlankLines: true, skipComments: true }],
      'max-params': ['error', { max: 4 }],
      /* ── SonarJS quality ───────────────────────────────── */
      'sonarjs/cognitive-complexity': ['error', 12],
      'sonarjs/no-duplicate-string': ['error', { threshold: 3 }],
      'sonarjs/no-identical-functions': 'error',
      'sonarjs/no-collapsible-if': 'error',
      'sonarjs/prefer-single-boolean-return': 'error',
    },
  },
  /* Data files are pure string arrays — exempt from max-lines */
  {
    files: ['src/data/**/*.ts'],
    rules: { 'max-lines': 'off' },
  },
]);
