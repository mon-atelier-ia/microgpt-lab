import { test, expect } from '@playwright/test';

test.describe('Solo mode', () => {
  test('page loads with TopBar and Solo mode active', async ({ page }) => {
    await page.goto('/');
    await expect(page.getByText('microgpt-lab')).toBeVisible();

    // Solo tab should be present and aria-selected
    const soloTab = page.getByRole('tab', { name: 'Solo' });
    await expect(soloTab).toBeVisible();
    await expect(soloTab).toHaveAttribute('aria-selected', 'true');
  });

  test('train and generate flow', async ({ page }) => {
    await page.goto('/');

    // The train button shows "Entraîner" initially
    const trainBtn = page.getByTestId('train-btn');
    await expect(trainBtn).toBeVisible();
    await expect(trainBtn).toHaveText(/entra[îi]ner/i);

    await trainBtn.click();

    // Button should transition to "Entraînement…" while training is in progress
    await expect(trainBtn).toHaveText(/entraînement/i, { timeout: 10000 });

    // Wait for training to finish — button text reverts to "Entraîner"
    await expect(trainBtn).toHaveText(/entra[îi]ner/i, { timeout: 30000 });

    // Loss curve canvas should be visible after training
    const canvas = page.locator('canvas');
    await expect(canvas).toBeVisible({ timeout: 5000 });

    // Click Generate
    const genBtn = page.getByRole('button', { name: 'Générer' });
    await genBtn.click();

    // Word grid should populate — the list is inside aria-label="Mots générés"
    const wordList = page.locator('[aria-label="Mots générés"] [role="list"]');
    await expect(wordList).toBeVisible({ timeout: 10000 });

    const items = wordList.locator('[role="listitem"]');
    expect(await items.count()).toBeGreaterThan(0);
  });
});
