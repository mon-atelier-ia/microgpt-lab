import { test, expect } from '@playwright/test';

test.describe('Compare mode', () => {
  test('switch to compare shows two panels', async ({ page }) => {
    await page.goto('/');

    // Click Compare tab
    const compareTab = page.getByRole('tab', { name: 'Compare' });
    await compareTab.click();
    await expect(compareTab).toHaveAttribute('aria-selected', 'true');

    // ParamsPanel renders "Modèle A" and "Modèle B" headings (colorVar.toUpperCase())
    await expect(page.getByText('Modèle A')).toBeVisible();
    await expect(page.getByText('Modèle B')).toBeVisible();
  });

  test('compare mode has independent train and generate buttons for each panel', async ({
    page,
  }) => {
    await page.goto('/');

    // Switch to Compare
    await page.getByRole('tab', { name: 'Compare' }).click();
    await expect(page.getByText('Modèle A')).toBeVisible();
    await expect(page.getByText('Modèle B')).toBeVisible();

    // Both panels should have a train button and a "Générer" button
    const trainBtns = page.getByTestId('train-btn');
    await expect(trainBtns).toHaveCount(2);

    const genBtns = page.getByRole('button', { name: 'Générer' });
    await expect(genBtns).toHaveCount(2);
  });

  test('switching back to Solo from Compare shows Solo panel', async ({ page }) => {
    await page.goto('/');

    // Switch to Compare
    await page.getByRole('tab', { name: 'Compare' }).click();
    await expect(page.getByText('Modèle B')).toBeVisible();

    // Switch back to Solo
    const soloTab = page.getByRole('tab', { name: 'Solo' });
    await soloTab.click();
    await expect(soloTab).toHaveAttribute('aria-selected', 'true');

    // In Solo mode only Modèle A is shown, not Modèle B
    await expect(page.getByText('Modèle A')).toBeVisible();
    await expect(page.getByText('Modèle B')).not.toBeVisible();
  });
});
