/**
 * Version constants (injected from versions.yaml via Vite) and semver helper.
 */

/** Client version from versions.yaml */
export const CLIENT_VERSION = __APP_VERSION__;

/** Protocol version from versions.yaml - must match server exactly */
export const CLIENT_PROTOCOL_VERSION = __PROTOCOL_VERSION__;

/**
 * Compare two semver versions.
 * Returns: -1 if a < b, 0 if a === b, 1 if a > b
 */
export function compareSemver(a: string, b: string): number {
  const partsA = a.split('.').map(Number);
  const partsB = b.split('.').map(Number);

  for (let i = 0; i < 3; i++) {
    const numA = partsA[i] || 0;
    const numB = partsB[i] || 0;
    if (numA < numB) return -1;
    if (numA > numB) return 1;
  }
  return 0;
}
