/**
 * Pure table sort/format helpers — kept out of the component so the logic is
 * unit-testable without the DOM and memoizable.
 */

export type SortDir = 'asc' | 'desc';
export interface TableSort {
  col: number;
  dir: SortDir;
}

const NUMERIC_RE = /^-?\$?[\d,]+(\.\d+)?%?$/;

function toNumber(cell: string): number | null {
  const s = cell.trim();
  if (!s) return null;
  if (!NUMERIC_RE.test(s)) return null;
  const n = Number(s.replace(/[$,%\s]/g, ''));
  return Number.isFinite(n) ? n : null;
}

/** A column is numeric when every non-blank cell parses as a number. */
export function isNumericColumn(rows: string[][], col: number): boolean {
  let sawValue = false;
  for (const row of rows) {
    const cell = (row[col] ?? '').trim();
    if (!cell) continue;
    if (toNumber(cell) === null) return false;
    sawValue = true;
  }
  return sawValue;
}

/**
 * Return a new array of rows sorted by `col`/`dir`. Numeric columns sort
 * numerically, others lexically (case-insensitive). Blank cells always sink to
 * the bottom regardless of direction. Stable.
 */
export function sortRows(rows: string[][], col: number, dir: SortDir): string[][] {
  const numeric = isNumericColumn(rows, col);
  const factor = dir === 'asc' ? 1 : -1;
  return rows
    .map((row, i) => ({ row, i }))
    .sort((a, b) => {
      const av = (a.row[col] ?? '').trim();
      const bv = (b.row[col] ?? '').trim();
      if (!av && !bv) return a.i - b.i;
      if (!av) return 1; // blanks last
      if (!bv) return -1;
      let cmp: number;
      if (numeric) {
        cmp = (toNumber(av) ?? 0) - (toNumber(bv) ?? 0);
      } else {
        cmp = av.localeCompare(bv, undefined, { sensitivity: 'base' });
      }
      return cmp !== 0 ? cmp * factor : a.i - b.i;
    })
    .map(({ row }) => row);
}
