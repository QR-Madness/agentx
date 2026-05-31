/**
 * Browser file download / upload helpers.
 *
 * Deliberately uses only DOM APIs (Blob, anchor download, FileReader) so it
 * works identically in the Tauri webview and in browser mode (`task dev:web`)
 * without pulling in a Tauri fs/dialog plugin. First file I/O in the client —
 * keep it dependency-free.
 */

/** Trigger a download of `data` as a JSON file named `filename`. */
export function downloadJson(data: unknown, filename: string): void {
  const json = typeof data === 'string' ? data : JSON.stringify(data, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  try {
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
  } finally {
    // Revoke on the next tick so the click has a chance to start the download.
    setTimeout(() => URL.revokeObjectURL(url), 0);
  }
}

/** Read a user-selected File as text and parse it as JSON. Rejects on bad JSON. */
export function readJsonFile<T = unknown>(file: File): Promise<T> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error('Failed to read file'));
    reader.onload = () => {
      try {
        resolve(JSON.parse(reader.result as string) as T);
      } catch {
        reject(new Error('File is not valid JSON'));
      }
    };
    reader.readAsText(file);
  });
}

/** A filesystem-safe timestamp suffix, e.g. "20260531T215800Z". */
export function fileTimestamp(d: Date = new Date()): string {
  return d.toISOString().replace(/[-:]/g, '').replace(/\.\d+/, '');
}
