import type { AgeAnalysis } from "./api";
import { HISTORY_STORAGE_KEY, type HistorySketch, type HistorySnapshot, deserializeHistory, historySessions, mergeHistory } from "./history";

const DB_NAME = "ai-police-sketch-history";
const DB_VERSION = 1;
const STORE_NAME = "sessions";
const MAX_SESSIONS = 10;

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    if (typeof indexedDB === "undefined") return reject(new Error("IndexedDB is unavailable."));
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = () => request.result.createObjectStore(STORE_NAME, { keyPath: "id" });
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error("Unable to open history storage."));
  });
}

function requestResult<T>(request: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error || new Error("History storage request failed."));
  });
}

export async function readHistory(): Promise<HistorySnapshot> {
  const db = await openDb();
  try {
    const records = await requestResult<Array<{ id: string; sketches: HistorySketch[]; ageAnalyses: Record<string, AgeAnalysis> }>>(db.transaction(STORE_NAME).objectStore(STORE_NAME).getAll());
    const sketches = records.flatMap((record) => record.sketches || []);
    const ageAnalyses = Object.assign({}, ...records.map((record) => record.ageAnalyses || {}));
    return { sketches: mergeHistory([], sketches), ageAnalyses };
  } finally { db.close(); }
}

export async function writeHistory(snapshot: HistorySnapshot): Promise<void> {
  const db = await openDb();
  try {
    const tx = db.transaction(STORE_NAME, "readwrite");
    const store = tx.objectStore(STORE_NAME);
    store.clear();
    const sessions = historySessions(snapshot.sketches);
    for (const session of sessions.slice(0, MAX_SESSIONS)) {
      const keys = new Set(session.versions.map((item) => `${item.id}:${item.version}`));
      const ageAnalyses = Object.fromEntries(Object.entries(snapshot.ageAnalyses).filter(([key]) => keys.has(key)));
      store.put({ id: session.id, updated_at: session.updated_at, sketches: session.versions, ageAnalyses });
    }
    await new Promise<void>((resolve, reject) => {
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error || new Error("Unable to save sketch history."));
      tx.onabort = () => reject(tx.error || new Error("History save was aborted."));
    });
  } finally { db.close(); }
}

export async function migrateLocalHistory(): Promise<HistorySnapshot> {
  const existing = await readHistory();
  let legacy: HistorySnapshot = { sketches: [], ageAnalyses: {} };
  try { legacy = deserializeHistory(localStorage.getItem(HISTORY_STORAGE_KEY)); } catch { /* malformed or inaccessible legacy storage is ignored */ }
  if (!existing.sketches.length && legacy.sketches.length) {
    const migrated = { sketches: legacy.sketches, ageAnalyses: legacy.ageAnalyses };
    await writeHistory(migrated);
    try { localStorage.removeItem(HISTORY_STORAGE_KEY); } catch { /* IndexedDB copy is safe; inaccessible localStorage can be retried next visit */ }
    return readHistory();
  }
  // The old payload may contain large image data even when IndexedDB already has records.
  if (existing.sketches.length) {
    const merged = { sketches: mergeHistory(existing.sketches, legacy.sketches), ageAnalyses: { ...legacy.ageAnalyses, ...existing.ageAnalyses } };
    if (legacy.sketches.length) await writeHistory(merged);
    try { localStorage.removeItem(HISTORY_STORAGE_KEY); } catch { /* IndexedDB copy is safe; inaccessible localStorage can be retried next visit */ }
    return readHistory();
  }
  try { localStorage.removeItem(HISTORY_STORAGE_KEY); } catch { /* inaccessible legacy storage is ignored */ }
  return existing;
}
