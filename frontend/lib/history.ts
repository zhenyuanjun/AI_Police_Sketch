import type { AgeAnalysis, Sketch } from "./api";

export const HISTORY_STORAGE_KEY = "ai-police-sketch-history-v1";

export type HistorySketch = Sketch & { created_at?: string };
export type HistorySnapshot = {
  sketches: HistorySketch[];
  ageAnalyses: Record<string, AgeAnalysis>;
};

const emptySnapshot = (): HistorySnapshot => ({ sketches: [], ageAnalyses: {} });

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function normalizeSketch(value: unknown): HistorySketch | null {
  if (!isObject(value) || typeof value.id !== "string" || !value.id ||
      !Number.isInteger(value.version) || Number(value.version) < 1 ||
      typeof value.description !== "string" ||
      !(typeof value.image_url === "string" || value.image_url === null)) return null;
  return {
    id: value.id,
    version: Number(value.version),
    description: value.description,
    refinement_instruction: typeof value.refinement_instruction === "string" ? value.refinement_instruction : null,
    image_url: value.image_url,
    status: typeof value.status === "string" ? value.status : "saved",
    witness_age: typeof value.witness_age === "number" && Number.isFinite(value.witness_age) ? value.witness_age : null,
    created_at: typeof value.created_at === "string" ? value.created_at : undefined,
  };
}

function normalizeAgeAnalysis(value: unknown): AgeAnalysis | null {
  if (!isObject(value) || typeof value.sketch_id !== "string" ||
      !Number.isInteger(value.version) || typeof value.ml_estimated_age !== "number" ||
      typeof value.out_of_training_range !== "boolean" || typeof value.model !== "string" ||
      typeof value.training_domain !== "string" || typeof value.interpretation !== "string" ||
      !isObject(value.artifact_sha256) || typeof value.artifact_sha256.backbone !== "string" ||
      typeof value.artifact_sha256.age_head !== "string") return null;
  return {
    sketch_id: value.sketch_id,
    version: Number(value.version),
    status: typeof value.status === "string" ? value.status : "available",
    ml_estimated_age: value.ml_estimated_age,
    out_of_training_range: value.out_of_training_range,
    model: value.model,
    training_domain: value.training_domain,
    interpretation: value.interpretation,
    artifact_sha256: {
      backbone: value.artifact_sha256.backbone,
      age_head: value.artifact_sha256.age_head,
    },
    analyzed_at: typeof value.analyzed_at === "string" ? value.analyzed_at : undefined,
  };
}

export function deserializeHistory(raw: string | null): HistorySnapshot {
  if (!raw) return emptySnapshot();
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!isObject(parsed) || parsed.schema_version !== 1 || !Array.isArray(parsed.sketches)) return emptySnapshot();
    const sketches = parsed.sketches.map(normalizeSketch).filter((item): item is HistorySketch => item !== null);
    const ageAnalyses: Record<string, AgeAnalysis> = {};
    if (isObject(parsed.ageAnalyses)) {
      for (const [key, value] of Object.entries(parsed.ageAnalyses)) {
        const analysis = normalizeAgeAnalysis(value);
        if (analysis && key === `${analysis.sketch_id}:${analysis.version}`) ageAnalyses[key] = analysis;
      }
    }
    return { sketches: mergeHistory([], sketches), ageAnalyses };
  } catch {
    return emptySnapshot();
  }
}

export function serializeHistory(snapshot: HistorySnapshot): string {
  return JSON.stringify({
    schema_version: 1,
    sketches: mergeHistory([], snapshot.sketches),
    ageAnalyses: snapshot.ageAnalyses,
  });
}

export function mergeHistory(...groups: HistorySketch[][]): HistorySketch[] {
  const merged = new Map<string, HistorySketch>();
  for (const group of groups) {
    for (const sketch of group) {
      const key = `${sketch.id}:${sketch.version}`;
      const existing = merged.get(key);
      if (!existing) {
        merged.set(key, sketch);
        continue;
      }
      const existingHasData = existing.image_url?.startsWith("data:image/") ?? false;
      const candidateHasData = sketch.image_url?.startsWith("data:image/") ?? false;
      const preferred = candidateHasData && !existingHasData ? sketch : existing;
      const other = preferred === sketch ? existing : sketch;
      merged.set(key, {
        ...other,
        ...preferred,
        description: preferred.description || other.description,
        refinement_instruction: preferred.refinement_instruction || other.refinement_instruction,
        witness_age: preferred.witness_age ?? other.witness_age,
        created_at: preferred.created_at || other.created_at,
      });
    }
  }
  return Array.from(merged.values()).sort((a, b) => {
    const dateOrder = (a.created_at || "").localeCompare(b.created_at || "");
    return dateOrder || a.id.localeCompare(b.id) || a.version - b.version;
  });
}

export function historyForSketch(sketches: HistorySketch[], sketchId: string): HistorySketch[] {
  return sketches.filter((sketch) => sketch.id === sketchId).sort((a, b) => a.version - b.version);
}

