export type Sketch = { id: string; version: number; description: string; refinement_instruction?: string | null; image_url: string | null; status: string; witness_age?: number | null };
export type FaceAnalysis = { model: string; embedding_dimension: number; inference_time_ms: number; device: string };
export type Benchmark = { model: string; runtime: string; device: string; embedding_dimension: number; warmup_runs: number; measured_runs: number; mean_latency_ms: number; median_latency_ms: number; min_latency_ms: number; max_latency_ms: number; std_latency_ms: number };
export type PersistedAgeAnalysis = { ml_estimated_age: number; out_of_training_range: boolean; analyzed_at: string; model: string; training_domain: string; interpretation: string; backbone_hash: string; age_head_hash: string };
export type SavedVersion = { version: number; generation_type: string; refinement_instruction?: string | null; image_url: string; age_analysis?: PersistedAgeAnalysis | null };
export type SavedSketch = { sketch_id: string; latest_version: number; generation_type: string; created_at: string; image_url: string; witness_age?: number | null; versions: SavedVersion[] };
export type SketchComparison = { sketch_id: string; version_a: number; version_b: number; model: string; embedding_dimension: number; cosine_similarity: number; cosine_distance: number; refinement_instruction?: string | null };
export type AgeAnalysis = { sketch_id: string; version: number; status: string; ml_estimated_age: number; out_of_training_range: boolean; model: string; training_domain: string; interpretation: string; artifact_sha256: { backbone: string; age_head: string } };
const baseUrl = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");
async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const headers = options?.body instanceof FormData ? options?.headers : { "Content-Type": "application/json", ...options?.headers };
  const response = await fetch(`${baseUrl}${path}`, { ...options, headers });
  if (!response.ok) { const body = await response.json().catch(() => null); throw new Error(body?.detail || "The sketch service is unavailable."); }
  return response.json();
}
export const api = {
  generate: (description: string, witnessAge?: number) => request<Sketch>("/api/sketches/generate", { method: "POST", body: JSON.stringify({ description, ...(witnessAge === undefined ? {} : { witness_age: witnessAge }) }) }),
  refine: (id: string, instruction: string) => request<Sketch>(`/api/sketches/${id}/refine`, { method: "POST", body: JSON.stringify({ instruction }) }),
  analyze: (id: string) => request<FaceAnalysis>(`/api/sketches/${id}/analyze`, { method: "POST" }),
  benchmark: (id: string) => request<Benchmark>(`/api/sketches/${id}/benchmark`, { method: "POST" }),
  saved: () => request<SavedSketch[]>("/api/sketches/saved"),
  compare: (id: string, versionA: number, versionB: number) => request<SketchComparison>(`/api/sketches/${id}/compare?version_a=${versionA}&version_b=${versionB}`),
  analyzeAge: (id: string, version: number) => request<AgeAnalysis>(`/api/sketches/${id}/versions/${version}/age-analysis`, { method: "POST" }),
};
