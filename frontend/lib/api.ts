export type Sketch = { id: string; version: number; description: string; refinement_instruction?: string | null; image_url: string | null; status: string };
const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const headers = options?.body instanceof FormData ? options?.headers : { "Content-Type": "application/json", ...options?.headers };
  const response = await fetch(`${baseUrl}${path}`, { ...options, headers });
  if (!response.ok) { const body = await response.json().catch(() => null); throw new Error(body?.detail || "The sketch service is unavailable."); }
  return response.json();
}
export const api = {
  generate: (description: string) => request<Sketch>("/api/sketches/generate", { method: "POST", body: JSON.stringify({ description }) }),
  generateFromReference: (file: File) => { const form = new FormData(); form.append("file", file); return request<Sketch>("/api/sketches/generate-from-reference", { method: "POST", body: form }); },
  refine: (id: string, instruction: string) => request<Sketch>(`/api/sketches/${id}/refine`, { method: "POST", body: JSON.stringify({ instruction }) }),
};
