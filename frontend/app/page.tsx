"use client";

import { FormEvent, useEffect, useState } from "react";
import Face3DViewer from "../components/Face3DViewer";
import { AgeAnalysis, api, Benchmark, FaceAnalysis, SavedSketch, Sketch, SketchComparison } from "../lib/api";
import { deserializeHistory, HISTORY_STORAGE_KEY, HistorySketch, historyForSketch, mergeHistory, serializeHistory } from "../lib/history";

const example = "Male, around 30 years old, oval face, short black hair, thick eyebrows, narrow eyes, medium-sized nose, thin lips.";

function MockPortraitFallback({ sketch }: { sketch: Sketch | null }) {
  return <div className="grain relative flex min-h-[480px] items-center justify-center overflow-hidden border border-[#d6d0c7] bg-[#ece8e1] shadow-inner"><div className="absolute left-5 top-5 text-[10px] uppercase tracking-[0.22em] text-[#8d867d]">Mock rendering / {sketch ? `v${sketch.version}` : "awaiting input"}</div>{sketch ? <div className="relative h-[360px] w-[260px] rounded-[48%_48%_45%_45%] border-[3px] border-[#6f6a63] bg-[#ded9d1] shadow-[inset_0_0_0_14px_#e7e2da]"><div className="absolute left-8 right-8 top-20 flex justify-between"><i className="h-3 w-14 rounded-full bg-[#625e59]" /><i className="h-3 w-14 rounded-full bg-[#625e59]" /></div><div className="absolute left-[54px] top-[115px] h-5 w-5 rounded-full bg-[#45433f] shadow-[124px_0_#45433f]" /><div className="absolute left-[112px] top-[124px] h-20 w-9 border-l-2 border-r-2 border-[#77716a]" /><div className="absolute bottom-[85px] left-[89px] h-3 w-20 rounded-[50%] border-b-2 border-[#65605b]" /><div className="absolute -bottom-6 left-[-28px] h-20 w-[316px] rounded-[50%] border-t-[3px] border-[#77716a]" /></div> : <div className="text-center"><div className="mx-auto mb-5 h-16 w-16 rounded-full border border-[#b8b0a5]" /><p className="font-serif text-xl text-[#5f5a53]">No sketch generated yet</p><p className="mt-2 text-sm text-[#888077]">Enter a description to begin.</p></div>}</div>;
}

function MockPortrait({ sketch }: { sketch: Sketch | null }) {
  if (sketch?.image_url) return <div className="grain relative flex min-h-[480px] items-center justify-center overflow-hidden border border-[#d6d0c7] bg-[#ece8e1] shadow-inner"><div className="absolute left-5 top-5 text-[10px] uppercase tracking-[0.22em] text-[#8d867d]">{sketch.status === "saved" ? "Saved rendering" : "OpenAI rendering"} / v{sketch.version}</div><img src={sketch.image_url} alt="Generated facial sketch" className="max-h-[440px] max-w-full object-contain" /></div>;
  return <MockPortraitFallback sketch={sketch} />;
}

function analysisKey(sketch: Pick<Sketch, "id" | "version">) { return `${sketch.id}:${sketch.version}`; }

function savedAgeAnalysis(saved: NonNullable<SavedSketch["versions"][number]["age_analysis"]>, sketchId: string, version: number): AgeAnalysis {
  return {
    sketch_id: sketchId,
    version,
    status: "available",
    ml_estimated_age: saved.ml_estimated_age,
    out_of_training_range: saved.out_of_training_range,
    model: saved.model,
    training_domain: saved.training_domain,
    interpretation: saved.interpretation,
    artifact_sha256: { backbone: saved.backbone_hash, age_head: saved.age_head_hash },
    analyzed_at: saved.analyzed_at,
  };
}

async function imageUrlToDataUrl(url: string): Promise<string | null> {
  try {
    const response = await fetch(url);
    if (!response.ok) return null;
    const blob = await response.blob();
    return await new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = () => resolve(typeof reader.result === "string" ? reader.result : null);
      reader.onerror = () => resolve(null);
      reader.readAsDataURL(blob);
    });
  } catch {
    return null;
  }
}

export default function Home() {
  const [description, setDescription] = useState("");
  const [witnessAge, setWitnessAge] = useState("");
  const [instruction, setInstruction] = useState("");
  const [history, setHistory] = useState<Sketch[]>([]);
  const [localHistory, setLocalHistory] = useState<HistorySketch[]>([]);
  const [historyReady, setHistoryReady] = useState(false);
  const [historyStorageWarning, setHistoryStorageWarning] = useState("");
  const [saved, setSaved] = useState<SavedSketch[]>([]);
  const [showAllHistory, setShowAllHistory] = useState(false);
  const [current, setCurrent] = useState<Sketch | null>(null);
  const [analysis, setAnalysis] = useState<FaceAnalysis | null>(null);
  const [benchmark, setBenchmark] = useState<Benchmark | null>(null);
  const [comparisons, setComparisons] = useState<Record<string, SketchComparison>>({});
  const [ageAnalyses, setAgeAnalyses] = useState<Record<string, AgeAnalysis>>({});
  const [ageError, setAgeError] = useState<{ key: string; message: string } | null>(null);
  const [ageLoadingKey, setAgeLoadingKey] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<"2d" | "3d">("2d");
  const [busy, setBusy] = useState<"generate" | "refine" | "benchmark" | "compare" | null>(null);
  const [error, setError] = useState("");
  const currentAgeKey = current ? analysisKey(current) : null;
  const currentAgeAnalysis = currentAgeKey ? ageAnalyses[currentAgeKey] : undefined;
  const currentAgeError = ageError?.key === currentAgeKey ? ageError.message : null;
  const canAnalyzeAge = Boolean(current?.id && current.version >= 1 && current.image_url);
  const apiBaseUrl = (process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000").replace(/\/$/, "");
  const historyEntriesByKey = new Map<string, { sketch: Sketch; age: AgeAnalysis | null }>();
  saved.forEach((savedSketch) => savedSketch.versions.forEach((version) => {
    const sketch: Sketch = {
      id: savedSketch.sketch_id,
      version: version.version,
      description: "Saved sketch",
      refinement_instruction: version.refinement_instruction,
      image_url: `${apiBaseUrl}${version.image_url}`,
      status: "saved",
      witness_age: savedSketch.witness_age,
    };
    const age = version.age_analysis ? savedAgeAnalysis(version.age_analysis, savedSketch.sketch_id, version.version) : null;
    historyEntriesByKey.set(analysisKey(sketch), { sketch, age });
  }));
  localHistory.forEach((sketch) => {
    const key = analysisKey(sketch);
    const existing = historyEntriesByKey.get(key);
    const age = ageAnalyses[key] ?? existing?.age ?? null;
    historyEntriesByKey.set(key, {
      sketch: { ...existing?.sketch, ...sketch },
      age,
    });
  });
  history.forEach((sketch) => {
    const key = analysisKey(sketch);
    const existing = historyEntriesByKey.get(key);
    const currentAge = ageAnalyses[key];
    historyEntriesByKey.set(key, {
      sketch: { ...existing?.sketch, ...sketch },
      age: currentAge ?? existing?.age ?? null,
    });
  });
  const historyEntries = Array.from(historyEntriesByKey.values()).reverse();
  const visibleHistoryEntries = showAllHistory ? historyEntries : historyEntries.slice(0, 10);

  useEffect(() => {
    let snapshot = deserializeHistory(null);
    try {
      snapshot = deserializeHistory(window.localStorage.getItem(HISTORY_STORAGE_KEY));
    } catch {
      setHistoryStorageWarning("Browser storage is unavailable; history will last only for this page session.");
    }
    setLocalHistory(snapshot.sketches);
    setAgeAnalyses(snapshot.ageAnalyses);
    setHistoryReady(true);

    api.saved().then(async (remoteSaved) => {
      setSaved(remoteSaved);
      const remoteSketches: HistorySketch[] = [];
      const remoteAnalyses: Record<string, AgeAnalysis> = {};
      for (const savedSketch of remoteSaved) {
        for (const version of savedSketch.versions) {
          const key = `${savedSketch.sketch_id}:${version.version}`;
          const localVersion = snapshot.sketches.find((item) => analysisKey(item) === key);
          const remoteUrl = `${apiBaseUrl}${version.image_url}`;
          const dataImageUrl = localVersion?.image_url?.startsWith("data:image/")
            ? localVersion.image_url
            : await imageUrlToDataUrl(remoteUrl);
          if (dataImageUrl) remoteSketches.push({
            id: savedSketch.sketch_id,
            version: version.version,
            description: localVersion?.description || "Saved sketch",
            refinement_instruction: version.refinement_instruction || localVersion?.refinement_instruction,
            image_url: dataImageUrl,
            status: "saved",
            witness_age: savedSketch.witness_age ?? localVersion?.witness_age ?? null,
            created_at: localVersion?.created_at || savedSketch.created_at,
          });
          if (version.age_analysis) remoteAnalyses[key] = savedAgeAnalysis(version.age_analysis, savedSketch.sketch_id, version.version);
        }
      }
      setLocalHistory((items) => mergeHistory(items, remoteSketches));
      setAgeAnalyses((items) => ({ ...remoteAnalyses, ...snapshot.ageAnalyses, ...items }));
    }).catch(() => setSaved([]));
  }, [apiBaseUrl]);

  useEffect(() => {
    if (!historyReady) return;
    try {
      window.localStorage.setItem(HISTORY_STORAGE_KEY, serializeHistory({ sketches: localHistory, ageAnalyses }));
      setHistoryStorageWarning("");
    } catch {
      setHistoryStorageWarning("Browser storage is full or unavailable; new history may not persist on this device.");
    }
  }, [historyReady, localHistory, ageAnalyses]);
  async function analyze(id: string) { try { setAnalysis(await api.analyze(id)); } catch { setAnalysis(null); } }
  function selectSketch(sketch: Sketch) { setCurrent(sketch); setAgeError(null); }
  function selectHistoryEntry(sketch: Sketch, age: AgeAnalysis | null) {
    selectSketch(sketch);
    setHistory(historyForSketch(historyEntries.map((entry) => entry.sketch), sketch.id));
    setAnalysis(null);
    setBenchmark(null);
    if (age) setAgeAnalyses((items) => ({ ...items, [analysisKey(sketch)]: age }));
  }
  async function generate(event: FormEvent) { event.preventDefault(); if (!description.trim()) return setError("Please enter an eyewitness description first."); const parsedAge = witnessAge.trim() === "" ? undefined : Number(witnessAge); if (parsedAge !== undefined && (!Number.isFinite(parsedAge) || parsedAge < 1 || parsedAge > 120)) return setError("Witness Age must be a number from 1 to 120."); setBusy("generate"); setError(""); try { const sketch = await api.generate(description, parsedAge); const savedSketch: HistorySketch = { ...sketch, created_at: new Date().toISOString() }; setCurrent(savedSketch); setHistory([savedSketch]); setLocalHistory((items) => mergeHistory(items, [savedSketch])); setAnalysis(null); setBenchmark(null); setAgeError(null); await analyze(sketch.id); } catch (err) { setError(err instanceof Error ? err.message : "Unable to generate sketch."); } finally { setBusy(null); } }
  async function refine(event: FormEvent) { event.preventDefault(); if (!current) return; if (!instruction.trim()) return setError("Please enter a refinement instruction."); setBusy("refine"); setError(""); try { const sketch = await api.refine(current.id, instruction); const savedSketch: HistorySketch = { ...sketch, created_at: new Date().toISOString() }; setCurrent(savedSketch); setHistory((items) => [...items, savedSketch]); setLocalHistory((items) => mergeHistory(items, [savedSketch])); setInstruction(""); setAnalysis(null); setBenchmark(null); setAgeError(null); await analyze(sketch.id); } catch (err) { setError(err instanceof Error ? err.message : "Unable to refine sketch."); } finally { setBusy(null); } }
  async function runBenchmark() { if (!current) return; setBusy("benchmark"); setError(""); try { setBenchmark(await api.benchmark(current.id)); } catch (err) { setError(err instanceof Error ? err.message : "Unable to run benchmark."); } finally { setBusy(null); } }
  async function analyzeAge() { if (!current || !canAnalyzeAge) return; const selected = current; const key = analysisKey(selected); setAgeLoadingKey(key); setAgeError(null); try { const result = await api.analyzeAge(selected.id, selected.version, selected.image_url); if (result.sketch_id !== selected.id || result.version !== selected.version) throw new Error("Age analysis returned a different sketch version."); setAgeAnalyses((items) => ({ ...items, [key]: { ...result, analyzed_at: new Date().toISOString() } })); } catch { setAgeError({ key, message: "Age analysis is unavailable for this sketch." }); } finally { setAgeLoadingKey(null); } }
  async function compareVersions(id: string, versionA: number, versionB: number) { const key = `${id}:${versionA}-${versionB}`; setBusy("compare"); setError(""); try { const result = await api.compare(id, versionA, versionB); setComparisons((items) => ({ ...items, [key]: result })); } catch (err) { setError(err instanceof Error ? err.message : "Unable to compare sketch versions."); } finally { setBusy(null); } }

  return <main className="mx-auto min-h-screen max-w-[1440px] px-6 py-8 md:px-12">
    <header className="flex items-start justify-between border-b border-[var(--line)] pb-7"><div><p className="mb-3 text-[11px] font-bold uppercase tracking-[0.24em] text-[var(--accent)]">Research Prototype · ML for the Arts</p><h1 className="font-serif text-4xl tracking-tight md:text-5xl">AI Police Sketch</h1><p className="mt-3 text-[15px] text-[var(--muted)]">Transform eyewitness descriptions into iterative facial sketches.</p></div><div className="hidden border-l border-[var(--line)] pl-5 text-right text-xs leading-5 text-[var(--muted)] md:block">PHASE 02A<br />LOCAL ML ANALYSIS<br />LOCAL SESSION</div></header>

    <div className="grid gap-8 py-9 lg:grid-cols-[minmax(300px,0.8fr)_minmax(500px,1.2fr)]">
      <section><div className="mb-4"><p className="label">01 / Input method</p><h2 className="section-title">Create a Starting Sketch</h2></div><form onSubmit={generate} className="border-t border-[var(--ink)] pt-4"><div className="mb-2 flex justify-end"><button type="button" className="text-xs text-[var(--accent)] underline underline-offset-4" onClick={() => setDescription(example)}>Load example</button></div><textarea value={description} onChange={(event) => setDescription(event.target.value)} maxLength={2000} className="h-52 w-full resize-none border border-[var(--line)] bg-white p-4 text-[15px] leading-6 outline-none transition focus:border-[var(--accent)]" placeholder="Describe the person's approximate age, face shape, hair, eyebrows, eyes, nose, lips, facial hair, distinguishing features, and other visible characteristics..." /><label className="mt-4 block text-sm font-bold text-[var(--ink)]">Witness Age <span className="font-normal text-[var(--muted)]">(optional)</span><input type="number" min="1" max="120" step="1" inputMode="numeric" value={witnessAge} onChange={(event) => setWitnessAge(event.target.value)} placeholder="35" className="mt-2 block w-full border border-[var(--line)] bg-white p-3 text-sm font-normal outline-none focus:border-[var(--accent)]" /></label><div className="mt-2 flex justify-between text-xs text-[var(--muted)]"><span>Natural language input</span><span>{description.length} / 2000</span></div><button disabled={Boolean(busy)} className="mt-7 w-full bg-[var(--ink)] px-5 py-4 text-sm font-bold uppercase tracking-[0.14em] text-white transition hover:bg-[#35424b] disabled:cursor-wait disabled:opacity-60">{busy === "generate" ? "Generating..." : "Generate Sketch"}</button></form>{current && <form onSubmit={refine} className="mt-12 border-t border-[var(--line)] pt-5"><p className="label">02 / Iterate</p><h2 className="section-title">Refine Sketch</h2><textarea value={instruction} onChange={(event) => setInstruction(event.target.value)} className="mt-4 h-28 w-full resize-none border border-[var(--line)] bg-white p-4 text-sm leading-6 outline-none focus:border-[var(--accent)]" placeholder="Example: Make the eyebrows thicker, make the jaw slightly wider, and add a small scar above the left eyebrow." /><button disabled={Boolean(busy)} className="mt-3 border border-[var(--ink)] px-5 py-3 text-xs font-bold uppercase tracking-[0.14em] hover:bg-white disabled:opacity-50">{busy === "refine" ? "Updating..." : "Update Sketch"}</button></form>}{error && <p className="mt-5 border-l-2 border-[var(--accent)] bg-[#fff6f1] p-3 text-sm text-[#8d4327]">{error}</p>}</section>
      <section><div className="mb-4"><p className="label">03 / Output</p><h2 className="section-title">Generated Sketch</h2><div className="mt-3 inline-flex border border-[var(--line)] bg-white p-1 text-xs font-bold uppercase tracking-[0.12em]"><button type="button" onClick={() => setViewMode("2d")} className={`px-3 py-2 ${viewMode === "2d" ? "bg-[var(--ink)] text-white" : "text-[var(--muted)]"}`}>2D Sketch</button><button type="button" onClick={() => setViewMode("3d")} className={`px-3 py-2 ${viewMode === "3d" ? "bg-[var(--ink)] text-white" : "text-[var(--muted)]"}`}>3D View</button></div></div>{viewMode === "3d" ? <Face3DViewer /> : <MockPortrait sketch={current} />}{current && viewMode === "2d" && <p className="mt-3 text-xs text-[var(--muted)]">{current.image_url ? "Generated image output." : "Mock output only."}</p>}
        {current && <div className="mt-6 border-t border-[var(--line)] pt-4"><p className="label">ML Age Analysis</p><div className="mt-3 border border-[var(--line)] bg-white p-4 text-sm text-[var(--muted)]">{canAnalyzeAge ? <button type="button" onClick={analyzeAge} disabled={ageLoadingKey !== null} className="border border-[var(--ink)] px-4 py-2 text-xs font-bold uppercase tracking-[0.12em] hover:bg-[#f7f4ee] disabled:cursor-wait disabled:opacity-50">{ageLoadingKey === currentAgeKey ? "Analyzing age..." : "Analyze Age"}</button> : <p>Age analysis requires an image-backed sketch.</p>}{currentAgeError && <p className="mt-3 border-l-2 border-[var(--accent)] bg-[#fff6f1] p-3 text-sm text-[#8d4327]">{currentAgeError}</p>}{currentAgeAnalysis?.out_of_training_range ? <p className="mt-3 border-l-2 border-[var(--accent)] bg-[#fff6f1] p-3 text-sm text-[#8d4327]">Model prediction is outside its training age range (1–101). Result withheld.</p> : currentAgeAnalysis && <div className="mt-4"><p className="text-xs font-bold uppercase tracking-[0.12em] text-[var(--muted)]">ML Estimated Age</p><p className="mt-1 font-serif text-3xl text-[var(--ink)]">{currentAgeAnalysis.ml_estimated_age.toFixed(1)} years</p>{current.witness_age != null && <div className="mt-4 grid grid-cols-3 gap-2 border-t border-[var(--line)] pt-3 text-sm"><span>Witness Age: <b className="text-[var(--ink)]">{current.witness_age}</b></span><span>ML Estimated Age: <b className="text-[var(--ink)]">{currentAgeAnalysis.ml_estimated_age.toFixed(1)}</b></span><span>Visual Age Difference: <b className="text-[var(--ink)]">{`${currentAgeAnalysis.ml_estimated_age - current.witness_age >= 0 ? "+" : ""}${(currentAgeAnalysis.ml_estimated_age - current.witness_age).toFixed(1)}`}</b></span></div>}<div className="mt-4 grid gap-2 text-sm sm:grid-cols-2"><span>Model: <b className="text-[var(--ink)]">INT8 ResNet50 + FP32 Age Head</b></span><span>Training domain: <b className="text-[var(--ink)]">{currentAgeAnalysis.training_domain}</b></span></div></div>}{currentAgeAnalysis && <p className="mt-4 border-t border-[var(--line)] pt-3 text-xs leading-5">Exploratory analysis of this generated sketch. The model was evaluated on AgeDB photographs, so its photo-domain accuracy does not establish accuracy on police sketches.</p>}</div></div>}
        {analysis && <div className="mt-6 border-t border-[var(--line)] pt-4"><p className="label">ML Analysis</p><div className="mt-3 grid grid-cols-2 gap-3 text-sm text-[var(--muted)]"><span>Model: <b className="text-[var(--ink)]">ResNet-50</b></span><span>Device: <b className="text-[var(--ink)]">{analysis.device.toUpperCase()}</b></span><span>Embedding dimension: <b className="text-[var(--ink)]">{analysis.embedding_dimension}</b></span><span>Inference time: <b className="text-[var(--ink)]">{analysis.inference_time_ms} ms</b></span></div><button type="button" onClick={runBenchmark} disabled={Boolean(busy)} className="mt-4 border border-[var(--ink)] px-4 py-2 text-xs font-bold uppercase tracking-[0.12em] hover:bg-white disabled:opacity-50">{busy === "benchmark" ? "Running..." : "Run Benchmark"}</button>{benchmark && <div className="mt-4 border-t border-[var(--line)] pt-4 text-sm text-[var(--muted)]"><p className="font-bold text-[var(--ink)]">PyTorch Baseline</p><div className="mt-2 grid grid-cols-2 gap-2"><span>Mean: {benchmark.mean_latency_ms} ms</span><span>Median: {benchmark.median_latency_ms} ms</span><span>Min: {benchmark.min_latency_ms} ms</span><span>Max: {benchmark.max_latency_ms} ms</span><span>Std Dev: {benchmark.std_latency_ms} ms</span><span>Measured Runs: {benchmark.measured_runs}</span><span>Warm-up Runs: {benchmark.warmup_runs}</span><span>Device: {benchmark.device.toUpperCase()}</span></div></div>}</div>}
      </section>
    </div>
    {history.length > 1 && <section className="border-t border-[var(--line)] pt-7"><p className="label">Refinement ML Analysis</p><h2 className="section-title mb-5">Visual Feature Similarity</h2><div className="space-y-4">{history.slice(1).map((item, index) => { const previous = history[index]; const key = `${item.id}:${previous.version}-${item.version}`; const result = comparisons[key]; return <div key={key} className="border border-[var(--line)] p-4"><div className="flex items-center justify-between"><span className="text-sm font-bold">V{previous.version} → V{item.version}</span><button type="button" onClick={() => compareVersions(item.id, previous.version, item.version)} disabled={Boolean(busy)} className="border border-[var(--ink)] px-3 py-2 text-xs font-bold uppercase tracking-[0.1em] hover:bg-white disabled:opacity-50">{busy === "compare" ? "Comparing..." : "Compare"}</button></div>{result && <div className="mt-3 grid grid-cols-2 gap-2 text-sm text-[var(--muted)]"><span>Cosine Similarity: <b className="text-[var(--ink)]">{result.cosine_similarity.toFixed(4)}</b></span><span>Cosine Distance: <b className="text-[var(--ink)]">{result.cosine_distance.toFixed(4)}</b></span><span>Model: {result.model}</span><span>Embedding: {result.embedding_dimension}-D</span><span className="col-span-2">Instruction: {result.refinement_instruction || "Not available in saved metadata"}</span></div>}</div>; })}</div></section>}
    <section className="border-t border-[var(--line)] pt-7">
      <div className="mb-4 flex items-end justify-between">
        <div><p className="label">04 / Archive</p><h2 className="section-title">Sketch History</h2></div>
        <span className="text-xs text-[var(--muted)]">{historyEntries.length} version{historyEntries.length === 1 ? "" : "s"}</span>
      </div>
      {historyStorageWarning && <p role="status" className="mb-4 border-l-2 border-[var(--accent)] bg-[#fff6f1] p-3 text-xs text-[#8d4327]">{historyStorageWarning}</p>}
      {visibleHistoryEntries.length ? <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
        {visibleHistoryEntries.map(({ sketch, age }) => {
          const selected = current?.id === sketch.id && current?.version === sketch.version;
          return <button key={analysisKey(sketch)} type="button" onClick={() => selectHistoryEntry(sketch, age)} aria-pressed={selected}
            className={`overflow-hidden border text-left transition ${selected ? "border-[var(--accent)] bg-white ring-1 ring-[var(--accent)]" : "border-[var(--line)] bg-white/60 hover:border-[var(--ink)]"}`}>
            <div className="relative aspect-[4/3] overflow-hidden bg-[#e8e3db]">
              {sketch.image_url ? <img src={sketch.image_url} alt={`Sketch version ${sketch.version}`} className="h-full w-full object-cover grayscale" /> : <div className="flex h-full items-center justify-center text-xs text-[var(--muted)]">No image</div>}
              <span className="absolute left-2 top-2 bg-[var(--ink)]/90 px-2 py-1 text-[10px] font-bold uppercase tracking-wider text-white">V{sketch.version} · {sketch.version === 1 ? "Initial" : "Refined"}</span>
            </div>
            <div className="space-y-1 p-3">
              <div className="flex items-center justify-between gap-2 text-xs">
                <span className="truncate font-semibold text-[var(--ink)]">{sketch.witness_age != null ? `Witness ${sketch.witness_age}` : "Witness age —"}</span>
                {sketch.version > 1 && <span className="shrink-0 text-[10px] uppercase tracking-wide text-[var(--accent)]">Edited</span>}
              </div>
              <p className="text-xs text-[var(--muted)]">{age ? age.out_of_training_range ? "ML age withheld" : `ML age ${age.ml_estimated_age.toFixed(1)}` : "ML age not analyzed"}</p>
              {sketch.refinement_instruction && <p className="text-[10px] text-[var(--muted)]">Refinement version</p>}
            </div>
          </button>;
        })}
      </div> : <p className="text-sm text-[var(--muted)]">Generated versions will appear here.</p>}
      {historyEntries.length > 10 && <button type="button" onClick={() => setShowAllHistory((value) => !value)} className="mt-4 text-xs font-semibold uppercase tracking-wider text-[var(--accent)] underline underline-offset-4">{showAllHistory ? "Show less" : `Show all ${historyEntries.length} versions`}</button>}
    </section>
    <footer className="mt-16 flex gap-3 border-t border-[var(--line)] pt-5 text-xs leading-5 text-[var(--muted)]"><span className="text-[var(--accent)]">NOTE</span><p>Research prototype. Generated sketches are interpretive visualizations and should not be treated as identification evidence.</p></footer>
  </main>;
}
