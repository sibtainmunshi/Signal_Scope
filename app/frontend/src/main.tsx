import React, { useEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { Activity, ArrowDownToLine, ArrowRight, Check, ChevronRight, CircleHelp,
  FileImage, Fingerprint, FlaskConical, ImagePlus, Layers3, LoaderCircle,
  ScanLine, ShieldCheck, SlidersHorizontal, Sparkles, Info, LockKeyhole, ArrowUpRight, CircleDot } from 'lucide-react'
import { HeatmapViewer } from './HeatmapViewer'
import { ReleaseEvidence, hasReleaseEvidence } from './ReleaseEvidence'
import './style.css'
import './refinement.css'

type Health = { ready: boolean; device?: string; model_version?: string; message?: string }
type Prediction = {
  label: 'real' | 'ai_generated'; ai_score: number; threshold: number; confidence: number;
  calibrated: boolean; review_recommended: boolean; inference_ms: number; model_version: string;
  checkpoint_sha256: string; image_width: number; image_height: number; limitations: string[]
}
type StabilityRow = { transformation: string; ai_score: number; score_change: number; label_changed: boolean; label: string }
type Explanation = {
  overlay_data_url: string; statements: string[]; method: string; semantic_artifact_verified: boolean; elapsed_ms: number;
  target_class?: 'real' | 'ai_generated';
  localisation?: { supported: boolean; returned_class_drop: number; comparison_drop: number };
  masking_diagnostic?: { limitation: string }
}
type Analysis = {
  prediction: Prediction;
  explanation: Explanation | null;
  robustness: null | { results: StabilityRow[]; max_absolute_score_change: number; label_flip_count: number; transformations_tested: number; note: string };
  metadata: { format: string; exif_present: boolean; fields: Record<string, string>; c2pa_status: string; note: string; fusion_policy: string }
}
type Metrics = { roc_auc: number | null; macro_f1: number; accuracy: number; false_positive_rate: number | null; count: number; split: string; dataset?: string; confusion_matrix: number[][]; unseen_generator_auc: number | null }
type External = { macro_generator_roc_auc: number; per_generator: { generator: string; roc_auc: number; accuracy: number; false_positive_rate: number; true_positive_rate: number }[]; unique_images_evaluated: number }
type ModelReport = { ready: boolean; architecture?: string; model_version?: string; image_size?: number; source_resolution?: string; training_source?: string; transparency_note?: string; threshold?: number; calibrated?: boolean; metrics: Metrics | null; checkpoint_sha256?: string; unseen_generator_status?: string; external_development?: External | null; external_development_matched?: External | null; external_reserved?: External | null; external_reserved_matched?: External | null; cifake_test?: Metrics | null }

type TabId = 'evidence' | 'stability' | 'metadata'
const TABS: [TabId, string][] = [['evidence', 'Evidence'], ['stability', 'Stability'], ['metadata', 'Metadata']]
const PANEL_ID = 'evidence-panel'
const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp']

const percentage = (value: number) => `${(value * 100).toFixed(1)}%`
const scorePercentage = (value: number) => value >= .999 ? '>99.9%' : value <= .001 ? '<0.1%' : percentage(value)
const pretty = (name: string) => ({ original: 'Original image', jpeg_q90: 'JPEG · quality 90', jpeg_q70: 'JPEG · quality 70', jpeg_q50: 'JPEG · quality 50', half_resolution: '50% resolution', mild_blur: 'Mild blur', simulated_screenshot: 'Screenshot (sim.)' }[name] || name)
const verdictTitle = (prediction: Prediction) => prediction.review_recommended ? 'Review recommended'
  : prediction.label === 'ai_generated' ? 'Likely AI-generated' : 'Likely real'
const points = (value: number) => `${value > 0 ? '+' : ''}${(value * 100).toFixed(1)} pp`
const c2paStatus = (value: string) => ({ no_marker_found: 'No marker found', marker_found_unverified: 'Marker found · unverified', not_checked: 'Not checked' }[value] || 'Status unavailable')

function EvidenceDetails({ explanation }: { explanation: Explanation }) {
  const localisation = explanation.localisation
  return <>
    <div className="evidence-heading"><span className="evidence-label"><Layers3 size={15} /> MODEL INFLUENCE</span><span className="timing">{(explanation.elapsed_ms / 1000).toFixed(1)} s</span></div>
    <div className={`localisation-card ${localisation?.supported ? 'supported' : ''}`}>
      <CircleDot size={18} />
      <div><strong>{localisation ? localisation.supported ? 'Local influence supported' : 'Verdict is not localised' : 'Influence map available'}</strong>
        <p>{localisation ? localisation.supported
          ? 'Masking the highlighted patch reduced the returned-class score more than equally sized corner patches.'
          : 'The masking check does not support treating one highlighted region as the reason for this verdict.'
          : 'Localisation measurements were not supplied for this result.'}</p></div>
    </div>
    {localisation && <div className="diagnostic-grid">
      <div><span>Highlighted patch</span><strong>{points(localisation.returned_class_drop)}</strong></div>
      <div><span>Corner comparison</span><strong>{points(localisation.comparison_drop)}</strong></div>
      <p>Change in the returned-class score after masking. Positive = score decreased. pp = percentage points.</p>
    </div>}
    <div className="statement-list">{explanation.statements.map((statement, index) => <p className="evidence-statement" key={index}><span>{String(index + 1).padStart(2, '0')}</span>{statement}</p>)}</div>
    <details className="technical-details"><summary>Explanation method & limits <ChevronRight size={14} /></summary><p>{explanation.method}</p><p>{explanation.masking_diagnostic?.limitation || 'Model attribution is not a map of verified image defects.'}</p><p>Verified visual artifact: {explanation.semantic_artifact_verified ? 'Reported by the model service' : 'Not established'}</p></details>
  </>
}

function ExternalResults({ title, report, matched, final = false }: { title: string; report: External; matched?: External | null; final?: boolean }) {
  return <section className="panel report-panel" style={{ marginBottom: 24 }}>
    <h2>{title}</h2>
    <p className="muted">{final
      ? `${report.unique_images_evaluated.toLocaleString()} unique images, scored after the model and threshold were fixed. Three GLIDE configurations and DALLE share 500 real photos; they represent two generator families. See each generator's detection and false-positive rates below.`
      : 'Guided-diffusion and LDM images were excluded from classifier training; these development checks guided model selection. LDM and the training Stable Diffusion models belong to related families.'}</p>
    {[['As distributed', report], ['Format-matched', matched]].map(([label, value]) => {
      const result = value as External | null | undefined
      return result && <div key={String(label)}>
        <div className="metadata-row"><span>{String(label)} mean ROC-AUC</span><strong>{result.macro_generator_roc_auc.toFixed(4)}</strong></div>
        {result.per_generator.map(row => <div className="metadata-row" key={row.generator}><span>{row.generator}</span><strong>AUC {row.roc_auc.toFixed(3)}<br /><small>Real false positives {percentage(row.false_positive_rate)} / AI detected {percentage(row.true_positive_rate)}</small></strong></div>)}
      </div>
    })}
    <p className="subtle-note">Format-matched scoring applies the same centre crop, resize and JPEG encoding to both labels. It reduces encoding and geometry differences; earlier processing traces can remain. ROC-AUC measures score ranking and is not classification accuracy.</p>
  </section>
}

function App() {
  const [page, setPage] = useState<'analyze' | 'report' | 'about'>('analyze')
  const [health, setHealth] = useState<Health | null>(null)
  const [model, setModel] = useState<ModelReport | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState('')
  const [analysis, setAnalysis] = useState<Analysis | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [announcement, setAnnouncement] = useState('')
  const [dragging, setDragging] = useState(false)
  const [explain, setExplain] = useState(true)
  const [robustness, setRobustness] = useState(true)
  const [tab, setTab] = useState<TabId>('evidence')
  const [stage, setStage] = useState(0)
  const fileInput = useRef<HTMLInputElement>(null)
  const tabRefs = useRef<Partial<Record<TabId, HTMLButtonElement | null>>>({})
  const controller = useRef<AbortController | null>(null)

  useEffect(() => {
    fetch('/api/health').then(r => r.json()).then(setHealth).catch(() => setHealth({ ready: false, message: 'Cannot reach the local inference service.' }))
    fetch('/api/model').then(r => r.json()).then(setModel).catch(() => {})
    return () => controller.current?.abort()
  }, [])
  useEffect(() => {
    if (!file) { setPreview(''); return }
    const url = URL.createObjectURL(file)
    setPreview(url)
    return () => URL.revokeObjectURL(url)
  }, [file])
  useEffect(() => { window.scrollTo({ top: 0, behavior: 'instant' }) }, [page])
  // Paces the status line through the stages this request actually runs. The
  // server returns one response with no progress events, so this is even pacing,
  // not measured progress; it holds on the last stage rather than looping.
  useEffect(() => {
    if (!busy) { setStage(0); return }
    const timer = setInterval(() => setStage(current => current + 1), 2200)
    return () => clearInterval(timer)
  }, [busy])

  // Clearing the input lets the same path be chosen again; browsers fire no
  // change event when the selected file is identical to the previous one.
  function resetFileInput() {
    if (fileInput.current) fileInput.current.value = ''
  }

  function openFilePicker() {
    resetFileInput()
    fileInput.current?.click()
  }

  function selectFile(selected: File | undefined) {
    if (!selected || busy) return
    // Validate before discarding any existing result, so a rejected file never
    // destroys the analysis the user already has on screen.
    if (!ACCEPTED_TYPES.includes(selected.type)) { setError('Choose a JPEG, PNG or WebP image.'); resetFileInput(); return }
    if (selected.size > 25 * 1024 * 1024) { setError('Choose an image no larger than 25 MiB.'); resetFileInput(); return }
    setError(''); setAnnouncement(''); setAnalysis(null); setFile(selected)
  }

  function clearImage() {
    setFile(null); setAnalysis(null); setError(''); setAnnouncement(''); resetFileInput()
  }

  async function runAnalysis() {
    if (!file) return
    setBusy(true); setError(''); setAnalysis(null); setAnnouncement('Analyzing image…')
    const body = new FormData()
    body.append('image', file); body.append('explain', String(explain)); body.append('robustness', String(robustness))
    controller.current = new AbortController()
    try {
      const response = await fetch('/api/predict', { method: 'POST', body, signal: controller.current.signal })
      const data = await response.json()
      if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : Array.isArray(data.detail)
        ? data.detail.map((item: { msg?: string }) => item.msg || 'Invalid input').join(' ') : 'Analysis could not be completed.')
      setAnalysis(data); setTab('evidence')
      const result = data.prediction as Prediction
      setAnnouncement(`Analysis complete. ${verdictTitle(result)}. AI-generated score ${scorePercentage(result.ai_score)}, decision threshold ${percentage(result.threshold)}.`)
    } catch (e) {
      if (e instanceof Error && e.name !== 'AbortError') { setError(e.message); setAnnouncement('') }
    } finally { setBusy(false) }
  }

  function downloadAnalysis() {
    if (!analysis) return
    const blob = new Blob([JSON.stringify({ filename: file?.name, ...analysis }, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a'); anchor.href = url; anchor.download = 'signalscope-analysis.json'; anchor.click()
    URL.revokeObjectURL(url)
  }

  // Left/Right/Home/End move selection between tabs, as the ARIA tabs pattern expects.
  function onTabKeyDown(event: React.KeyboardEvent<HTMLDivElement>) {
    const order = TABS.map(([id]) => id)
    const current = order.indexOf(tab)
    const next = event.key === 'ArrowRight' ? (current + 1) % order.length
      : event.key === 'ArrowLeft' ? (current - 1 + order.length) % order.length
      : event.key === 'Home' ? 0
      : event.key === 'End' ? order.length - 1 : -1
    if (next < 0) return
    event.preventDefault()
    setTab(order[next])
    tabRefs.current[order[next]]?.focus()
  }

  const verdict = analysis?.prediction
  const title = verdict ? verdictTitle(verdict) : ''
  const releaseMatched = hasReleaseEvidence(model)
  const stages = ['Reading the image and applying EXIF orientation…',
    'Encoding with the frozen CLIP ViT-L/14 vision tower…',
    'Scoring the embedding against the trained MLP head…',
    ...(explain ? ['Mapping which regions influence this score…'] : []),
    ...(robustness ? ['Re-testing under compression, resize, blur and a simulated screenshot…'] : [])]
  // Measured CPU costs: ~0.4 s to score, ~5.5 s for the influence map, ~5 s for
  // the transformation set. Stated as a range because image size moves it.
  const estimate = explain && robustness ? 'usually 10-12 seconds'
    : explain ? 'usually 6-7 seconds' : robustness ? 'usually 5-6 seconds' : 'usually under a second'

  return <div className="shell">
    <a className="skip-link" href="#main-content">Skip to main content</a>
    <aside className="sidebar">
      <a className="brand" href="#" onClick={e => { e.preventDefault(); setPage('analyze') }} aria-label="SignalScope home">
        <span className="brand-icon"><ScanLine size={24} strokeWidth={1.6} /></span>
        <span>Signal<span className="brand-light">Scope</span><small>IMAGE INTELLIGENCE</small></span>
      </a>
      <div className="nav-label">WORKSPACE <span>{page === 'analyze' ? '01' : page === 'report' ? '02' : '03'} / 03</span></div>
      <nav aria-label="Main navigation">
        <button className={page === 'analyze' ? 'nav-item active' : 'nav-item'} aria-current={page === 'analyze' ? 'page' : undefined} onClick={() => setPage('analyze')}><ScanLine size={19} /><span className="nav-text">Analyze image</span><ChevronRight size={15} /></button>
        <button className={page === 'report' ? 'nav-item active' : 'nav-item'} aria-current={page === 'report' ? 'page' : undefined} onClick={() => setPage('report')}><FlaskConical size={19} /><span className="nav-text">Model report</span></button>
        <button className={page === 'about' ? 'nav-item active' : 'nav-item'} aria-current={page === 'about' ? 'page' : undefined} onClick={() => setPage('about')}><CircleHelp size={19} /><span className="nav-text">How it works</span></button>
      </nav>
      <div className="sidebar-note"><span className="eyebrow">A LITTLE MORE PERSPECTIVE</span><strong>Evidence before<br />certainty.</strong><p>Inspect the signal.<br />Understand what it can tell you.</p></div>
      <div className="sidebar-bottom"><span className={`status-dot ${health?.ready ? 'online' : ''}`} /><span>{health?.ready ? 'Local model connected' : health ? 'Model unavailable' : 'Connecting to model…'}</span><small>SIH 2026 <span>RESEARCH BUILD</span></small></div>
    </aside>
    <div className="workspace">
      <p className="visually-hidden" role="status">{announcement}</p>
      <main id="main-content" tabIndex={-1}>
        {page === 'analyze' && <>
          <div className="page-heading analyze-heading"><div><div className="eyebrow"><span className="eyebrow-line" /> LOOK A LITTLE CLOSER</div><h1>Every image has <span>a signal.</span></h1><p>A closer look at what's real, what's synthetic, and the evidence in between.</p></div><div className="heading-aside"><ScanLine size={26} strokeWidth={1.2} /><span>From pixels<br />to perspective.</span></div></div>
          <div className="workflow-strip" aria-label="Analysis workflow"><span className={!file ? 'current' : 'done'}><i>{file ? <Check size={12} /> : '01'}</i>Add an image</span><ChevronRight size={13} /><span className={file && !analysis ? 'current' : analysis ? 'done' : ''}><i>{analysis ? <Check size={12} /> : '02'}</i>Read the signal</span><ChevronRight size={13} /><span className={analysis ? 'current' : ''}><i>03</i>Explore the evidence</span></div>
          {!health?.ready && health && <div role="status" className="notice"><Info size={17} /><div><strong>Model unavailable</strong><p>{health.message || 'The local detector is not ready.'} Start the local inference service to analyze an image.</p></div></div>}
          <div className="analysis-grid">
            <section className="panel input-panel" aria-labelledby="image-panel-title">
              <div className="panel-heading"><span className="number">01</span><h2 id="image-panel-title">Image workspace</h2><span className="panel-caption">{analysis?.explanation ? 'INTERACTIVE VIEW' : 'YOUR SOURCE IMAGE'}</span></div>
              <input ref={fileInput} type="file" accept="image/jpeg,image/png,image/webp" onChange={e => selectFile(e.target.files?.[0])} className="visually-hidden" aria-label="Choose image" disabled={busy} />
              <div className={`drop-zone ${preview ? 'has-image' : ''} ${dragging ? 'dragging' : ''}`} onDragOver={e => { e.preventDefault(); if (!busy) setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={e => { e.preventDefault(); setDragging(false); selectFile(e.dataTransfer.files[0]) }}>
                {preview ? <HeatmapViewer original={preview} overlay={analysis?.explanation?.overlay_data_url} filename={file?.name || 'Uploaded image'} busy={busy} onRemove={clearImage} />
                : <button className="upload-button" onClick={openFilePicker}><span className="upload-corner top-left" /><span className="upload-corner bottom-right" /><div className="upload-illustration" aria-hidden="true"><span className="image-card-back" /><span className="image-card-front"><ImagePlus size={35} strokeWidth={1.25} /></span><span className="upload-plus">+</span></div><strong>Bring an image.<br />Find some clarity.</strong><span>Drag & drop here, or <em>browse files <ArrowUpRight size={12} /></em></span><small>JPEG, PNG, WEBP <i /> UP TO 25 MiB</small></button>}
              </div>
              {file && <div className="file-details"><FileImage size={13} /><span>{(file.size / 1024).toFixed(0)} KB</span><span>{verdict ? `${verdict.image_width} × ${verdict.image_height} px` : 'Ready for analysis'}</span><button disabled={busy} onClick={openFilePicker}>Change image <ArrowUpRight size={12} /></button></div>}
              <div className="analysis-options"><div className="option-heading"><SlidersHorizontal size={13} /> GO A LITTLE DEEPER <span>OPTIONAL</span></div>
                <label><span className="option-icon"><Layers3 size={17} /></span><span>Model influence map<small>Inspect the regions that affect the score</small></span><input type="checkbox" checked={explain} onChange={e => setExplain(e.target.checked)} disabled={busy} /></label>
                <label><span className="option-icon"><Activity size={17} /></span><span>Robustness check<small>Test compression, resizing, blur & screenshot simulation</small></span><input type="checkbox" checked={robustness} onChange={e => setRobustness(e.target.checked)} disabled={busy} /></label>
                <p className="option-estimate"><Info size={13} /> Local CPU inference: <strong>{estimate}</strong> with the checks you have selected.</p>
              </div>
              {error && <div className="error" role="alert"><Info size={16} />{error}</div>}
              <button className="analyze-button" disabled={!file || busy || !health?.ready} onClick={runAnalysis}>{busy ? <><LoaderCircle className="spin" size={18} /> Analyzing image…</> : <><ScanLine size={18} /><span>Analyze image</span><ArrowRight size={18} /></>}</button>
              <p className="privacy-note"><LockKeyhole size={12} /> Processed on this machine. Uploads are not saved.</p>
            </section>
            <section className="panel results-panel" aria-labelledby="result-panel-title" aria-busy={busy}>
              <div className="panel-heading"><span className="number">02</span><h2 id="result-panel-title">The assessment</h2>{analysis ? <button className="export-button" onClick={downloadAnalysis} aria-label="Download analysis JSON"><ArrowDownToLine size={14} /><span>Export</span></button> : <span className="panel-caption">{busy ? 'PROCESSING' : 'AWAITING IMAGE'}</span>}</div>
              {!analysis ? <div className={`empty-state ${busy ? 'is-loading' : ''}`}>
                <div className={`radar ${busy ? 'scanning' : ''}`} aria-hidden="true"><span /><span /><span /><div className="radar-core"><ScanLine size={30} strokeWidth={1.2} /></div><i /><i /></div>
                <span className="eyebrow">{busy ? 'FOLLOWING THE SIGNAL' : 'EVIDENCE, IN FOCUS'}</span><h3>{busy ? 'Taking a closer look.' : 'More than a real-or-AI label.'}</h3><p className={busy ? 'stage-line' : undefined} role={busy ? 'status' : undefined}>{busy ? stages[Math.min(stage, stages.length - 1)] : 'Understand the assessment, see the model’s influence, and discover how the result holds up.'}</p>
                {busy ? <div className="processing-track" aria-label="Analysis in progress"><span /></div> : <div className="empty-evidence"><span><CircleDot size={15} />Visual score</span><span><Layers3 size={15} />Influence map</span><span><Activity size={15} />Stability</span></div>}
                <span className="empty-footnote">{busy ? `Results appear when all selected checks finish — ${estimate} on this machine.` : 'Your analysis will appear here.'}</span>
              </div> : <>
                <div className={`verdict-card ${verdict!.review_recommended ? 'uncertain' : verdict!.label === 'ai_generated' ? 'synthetic' : 'real'}`}>
                  <div className="verdict-top"><span className="eyebrow"><span className="verdict-dot" /> VISUAL ASSESSMENT</span><span className="timing">{verdict!.inference_ms.toFixed(0)} ms inference</span></div>
                  <h3>{title}</h3><p>{verdict!.review_recommended ? 'This image needs closer review. Read the model notes and inspect the supporting evidence.' : 'An assessment of visual patterns. The image’s origin has not been independently verified.'}</p>
                  <div className="score-row"><span>AI-generated score<small>Model signal · 0–100%</small></span><strong>{scorePercentage(verdict!.ai_score)}</strong></div>
                  <div className="score-track" role="img" aria-label={`AI score ${percentage(verdict!.ai_score)}; decision threshold ${percentage(verdict!.threshold)}`}><span style={{ width: percentage(verdict!.ai_score) }} /><i style={{ left: percentage(verdict!.threshold) }} /></div>
                  <div className="score-labels"><span>Lower AI signal</span><span>Higher AI signal</span></div>
                  <div className="threshold-note"><span className="threshold-key" /> Decision threshold <strong>{percentage(verdict!.threshold)}</strong></div>
                  <div className="verdict-facts"><div><span>Binary label</span><strong>{verdict!.label === 'ai_generated' ? 'AI-generated' : 'Real'}</strong></div><div><span>Returned-class confidence</span><strong>{scorePercentage(verdict!.confidence)}</strong></div></div>
                  <p className="calibration-note">{verdict!.calibrated ? 'Calibrated on held-out development data.' : 'Uncalibrated model score.'} Class confidence is a model score, not the probability that this verdict is correct.</p>
                </div>
                {verdict!.limitations.length > 0 && <details className="model-limitations" open={verdict!.review_recommended || undefined}><summary><Info size={14} /> Model notes & limitations <span>{verdict!.limitations.length}</span><ChevronRight size={14} /></summary><ul>{verdict!.limitations.map((note, index) => <li key={index}>{note}</li>)}</ul></details>}
                <div className="result-tabs" role="tablist" aria-label="Evidence types" onKeyDown={onTabKeyDown}>{TABS.map(([id, label]) => <button key={id} id={`tab-${id}`} ref={element => { tabRefs.current[id] = element }} role="tab" aria-selected={tab === id} aria-controls={PANEL_ID} tabIndex={tab === id ? 0 : -1} className={tab === id ? 'selected' : ''} onClick={() => setTab(id)}>{id === 'evidence' ? <Layers3 size={14} /> : id === 'stability' ? <Activity size={14} /> : <FileImage size={14} />}{label}</button>)}</div>
                <div className="tab-content" role="tabpanel" id={PANEL_ID} aria-labelledby={`tab-${tab}`} tabIndex={0}>
                  {tab === 'evidence' && (analysis.explanation ? <EvidenceDetails explanation={analysis.explanation} /> : <div className="optional-empty"><Layers3 size={24} /><h3>Take a closer look</h3><p>Enable the model influence map, then analyze again to explore the visual evidence.</p></div>)}
                  {tab === 'stability' && (analysis.robustness ? <>
                    <div className={`stability-summary ${analysis.robustness.label_flip_count ? 'has-flips' : ''}`}><Activity size={21} /><div><strong>{analysis.robustness.label_flip_count === 0 ? 'Verdict remained stable' : `${analysis.robustness.label_flip_count} verdict change${analysis.robustness.label_flip_count === 1 ? '' : 's'}`}</strong><span>across {analysis.robustness.transformations_tested} tested transformations</span></div></div>
                    <div className="stability-head"><span>Transformation</span><span>AI score / change</span></div>
                    {analysis.robustness.results.map(row => <div className={`stability-row ${row.label_changed ? 'flipped' : ''}`} key={row.transformation}><span>{pretty(row.transformation)}<small>{row.label_changed ? 'Verdict changed' : 'Same verdict'}</small></span><div className="stability-track"><i style={{ width: percentage(row.ai_score) }} /><b style={{ left: percentage(verdict!.threshold) }} title="Decision threshold" /></div><strong>{scorePercentage(row.ai_score)}<small>{points(row.score_change)}</small></strong></div>)}
                    <div className="stability-maximum"><span>Largest absolute score shift</span><strong>{percentage(analysis.robustness.max_absolute_score_change).replace('%', ' pp')}</strong></div><p className="subtle-note">Bars show the AI score; the marker is the fixed decision threshold. pp = percentage points. Stability does not establish correctness.</p><p className="subtle-note">{analysis.robustness.note}</p>
                  </> : <div className="optional-empty"><Activity size={24} /><h3>How well does the signal hold up?</h3><p>Enable the robustness check, then analyze again to compare image transformations.</p></div>)}
                  {tab === 'metadata' && <><div className="evidence-label"><FileImage size={15} /> FILE & PROVENANCE</div><div className="metadata-row"><span>File format</span><strong>{analysis.metadata.format}</strong></div><div className="metadata-row"><span>EXIF metadata</span><strong>{analysis.metadata.exif_present ? 'Present' : 'Not present'}</strong></div><div className="metadata-row"><span>Content Credentials</span><strong>{c2paStatus(analysis.metadata.c2pa_status)}</strong></div>{Object.entries(analysis.metadata.fields).map(([key, value]) => <div className="metadata-row" key={key}><span>{key}</span><strong>{value}</strong></div>)}<div className="provenance-note"><Info size={15} /><p>C2PA marker detection is a limited scan, not signature validation. Missing metadata does not establish that an image is synthetic.</p></div><p className="subtle-note">{analysis.metadata.note}</p><p className="subtle-note">{analysis.metadata.fusion_policy}</p></>}
                </div>
              </>}
            </section>
          </div>
          <div className="transparency"><ShieldCheck size={21} /><div><strong>Useful signals. Honest limits.</strong><p>{model?.transparency_note || 'New generators and unfamiliar image sources can cause errors. Use this assessment to support a closer review.'}</p></div><button onClick={() => setPage('report')}>Explore model report <ArrowUpRight size={16} /></button></div>
        </>}
        {page === 'report' && <>
          <div className="page-heading"><div className="eyebrow"><span className="eyebrow-line" /> MEASURED, NOT ASSUMED</div><h1>Model report.</h1><p>A clear record of the model, its measurements, and its limitations.</p></div>
          <div className="model-identity"><div className="model-identity-icon"><Fingerprint size={35} strokeWidth={1.2} /></div><div><span className="eyebrow">{model?.ready ? 'LOADED CHECKPOINT' : 'MODEL INFORMATION'}</span><h2>{model?.architecture || 'Awaiting model connection'}</h2><p>{model?.model_version || 'No loaded model reported'}</p></div><span className="identity-badge"><span className={`status-dot ${model?.ready ? 'online' : ''}`} />{model?.ready ? 'Model loaded' : 'Unavailable'}</span></div>
          <div className="notice"><Info size={18} /><div><strong>Read the numbers in context.</strong><p>Self-evaluated public benchmarks. Organizer baseline and hidden-test scores were not supplied. Published release measurements are shown only when the loaded checkpoint and threshold match.</p></div></div>
          {model?.transparency_note && <section className="panel report-context"><h2>Deployment context</h2><p>{model.transparency_note}</p></section>}
          {releaseMatched && <ReleaseEvidence />}
          {model?.external_reserved && <ExternalResults title="Reserved generators: final results" report={model.external_reserved} matched={model.external_reserved_matched} final />}
          {(model?.metrics || !releaseMatched) && <><div className="section-title"><h2>In-domain validation</h2><span>LIVE SERVICE METRICS</span></div>
          <div className="metric-grid">{[['ROC-AUC', model?.metrics?.roc_auc?.toFixed(4)], ['Macro-F1', model?.metrics?.macro_f1?.toFixed(4)], ['Accuracy', model?.metrics ? percentage(model.metrics.accuracy) : undefined], ['False-positive rate', model?.metrics?.false_positive_rate != null ? percentage(model.metrics.false_positive_rate) : undefined]].map(([label, value], index) => <div className="metric-card" key={label}><span>{label}<small>0{index + 1}</small></span><strong>{value || '—'}</strong><small>{model?.metrics ? `${model.metrics.dataset ?? ''} ${model.metrics.split}`.trim() : 'Not supplied for this checkpoint'}</small></div>)}</div></>}
          {model?.external_development && <ExternalResults title="External development" report={model.external_development} matched={model.external_development_matched} />}
          {model?.cifake_test && <section className="panel report-panel spaced-panel"><h2>CIFAKE author test</h2><p className="muted">{model.cifake_test.count.toLocaleString()} images, evaluated after freeze. This is an in-domain test of small CIFAKE images.</p><div className="metadata-row"><span>ROC-AUC / accuracy</span><strong>{model.cifake_test.roc_auc?.toFixed(4)} / {percentage(model.cifake_test.accuracy)}</strong></div></section>}
          <div className="report-grid"><section className="panel report-panel"><div className="report-title"><Fingerprint size={18} /><h2>Reproducibility record</h2></div>{[['Model', model?.model_version], ['Architecture', model?.architecture], ['Training source', model?.training_source], ['Source resolution', model?.source_resolution], ['Crop input', model?.image_size ? `${model.image_size} × ${model.image_size} px` : undefined], ['Decision threshold', model?.threshold != null ? percentage(model.threshold) : undefined], ['Calibration', model?.calibrated != null ? model.calibrated ? 'Development-calibrated' : 'Uncalibrated' : undefined], ['Validation samples (live service)', model?.metrics?.count?.toLocaleString()], ['Unseen-generator result', model?.unseen_generator_status]].map(([key, value]) => <div className="metadata-row" key={key}><span>{key}</span><strong>{value || 'Not supplied'}</strong></div>)}<div className="checkpoint-hash"><span>CHECKPOINT SHA-256</span><code>{model?.checkpoint_sha256 || 'No checkpoint loaded'}</code></div></section>
          {(!releaseMatched || model?.metrics?.confusion_matrix) && <section className="panel report-panel"><div className="report-title"><Activity size={18} /><h2>Validation confusion matrix</h2></div><p className="muted">Rows are actual labels; columns are predicted labels.</p>{model?.metrics?.confusion_matrix ? <div className="matrix"><div /><div>Predicted real</div><div>Predicted AI</div><div>Actual real</div>{model.metrics.confusion_matrix[0].map((number, index) => <strong key={index}>{number.toLocaleString()}</strong>)}<div>Actual AI</div>{model.metrics.confusion_matrix[1].map((number, index) => <strong key={index}>{number.toLocaleString()}</strong>)}</div> : <div className="report-empty"><FlaskConical size={30} strokeWidth={1.3} /><strong>No validation matrix supplied</strong><p>The model service has not supplied a matrix for this checkpoint.</p></div>}<div className="report-note"><Info size={15} /><p>ROC-AUC measures score ranking. Accuracy and false-positive rate depend on the chosen threshold. These describe a benchmark, not certainty for an individual image.</p></div></section>}</div>
        </>}
        {page === 'about' && <>
          <div className="page-heading"><div className="eyebrow"><span className="eyebrow-line" /> UNDERSTAND THE EVIDENCE</div><h1>Look beyond <span>the verdict.</span></h1><p>A little context makes every signal more useful.</p></div>
          <div className="about-intro"><span className="eyebrow">FROM IMAGE TO INSIGHT</span><h2>A second look.<br />An informed decision.</h2><p>SignalScope examines visual patterns associated with synthetic imagery. Explore what the detector sees, how its assessment changes, and where its evidence stops.</p></div>
          <div className="about-grid">{[
            { Icon: ScanLine, title: 'Read the visual signal', text: 'A trained classifier scores patterns associated with synthetic images. The AI score and fixed threshold determine a binary label. Returned-class confidence is a model score, not a guarantee of correctness.' },
            { Icon: Layers3, title: 'Inspect the influence', text: 'The active CLIP model uses input-gradient attribution to visualise influence. Compare it with the original image, then read the masking diagnostic. A highlighted area does not establish a visible defect or prove where an image was generated.' },
            { Icon: Activity, title: 'Test the stability', text: 'Compression, resizing, blur and a simulated screenshot can change a score. Compare these measurements and any verdict changes. A stable result can still be wrong; a simulated screenshot is not a test of every device or capture method.' },
            { Icon: ShieldCheck, title: 'Understand the boundaries', text: 'EXIF and Content Credentials markers are separate context; they never override the visual score. Checkpoint identity and available evaluations are disclosed in the model report. New generators and unfamiliar image sources can cause errors.' },
            { Icon: FlaskConical, title: 'Explore the measured results', text: 'The model report brings together public benchmark scores, confusion matrices, degradation tests and active-defence failures. Published evidence appears only for its matching checkpoint. Organizer hidden-test results remain unavailable.' },
            { Icon: ArrowDownToLine, title: 'Keep the evidence', text: 'Download the complete analysis as JSON: prediction, influence map, masking checks, stability and metadata. Batch prediction is available through the documented CLI; this workspace analyzes one image at a time.' },
          ].map(({ Icon, title, text }, index) => <section className="panel about-card" key={title}><div className="about-card-top"><span><Icon size={23} strokeWidth={1.5} /></span><small>0{index + 1}</small></div><h2>{title}</h2><p>{text}</p></section>)}</div>
          <div className="scope-note"><Info size={18} /><div><strong>What this release covers</strong><p>Core classification · A: influence explanations · C: degradation measurements · D: EXIF / C2PA marker scan · F: local interface and batch CLI · G: active-defence analysis. Generator attribution (B) and image-caption consistency (E) are not implemented. C2PA signatures are not verified.</p></div></div>
          <div className="transparency"><Sparkles size={21} /><div><strong>Built for SIH 2026. Designed for responsible review.</strong><p>For general synthetic imagery: products, objects, art and scenes. Do not upload images of identifiable people. No face-swap identification, personal profiling or political-claim adjudication.</p></div></div>
        </>}
        <footer><span><ScanLine size={14} /> SignalScope <i>/</i> See the signal.</span><span>LOCAL INFERENCE <i>·</i> REPRODUCIBLE EVIDENCE</span></footer>
      </main>
    </div>
  </div>
}

createRoot(document.getElementById('root')!).render(<React.StrictMode><App /></React.StrictMode>)
