import React, { useEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { Activity, ArrowDownToLine, ArrowRight, Check, ChevronRight, CircleHelp,
  FileImage, Fingerprint, FlaskConical, ImagePlus, Layers3, LoaderCircle,
  ScanLine, ShieldCheck, SlidersHorizontal, Sparkles, X } from 'lucide-react'
import './style.css'

type Health = { ready: boolean; device?: string; model_version?: string; message?: string }
type Prediction = {
  label: 'real' | 'ai_generated'; ai_score: number; threshold: number; confidence: number;
  calibrated: boolean; review_recommended: boolean; inference_ms: number; model_version: string;
  checkpoint_sha256: string; image_width: number; image_height: number; limitations: string[]
}
type StabilityRow = { transformation: string; ai_score: number; score_change: number; label_changed: boolean; label: string }
type Analysis = {
  prediction: Prediction;
  explanation: null | { overlay_data_url: string; statements: string[]; method: string; semantic_artifact_verified: boolean; elapsed_ms: number };
  robustness: null | { results: StabilityRow[]; max_absolute_score_change: number; label_flip_count: number; transformations_tested: number; note: string };
  metadata: { format: string; exif_present: boolean; fields: Record<string, string>; c2pa_status: string; note: string; fusion_policy: string }
}
type Metrics = { roc_auc: number | null; macro_f1: number; accuracy: number; false_positive_rate: number | null; count: number; split: string; dataset?: string; confusion_matrix: number[][]; unseen_generator_auc: number | null }
type External = { macro_generator_roc_auc: number; per_generator: { generator: string; roc_auc: number; accuracy: number; false_positive_rate: number }[] }
type ModelReport = { ready: boolean; architecture?: string; model_version?: string; image_size?: number; source_resolution?: string; training_source?: string; transparency_note?: string; threshold?: number; calibrated?: boolean; metrics: Metrics | null; checkpoint_sha256?: string; unseen_generator_status?: string; external_development?: External | null; external_development_matched?: External | null }

const percentage = (value: number) => `${(value * 100).toFixed(1)}%`
const scorePercentage = (value: number) => value >= .999 ? '>99.9%' : value <= .001 ? '<0.1%' : percentage(value)
const pretty = (name: string) => ({ original: 'Original image', jpeg_q90: 'JPEG · quality 90', jpeg_q70: 'JPEG · quality 70', jpeg_q50: 'JPEG · quality 50', half_resolution: '50% resolution', mild_blur: 'Mild blur' }[name] || name)

function App() {
  const [page, setPage] = useState<'analyze' | 'report' | 'about'>('analyze')
  const [health, setHealth] = useState<Health | null>(null)
  const [model, setModel] = useState<ModelReport | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState('')
  const [analysis, setAnalysis] = useState<Analysis | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [dragging, setDragging] = useState(false)
  const [explain, setExplain] = useState(true)
  const [robustness, setRobustness] = useState(true)
  const [showHeatmap, setShowHeatmap] = useState(true)
  const [tab, setTab] = useState<'evidence' | 'stability' | 'metadata'>('evidence')
  const fileInput = useRef<HTMLInputElement>(null)
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

  function selectFile(selected: File | undefined) {
    if (!selected || busy) return
    setError(''); setAnalysis(null)
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(selected.type)) { setError('Choose a JPEG, PNG or WebP image.'); return }
    if (selected.size > 10 * 1024 * 1024) { setError('Choose an image smaller than 10 MiB.'); return }
    setFile(selected)
  }

  async function runAnalysis() {
    if (!file) return
    setBusy(true); setError(''); setAnalysis(null)
    const body = new FormData()
    body.append('image', file); body.append('explain', String(explain)); body.append('robustness', String(robustness))
    controller.current = new AbortController()
    try {
      const response = await fetch('/api/predict', { method: 'POST', body, signal: controller.current.signal })
      const data = await response.json()
      if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Analysis could not be completed.')
      setAnalysis(data); setTab('evidence'); setShowHeatmap(true)
    } catch (e) {
      if (e instanceof Error && e.name !== 'AbortError') setError(e.message)
    } finally { setBusy(false) }
  }

  function downloadAnalysis() {
    if (!analysis) return
    const blob = new Blob([JSON.stringify({ filename: file?.name, ...analysis }, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const anchor = document.createElement('a'); anchor.href = url; anchor.download = 'signalscope-analysis.json'; anchor.click()
    URL.revokeObjectURL(url)
  }

  const verdict = analysis?.prediction
  const title = verdict?.review_recommended ? 'Review recommended' : verdict?.label === 'ai_generated' ? 'Likely AI-generated' : 'Likely real'

  return <div className="shell">
    <aside className="sidebar">
      <a className="brand" href="#" onClick={e => { e.preventDefault(); setPage('analyze') }} aria-label="SignalScope home"><span className="brand-icon"><ScanLine size={23} /></span><span>Signal<span className="brand-light">Scope</span><small>IMAGE INTELLIGENCE</small></span></a>
      <div className="nav-label">WORKSPACE</div>
      <nav aria-label="Main navigation">
        <button className={page === 'analyze' ? 'nav-item active' : 'nav-item'} onClick={() => setPage('analyze')}><ScanLine size={18} /> Analyze image <ChevronRight size={15} /></button>
        <button className={page === 'report' ? 'nav-item active' : 'nav-item'} onClick={() => setPage('report')}><FlaskConical size={18} /> Model report</button>
        <button className={page === 'about' ? 'nav-item active' : 'nav-item'} onClick={() => setPage('about')}><CircleHelp size={18} /> How it works</button>
      </nav>
      <div className="sidebar-note"><Fingerprint size={25} /><strong>Evidence before certainty.</strong><p>Inspect the signal. Understand its limits. Make an informed call.</p></div>
      <div className="sidebar-bottom"><span className={`status-dot ${health?.ready ? 'online' : ''}`} /><span>{health?.ready ? 'Local model connected' : health ? 'Model unavailable' : 'Connecting to model…'}</span><small>SIH 2026 · Development build</small></div>
    </aside>
    <div className="workspace">
      <header className="topbar"><div className="breadcrumb">Workspace <ChevronRight size={13} /><span>{page === 'analyze' ? 'Image analysis' : page === 'report' ? 'Model report' : 'How it works'}</span></div><span className="local-badge"><span className="status-dot online" /> LOCAL PROCESSING</span></header>
      <main>
        {page === 'analyze' && <>
          <div className="page-heading"><div><div className="eyebrow">LOOK A LITTLE CLOSER</div><h1>Every image has a signal.</h1><p>Explore whether an image is synthetic, with evidence you can inspect.</p></div><div className="heading-symbol"><ScanLine size={38} strokeWidth={1} /></div></div>
          {!health?.ready && health && <div role="status" className="notice">{health.message} Start the backend with a trained checkpoint to analyze images.</div>}
          <div className="analysis-grid">
            <section className="panel input-panel">
              <div className="panel-heading"><span className="number">01</span><h2>Your image</h2><span className="panel-caption">JPEG, PNG, WEBP</span></div>
              <input ref={fileInput} type="file" accept="image/jpeg,image/png,image/webp" onChange={e => selectFile(e.target.files?.[0])} className="visually-hidden" aria-label="Choose image" disabled={busy} />
              <div className={`drop-zone ${preview ? 'has-image' : ''} ${dragging ? 'dragging' : ''}`} onDragOver={e => { e.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={e => { e.preventDefault(); setDragging(false); selectFile(e.dataTransfer.files[0]) }}>
                {preview ? <><img className="image-preview" src={analysis?.explanation && showHeatmap ? analysis.explanation.overlay_data_url : preview} alt={analysis?.explanation && showHeatmap ? 'Model influence overlay; not verified artifact segmentation' : 'Uploaded image'} /><div className="image-toolbar"><span><FileImage size={13} /> {file?.name}</span><button aria-label="Remove image" disabled={busy} onClick={() => { setFile(null); setAnalysis(null); if (fileInput.current) fileInput.current.value = '' }}><X size={16} /></button></div>{analysis?.explanation && <button className="overlay-toggle" onClick={() => setShowHeatmap(!showHeatmap)}><Layers3 size={14} />{showHeatmap ? 'View original' : 'View model influence'}</button>}</> : <button className="upload-button" onClick={() => fileInput.current?.click()}><div className="upload-icon"><ImagePlus size={30} strokeWidth={1.4} /></div><strong>Drop an image here</strong><span>or <em>browse files</em> to get started</span><small>Up to 10 MiB · Images stay local</small></button>}
              </div>
              {file && <div className="file-details"><span>{(file.size / 1024).toFixed(0)} KB</span><span>{verdict ? `${verdict.image_width} × ${verdict.image_height} px` : 'Ready for analysis'}</span><button disabled={busy} onClick={() => fileInput.current?.click()}>Change image</button></div>}
              <div className="analysis-options"><div className="option-heading"><SlidersHorizontal size={14} /> ANALYSIS OPTIONS</div><label><input type="checkbox" checked={explain} onChange={e => setExplain(e.target.checked)} disabled={busy} /><span>Model influence map<small>See which regions influence the result</small></span></label><label><input type="checkbox" checked={robustness} onChange={e => setRobustness(e.target.checked)} disabled={busy} /><span>Robustness check<small>Compare compression, resizing and blur</small></span></label></div>
              {error && <div className="error" role="alert">{error}</div>}
              <button className="analyze-button" disabled={!file || busy || !health?.ready} onClick={runAnalysis}>{busy ? <><LoaderCircle className="spin" size={18} /> Analyzing image…</> : <><ScanLine size={18} /> Analyze image <ArrowRight size={18} /></>}</button>
              <p className="privacy-note"><ShieldCheck size={13} /> Processed on this machine. Uploads are not saved.</p>
            </section>
            <section className="panel results-panel" aria-live="polite">
              <div className="panel-heading"><span className="number">02</span><h2>Analysis</h2>{analysis && <button className="icon-button" onClick={downloadAnalysis} aria-label="Download analysis JSON"><ArrowDownToLine size={17} /></button>}</div>
              {!analysis ? <div className="empty-state"><div className={`radar ${busy ? 'scanning' : ''}`}><span /><span /><ScanLine size={35} strokeWidth={1.2} /></div><h3>{busy ? 'Reading the image’s signals' : 'A clearer picture starts here.'}</h3><p>{busy ? 'Running the detector and your selected evidence checks on the local model.' : 'Add an image to explore its visual score, model influence and stability.'}</p><div className="empty-tags"><span><Fingerprint size={12} /> Visual evidence</span><span><Activity size={12} /> Stability checks</span></div></div> : <>
                <div className={`verdict-card ${verdict?.review_recommended ? 'uncertain' : verdict?.label === 'ai_generated' ? 'synthetic' : 'real'}`}><div className="verdict-top"><span className="eyebrow">VISUAL ASSESSMENT</span><span className="timing">{verdict!.inference_ms.toFixed(0)} ms</span></div><h3>{title}</h3><p>{verdict!.review_recommended ? 'The score is near the operating threshold. Inspect the evidence before deciding.' : 'A model assessment of visual patterns, not a verified statement of origin.'}</p><div className="score-row"><span>AI-generated score</span><strong>{scorePercentage(verdict!.ai_score)}</strong></div><div className="score-track"><span style={{ width: percentage(verdict!.ai_score) }} /><i style={{ left: percentage(verdict!.threshold) }} title={`Decision threshold ${percentage(verdict!.threshold)}`} /></div><div className="score-labels"><span>Lower AI signal</span><span>Higher AI signal</span></div><div className="calibration-note">{verdict!.calibrated ? 'Calibrated on held-out development data' : 'Uncalibrated model score'} · threshold {percentage(verdict!.threshold)}</div></div>
                <div className="result-tabs" role="tablist" aria-label="Evidence types">{(['evidence','stability','metadata'] as const).map(t => <button key={t} role="tab" aria-selected={tab === t} className={tab === t ? 'selected' : ''} onClick={() => setTab(t)}>{t === 'evidence' ? 'Evidence' : t === 'stability' ? 'Stability' : 'Metadata'}</button>)}</div>
                <div className="tab-content" role="tabpanel">
                  {tab === 'evidence' && (analysis.explanation ? <><div className="evidence-label"><Layers3 size={15} /> MODEL INFLUENCE</div>{analysis.explanation.statements.map((s,i) => <p className="evidence-statement" key={i}>{s}</p>)}<div className="subtle-note">The overlay shows model attribution. It is not a map of verified image defects.</div></> : <p className="muted">Enable the model influence map and run again to inspect this evidence.</p>)}
                  {tab === 'stability' && (analysis.robustness ? <><div className="stability-summary"><strong>{analysis.robustness.label_flip_count === 0 ? 'Verdict remained stable' : `${analysis.robustness.label_flip_count} verdict change(s)`}</strong><span>across {analysis.robustness.transformations_tested} transformations</span></div>{analysis.robustness.results.map(r => <div className="stability-row" key={r.transformation}><span>{pretty(r.transformation)}</span><div><i style={{ width: percentage(r.ai_score) }} /></div><strong>{scorePercentage(r.ai_score)}</strong>{r.label_changed ? <span className="flip-indicator">Δ</span> : <Check size={12} />}</div>)}<p className="subtle-note">{analysis.robustness.note}</p></> : <p className="muted">Enable the robustness check and run again to compare transformations.</p>)}
                  {tab === 'metadata' && <><div className="metadata-row"><span>File format</span><strong>{analysis.metadata.format}</strong></div><div className="metadata-row"><span>EXIF metadata</span><strong>{analysis.metadata.exif_present ? 'Present' : 'Not present'}</strong></div><div className="metadata-row"><span>C2PA credentials</span><strong>Not checked</strong></div>{Object.entries(analysis.metadata.fields).map(([k,v]) => <div className="metadata-row" key={k}><span>{k}</span><strong>{v}</strong></div>)}<p className="subtle-note">{analysis.metadata.note}</p><p className="subtle-note">{analysis.metadata.fusion_policy}</p></>}
                </div>
              </>}
            </section>
          </div>
          <div className="transparency"><ShieldCheck size={19} /><div><strong>Useful signals. Honest limits.</strong><p>{model?.transparency_note || 'External development checks show substantial domain-shift errors. Results should support review, not replace it.'}</p></div><button onClick={() => setPage('report')}>View model report <ArrowRight size={15} /></button></div>
        </>}
        {page === 'report' && <><div className="page-heading"><div className="eyebrow">MEASURED, NOT ASSUMED</div><h1>Model report.</h1><p>Dataset, operating point and actual recorded development results.</p></div><div className="notice">These are self-evaluated public-benchmark results. Organizer baseline and hidden-test scores were not supplied.</div><div className="metric-grid">{[['ROC-AUC',model?.metrics?.roc_auc?.toFixed(4)],['Macro-F1',model?.metrics?.macro_f1?.toFixed(4)],['Accuracy',model?.metrics ? percentage(model.metrics.accuracy) : undefined],['False-positive rate',model?.metrics?.false_positive_rate != null ? percentage(model.metrics.false_positive_rate) : undefined]].map(([label,value]) => <div className="metric-card" key={label}><span>{label}</span><strong>{value || 'Pending'}</strong><small>{model?.metrics ? `${model.metrics.dataset ?? ''} ${model.metrics.split}`.trim() : 'No measured result yet'}</small></div>)}</div>{model?.external_development && <section className="panel report-panel" style={{ marginBottom: 24 }}><h2>External development: generalization gap</h2><p className="muted">Image sources and generators not used in training. These checks guide development; final reserved generators remain unevaluated. Format-matched scoring gives both labels the same crop, resize and JPEG, so file format cannot decide the result.</p><div className="metadata-row"><span>Mean generator ROC-AUC</span><strong>{model.external_development.macro_generator_roc_auc.toFixed(4)}</strong></div>{model.external_development.per_generator.map(row => <div className="metadata-row" key={row.generator}><span>{row.generator}</span><strong>AUC {row.roc_auc.toFixed(4)} · real-image false positives {percentage(row.false_positive_rate)}</strong></div>)}{model.external_development_matched && <><div className="metadata-row"><span>Format-matched mean ROC-AUC</span><strong>{model.external_development_matched.macro_generator_roc_auc.toFixed(4)}</strong></div>{model.external_development_matched.per_generator.map(row => <div className="metadata-row" key={'matched-' + row.generator}><span>{row.generator} (format-matched)</span><strong>AUC {row.roc_auc.toFixed(4)} · real-image false positives {percentage(row.false_positive_rate)}</strong></div>)}</>}</section>}<div className="report-grid"><section className="panel report-panel"><h2>Reproducibility record</h2>{[['Model', model?.model_version],['Architecture',model?.architecture],['Training source',model?.training_source],['Native data resolution',model?.source_resolution],['Model input',model?.image_size ? `${model.image_size} × ${model.image_size}` : undefined],['Evaluation samples',model?.metrics?.count?.toLocaleString()],['Unseen-generator result',model?.unseen_generator_status]].map(([k,v]) => <div className="metadata-row" key={k}><span>{k}</span><strong>{v || 'Pending'}</strong></div>)}<p className="subtle-note hash">Checkpoint SHA-256<br />{model?.checkpoint_sha256 || 'No checkpoint loaded'}</p></section><section className="panel report-panel"><h2>Confusion matrix</h2><p className="muted">Rows are actual labels; columns are predicted labels.</p>{model?.metrics?.confusion_matrix ? <div className="matrix"><div /><div>Predicted real</div><div>Predicted AI</div><div>Actual real</div>{model.metrics.confusion_matrix[0].map((n,i) => <strong key={i}>{n.toLocaleString()}</strong>)}<div>Actual AI</div>{model.metrics.confusion_matrix[1].map((n,i) => <strong key={i}>{n.toLocaleString()}</strong>)}</div> : <div className="report-empty">A measured evaluation will appear here after training.</div>}</section></div></>}
        {page === 'about' && <><div className="page-heading"><div className="eyebrow">UNDERSTAND THE EVIDENCE</div><h1>Look beyond the verdict.</h1><p>SignalScope makes its evidence and limitations inspectable.</p></div><div className="about-grid">{[{Icon:ScanLine,title:'A visual detector',text:'A locally trained classifier scores visual patterns associated with synthetic images. The score and fixed threshold determine the binary label.'},{Icon:Layers3,title:'Model-linked regions',text:'Grad-CAM highlights regions that influence the selected detector. A masking diagnostic measures how a patch affects its score. Neither establishes a specific visual defect.'},{Icon:Activity,title:'Measured stability',text:'The same image is compressed, resized and blurred. We show changes in score and verdict; stability alone does not mean correctness.'},{Icon:ShieldCheck,title:'Transparent boundaries',text:'Training data, splits, results and checkpoint identity are documented. External development failures are reported alongside the stronger in-domain results. Final reserved generators remain separate.'}].map(({Icon,title,text}) => <section className="panel about-card" key={title}><Icon size={25} /><h2>{title}</h2><p>{text}</p></section>)}</div><div className="transparency"><Sparkles size={20} /><div><strong>Built for SIH 2026</strong><p>For general synthetic imagery, products, objects and scenes. No face-swap identification or political-claim adjudication.</p></div></div></>}
        <footer><span>SignalScope <i>/</i> See the signal.</span><span>Local inference · Reproducible evidence</span></footer>
      </main>
    </div>
  </div>
}

createRoot(document.getElementById('root')!).render(<React.StrictMode><App /></React.StrictMode>)
