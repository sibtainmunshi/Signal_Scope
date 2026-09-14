import { useEffect, useState } from 'react'
import { ChevronsLeftRight, FileImage, Image, Layers3, ScanLine, X } from 'lucide-react'

type View = 'original' | 'influence' | 'compare'

/** Both layers use the same contained image geometry; the API overlay is never recoloured or cropped. */
export function HeatmapViewer({ original, overlay, filename, busy, onRemove }: {
  original: string; overlay?: string; filename: string; busy: boolean; onRemove: () => void
}) {
  const [view, setView] = useState<View>('original')
  const [split, setSplit] = useState(50)
  useEffect(() => { setView(overlay ? 'compare' : 'original'); setSplit(50) }, [original, overlay])
  const modes = [
    { id: 'original' as const, label: 'Original', Icon: Image },
    { id: 'influence' as const, label: 'Influence', Icon: Layers3 },
    { id: 'compare' as const, label: 'Compare', Icon: ChevronsLeftRight },
  ]

  return <div className={`image-inspector ${busy ? 'is-scanning' : ''}`}>
    <div className="inspector-toolbar">
      <span className="inspector-filename" title={filename}><FileImage size={14} /><span>{filename}</span></span>
      <button className="inspector-remove" aria-label="Remove image" disabled={busy} onClick={onRemove}><X size={16} /></button>
    </div>
    <div className="inspector-stage">
      <img className="inspector-image" src={view === 'influence' && overlay ? overlay : original}
        alt={view === 'influence' ? 'Model influence overlay. Highlighted regions are not verified image defects.' : 'Uploaded original image'} />
      {view === 'compare' && overlay && <>
        <img className="inspector-image comparison-layer" src={overlay} alt="Model influence overlay on the right side of the comparison"
          style={{ clipPath: `inset(0 0 0 ${split}%)` }} />
        <div className="comparison-divider" style={{ left: `${split}%` }} aria-hidden="true"><span><ChevronsLeftRight size={19} /></span></div>
        <input className="comparison-slider" type="range" min="0" max="100" value={split}
          aria-label="Original and influence comparison" aria-valuetext={`${split}% original, ${100 - split}% influence`}
          aria-describedby="comparison-help" onChange={event => setSplit(Number(event.target.value))} />
      </>}
      <div className="stage-labels" aria-hidden="true">
        <span>{view === 'influence' ? 'MODEL INFLUENCE' : 'ORIGINAL'}</span>
        {view === 'compare' && <span>MODEL INFLUENCE</span>}
      </div>
      {busy && <div className="scan-indicator"><ScanLine size={14} /> Analyzing visual patterns</div>}
    </div>
    {overlay ? <>
      <div className="inspector-controls" role="group" aria-label="Image view">
        {modes.map(({ id, label, Icon }) => <button key={id} aria-pressed={view === id}
          onClick={() => setView(id)}><Icon size={14} />{label}</button>)}
      </div>
      <div className="influence-legend"><span>Lower influence</span><i aria-hidden="true" /><span>Higher influence</span></div>
      <p className="inspector-note" id="comparison-help">{view === 'compare' ? 'Drag the divider or use arrow keys to compare. ' : ''}Warm highlights show model influence, not verified defects. Dimmed borders, when present, were not analysed.</p>
    </> : <div className="inspector-idle"><span className="status-dot" />{busy ? 'Your selected evidence checks are running' : 'Image loaded. Ready for a closer look.'}</div>}
  </div>
}
