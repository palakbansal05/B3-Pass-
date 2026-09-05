import { useState } from 'react'
import './App.css'

const examples = [
  { label: 'Caffeine', smiles: 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C' },
  { label: 'Ibuprofen', smiles: 'CC(C)CC1=CC=C(C=C1)[C@@H](C)C(=O)O' },
  { label: 'Aspirin', smiles: 'CC(=O)OC1=CC=CC=C1C(=O)O' },
]

function formatDescriptor(value, digits = 2) {
  return typeof value === 'number' ? value.toFixed(digits) : '—'
}

function App() {
  const [smiles, setSmiles] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function analyzeMolecule(event) {
    event.preventDefault()
    if (!smiles.trim()) {
      setError('Enter a SMILES string to begin the analysis.')
      return
    }

    setLoading(true)
    setError('')
    setResult(null)
    try {
      const apiBase = import.meta.env.VITE_API_URL || ''
      const response = await fetch(`${apiBase}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ smiles: smiles.trim() }),
      })
      const responseText = await response.text()
      let data = {}
      if (responseText.trim()) {
        try {
          data = JSON.parse(responseText)
        } catch {
          throw new Error(`Prediction service returned an invalid response (${response.status}).`)
        }
      }
      if (!response.ok) {
        throw new Error(data.error || `Prediction service returned HTTP ${response.status}.`)
      }
      setResult(data)
    } catch (requestError) {
      setError(requestError instanceof TypeError
        ? 'Could not connect to the prediction service. Start the Flask backend on port 5000.'
        : requestError.message || 'Could not connect to the prediction service.')
    } finally {
      setLoading(false)
    }
  }

  const probability = result?.prediction ?? null
  const passes = probability !== null && probability >= 0.5
  const confidence = probability === null ? null : Math.round(Math.abs(probability - 0.5) * 200)

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="/" aria-label="B3-Pass home">
          <span className="brand-mark">B3</span>
          <span><strong>B3-Pass</strong><small>Blood-brain barrier intelligence</small></span>
        </a>
        <div className="topbar-meta"><span className="status-dot" /> Model online <span className="meta-divider" /> Research workspace</div>
      </header>

      <main className="workspace">
        <section className="intro-row">
          <div>
            <p className="eyebrow">Molecule screening / 01</p>
            <h1>Can this molecule<br /><em>reach the brain?</em></h1>
            <p className="intro-copy">A fast, evidence-led BBB permeability screen for early drug discovery and translational research.</p>
          </div>
          <div className="intro-note"><span className="note-number">01</span><span>Predict<br />Interpret<br />Decide</span></div>
        </section>

        <section className="analysis-grid">
          <div className="input-panel panel">
            <div className="panel-heading"><div><span className="section-index">01</span><h2>Define molecule</h2></div><span className="panel-label">SMILES input</span></div>
            <form onSubmit={analyzeMolecule}>
              <label htmlFor="smiles">Canonical or isomeric SMILES</label>
              <div className={`input-wrap ${error ? 'has-error' : ''}`}>
                <textarea id="smiles" value={smiles} onChange={(event) => setSmiles(event.target.value)} placeholder="Paste a SMILES string here..." rows="4" spellCheck="false" />
                <span className="char-count">{smiles.length} chars</span>
              </div>
              {error && <p className="error-message" role="alert">{error}</p>}
              <button className="analyze-button" type="submit" disabled={loading}>
                {loading ? <><span className="spinner" /> Running screen</> : <>Run BBB screen <span>↗</span></>}
              </button>
            </form>
            <div className="examples"><span>Try an example</span>{examples.map((example) => <button key={example.label} type="button" onClick={() => { setSmiles(example.smiles); setError('') }}>{example.label}</button>)}</div>
            <div className="input-footnote"><span className="lock-symbol">◎</span> Your molecule stays in this analysis session</div>
          </div>

          <div className={`result-panel panel ${result ? 'has-result' : ''}`}>
            <div className="panel-heading"><div><span className="section-index">02</span><h2>Screening result</h2></div><span className="panel-label">Model output</span></div>
            {!result && !loading && <div className="empty-result"><div className="molecule-mark" aria-hidden="true"><span /><span /><span /><i /><i /></div><h3>Waiting for a molecule</h3><p>Submit a SMILES string to see permeability probability, physicochemical descriptors, and a research note.</p></div>}
            {loading && <div className="empty-result loading-result"><div className="loading-ring" /><h3>Reading molecular profile</h3><p>Calculating descriptors and retrieving relevant literature.</p></div>}
            {result && <div className="result-content">
              <div className={`verdict ${passes ? 'pass' : 'hold'}`}><span className="verdict-icon">{passes ? '↑' : '↓'}</span><div><span className="verdict-kicker">Predicted to {passes ? 'cross' : 'remain outside'} the BBB</span><strong>{passes ? 'Brain-access likely' : 'Brain-access unlikely'}</strong></div><span className="verdict-score">{Math.round(probability * 100)}<small>%</small></span></div>
              <div className="meter-block"><div className="meter-labels"><span>Non-permeable</span><span>Permeable</span></div><div className="meter"><span style={{ width: `${Math.max(3, probability * 100)}%` }} /></div><div className="meter-value"><strong>{(probability * 100).toFixed(1)}%</strong> probability of BBB permeability <span>·</span> confidence {confidence}%</div></div>
              <div className="decision-callout"><span className="callout-icon">✦</span><p><strong>What this means</strong>{passes ? ' This profile is more consistent with a candidate intended for a central nervous system target.' : ' This profile is more consistent with a candidate intended for a peripheral target.'} Treat the screen as an early filter, not a substitute for in vitro or in vivo validation.</p></div>
              <div className="result-divider" />
              <div className="descriptor-header"><span>Physicochemical profile</span><small>{result.out_of_distribution ? 'Outside training range' : 'Within training range'}</small></div>
              <div className="descriptor-grid">{[['Mol. weight', result.descriptors?.MolWt, 'Da'], ['LogP', result.descriptors?.LogP, ''], ['TPSA', result.descriptors?.TPSA, 'Å²'], ['H-bond donors', result.descriptors?.NumHDonors, ''], ['H-bond acceptors', result.descriptors?.NumHAcceptors, ''], ['Rotatable bonds', result.descriptors?.NumRotatableBonds, '']].map(([label, value, unit]) => <div className="descriptor" key={label}><span>{label}</span><strong>{formatDescriptor(value, label === 'LogP' ? 2 : 1)}<small>{unit}</small></strong></div>)}</div>
            </div>}
          </div>
        </section>

        {result?.report && <section className="report-panel"><div className="report-tag">03 / Research note</div><div><h2>Model interpretation</h2><p>{result.report}</p></div><span className="report-mark">“</span></section>}
        <footer><span>B3-PASS / BBB permeability screening</span><span>For research use · Validate experimentally before making development decisions</span></footer>
      </main>
    </div>
  )
}

export default App