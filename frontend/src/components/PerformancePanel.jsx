import { useState, useEffect } from 'react'
import './PerformancePanel.css'

function pct(n) {
  return n == null ? '—' : `${(n * 100).toFixed(1)}%`
}

function StatBox({ label, value, sub }) {
  return (
    <div className="stat-box">
      <span className="stat-value">{value}</span>
      <span className="stat-label">{label}</span>
      {sub && <span className="stat-sub">{sub}</span>}
    </div>
  )
}

const MODEL_DISPLAY = {
  v2: 'v2 · pitcher-aware',
  v1: 'v1 · rolling team stats',
}

function ModelStats({ modelKey, stats }) {
  const o = stats?.overall
  const v = stats?.value_picks
  const label = MODEL_DISPLAY[modelKey] || modelKey

  return (
    <div className="model-stats">
      <h3 className="model-stats-title">Model {label}</h3>
      <div className="stat-row">
        <StatBox label="Games" value={o?.games ?? '—'} />
        <StatBox
          label="Overall accuracy"
          value={pct(o?.accuracy)}
          sub={o?.games ? `${o.correct}/${o.games} correct` : null}
        />
        <StatBox
          label="Value pick accuracy"
          value={pct(v?.accuracy)}
          sub={v?.games ? `${v.correct}/${v.games}` : 'No value picks'}
        />
        <StatBox
          label="Value pick ROI"
          value={v?.roi != null ? `${(v.roi * 100).toFixed(1)}%` : '—'}
          sub="$100/game"
        />
      </div>

      {stats?.calibration?.length > 0 && (
        <div className="calibration">
          <p className="cal-title">Calibration (confidence vs actual win rate)</p>
          <div className="cal-rows">
            {stats.calibration.map(c => {
              const fill = Math.round(c.actual_win_rate * 100)
              return (
                <div key={c.bucket} className="cal-row">
                  <span className="cal-bucket">{c.bucket}</span>
                  <div className="cal-bar-wrap">
                    <div className="cal-bar" style={{ width: `${fill}%` }} />
                  </div>
                  <span className="cal-rate">{pct(c.actual_win_rate)}</span>
                  <span className="cal-n">({c.games}g)</span>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}

export default function PerformancePanel() {
  const [data, setData]       = useState(null)
  const [loading, setLoading] = useState(true)
  const [open, setOpen]       = useState(false)

  const load = async () => {
    setLoading(true)
    try {
      const res = await fetch('/api/performance')
      setData(await res.json())
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (open && !data) load()
  }, [open])

  const byModel   = data?.by_model || {}
  const modelKeys = Object.keys(byModel)
  // Summary badge uses the first available model's overall accuracy
  const primary   = byModel[modelKeys[0]]

  return (
    <div className="perf-panel">
      <button className="perf-toggle" onClick={() => setOpen(x => !x)}>
        {open ? '▲' : '▼'} Model Performance
        {primary && !loading && (
          <span className="perf-badge">
            {pct(primary.overall?.accuracy)} overall
            {primary.value_picks?.games
              ? ` · ${pct(primary.value_picks?.accuracy)} on value picks`
              : ''}
          </span>
        )}
      </button>

      {open && (
        <div className="perf-body">
          {loading ? (
            <p className="perf-loading">Resolving results…</p>
          ) : !data || data.resolved_games === 0 ? (
            <p className="perf-empty">No resolved games yet. Check back after today's games finish.</p>
          ) : (
            <>
              <p className="perf-subtitle">{data.resolved_games} resolved games</p>
              {modelKeys.map(key => (
                <ModelStats key={key} modelKey={key} stats={byModel[key]} />
              ))}
              <button className="perf-refresh" onClick={load}>Refresh</button>
            </>
          )}
        </div>
      )}
    </div>
  )
}
