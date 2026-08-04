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

export default function PerformancePanel() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [open, setOpen] = useState(false)

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

  const o = data?.overall
  const v = data?.value_picks

  return (
    <div className="perf-panel">
      <button className="perf-toggle" onClick={() => setOpen(x => !x)}>
        {open ? '▲' : '▼'} Model Performance
        {o && !loading && (
          <span className="perf-badge">{pct(o.accuracy)} overall · {pct(v?.accuracy)} on value picks</span>
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
              <div className="stat-row">
                <StatBox label="Games tracked" value={data.resolved_games} />
                <StatBox
                  label="Overall accuracy"
                  value={pct(o?.accuracy)}
                  sub={`${o?.correct}/${o?.games} correct`}
                />
                <StatBox
                  label="Value pick accuracy"
                  value={pct(v?.accuracy)}
                  sub={v?.games ? `${v.correct}/${v.games} correct` : 'No value picks yet'}
                />
                <StatBox
                  label="Value pick ROI"
                  value={v?.roi != null ? `${(v.roi * 100).toFixed(1)}%` : '—'}
                  sub="$100/game on predicted winner"
                />
              </div>

              {data.calibration?.length > 0 && (
                <div className="calibration">
                  <p className="cal-title">Calibration (model confidence vs actual win rate)</p>
                  <div className="cal-rows">
                    {data.calibration.map(c => {
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

              <button className="perf-refresh" onClick={load}>Refresh</button>
            </>
          )}
        </div>
      )}
    </div>
  )
}
