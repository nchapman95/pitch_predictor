import { useState, useEffect, useCallback } from 'react'
import GameCard from './components/GameCard'
import './App.css'

const REFRESH_INTERVAL = 60_000 // 60 seconds

export default function App() {
  const [games, setGames] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [lastUpdated, setLastUpdated] = useState(null)
  const [countdown, setCountdown] = useState(REFRESH_INTERVAL / 1000)

  const fetchGames = useCallback(async () => {
    try {
      const res = await fetch('/api/games')
      if (!res.ok) throw new Error(`Server error: ${res.status}`)
      const data = await res.json()
      setGames(data.games || [])
      setLastUpdated(new Date())
      setError(null)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
      setCountdown(REFRESH_INTERVAL / 1000)
    }
  }, [])

  // Initial fetch + interval refresh
  useEffect(() => {
    fetchGames()
    const interval = setInterval(fetchGames, REFRESH_INTERVAL)
    return () => clearInterval(interval)
  }, [fetchGames])

  // Countdown timer
  useEffect(() => {
    const tick = setInterval(() => {
      setCountdown(c => (c > 0 ? c - 1 : 0))
    }, 1000)
    return () => clearInterval(tick)
  }, [lastUpdated])

  const today = new Date().toLocaleDateString('en-US', {
    weekday: 'long', month: 'long', day: 'numeric', year: 'numeric',
  })

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <div>
            <h1 className="title">⚾ MLB Game Predictor</h1>
            <p className="subtitle">{today}</p>
          </div>
          <div className="refresh-info">
            {lastUpdated && (
              <span className="last-updated">
                Updated {lastUpdated.toLocaleTimeString()}
              </span>
            )}
            <button className="refresh-btn" onClick={fetchGames} disabled={loading}>
              {loading ? 'Loading…' : `Refresh (${countdown}s)`}
            </button>
          </div>
        </div>
      </header>

      <main className="main">
        {error && (
          <div className="error-banner">
            {error.includes('502') || error.includes('Odds API')
              ? '⚠️ Could not reach odds API. Check your ODDS_API_KEY in .env'
              : `⚠️ ${error}`}
          </div>
        )}

        {loading && games.length === 0 ? (
          <div className="loading-state">Loading today's games…</div>
        ) : games.length === 0 ? (
          <div className="empty-state">
            <p>No MLB games scheduled for today.</p>
            <p className="muted">Check back on a game day, or verify your API key.</p>
          </div>
        ) : (
          <div className="games-grid">
            {games.map(game => (
              <GameCard key={game.id} game={game} />
            ))}
          </div>
        )}
      </main>

      <footer className="footer">
        <span>Odds from The Odds API · Predictions from ML model trained on pybaseball data</span>
        <span className="muted"> · Not financial/betting advice</span>
      </footer>
    </div>
  )
}
