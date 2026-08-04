import './GameCard.css'

const CONFIDENCE_COLOR = {
  low: '#7d8590',
  medium: '#d29922',
  high: '#2ea043',
}

const MODEL_LABEL = {
  v2: 'v2 · pitcher',
  v1: 'v1 · rolling',
}

function americanToImplied(odds) {
  if (odds == null) return null
  if (odds < 0) return (-odds) / (-odds + 100)
  return 100 / (odds + 100)
}

function getValueAlerts(homeTeam, awayTeam, bestOdds, prediction) {
  if (!bestOdds || !prediction) return []
  const alerts = []
  const teams = [
    { name: homeTeam, modelProb: prediction.home_win_prob, odds: bestOdds[homeTeam] },
    { name: awayTeam, modelProb: prediction.away_win_prob, odds: bestOdds[awayTeam] },
  ]
  for (const { name, modelProb, odds } of teams) {
    const implied = americanToImplied(odds)
    if (implied == null) continue
    const edge = modelProb - implied
    if (edge >= 0.05) {
      alerts.push({ team: name, modelProb, implied, edge })
    }
  }
  return alerts
}

function formatAmericanOdds(n) {
  if (n == null) return '—'
  return n > 0 ? `+${n}` : `${n}`
}

function formatTime(iso) {
  const d = new Date(iso)
  return d.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', timeZoneName: 'short' })
}

function ProbBar({ homeProb, awayProb, homeTeam, awayTeam }) {
  const homePct = Math.round(homeProb * 100)
  const awayPct = Math.round(awayProb * 100)
  return (
    <div className="prob-bar-wrapper">
      <span className="prob-label">{awayTeam.split(' ').at(-1)} {awayPct}%</span>
      <div className="prob-bar">
        <div className="prob-away" style={{ width: `${awayPct}%` }} />
        <div className="prob-home" style={{ width: `${homePct}%` }} />
      </div>
      <span className="prob-label">{homePct}% {homeTeam.split(' ').at(-1)}</span>
    </div>
  )
}

function ModelPrediction({ modelKey, prediction, homeTeam, awayTeam, bestOdds }) {
  const winner     = prediction?.predicted_winner
  const confidence = prediction?.confidence
  const alerts     = getValueAlerts(homeTeam, awayTeam, bestOdds, prediction)
  const label      = MODEL_LABEL[modelKey] || prediction?.model_used || modelKey

  return (
    <div className="model-prediction">
      <div className="prediction-header">
        <span className="pred-label">{label}</span>
        <span className="confidence-badge" style={{ color: CONFIDENCE_COLOR[confidence] }}>
          {confidence} confidence
        </span>
      </div>
      <ProbBar
        homeProb={prediction.home_win_prob}
        awayProb={prediction.away_win_prob}
        homeTeam={homeTeam}
        awayTeam={awayTeam}
      />
      <p className="predicted-winner-text">
        Predicted winner:{' '}
        <strong style={{ color: CONFIDENCE_COLOR[confidence] }}>{winner}</strong>
      </p>
      {alerts.map(({ team, modelProb, implied, edge }) => (
        <div key={team} className="value-alert">
          <span className="value-icon">★</span>
          <span>
            <strong>{team.split(' ').at(-1)}</strong>{' '}
            model ({(modelProb * 100).toFixed(1)}%) vs implied ({(implied * 100).toFixed(1)}%)
            {' '}— <strong>+{(edge * 100).toFixed(1)}% edge</strong>
          </span>
        </div>
      ))}
    </div>
  )
}

function OddsRow({ bookmaker, homeTeam, awayTeam, prices }) {
  const homeOdds = prices?.[homeTeam]
  const awayOdds = prices?.[awayTeam]
  return (
    <tr>
      <td className="book-name">{bookmaker}</td>
      <td className={`odds-cell ${awayOdds < 0 ? 'fav' : 'dog'}`}>{formatAmericanOdds(awayOdds)}</td>
      <td className={`odds-cell ${homeOdds < 0 ? 'fav' : 'dog'}`}>{formatAmericanOdds(homeOdds)}</td>
    </tr>
  )
}

function PitcherBadge({ name }) {
  if (!name) return <span className="pitcher-name unknown">TBD</span>
  const last = name.split(' ').slice(1).join(' ') || name
  return <span className="pitcher-name">{last}</span>
}

export default function GameCard({ game }) {
  const {
    home_team, away_team, commence_time, odds, best_odds,
    predictions,
    home_pitcher, away_pitcher,
  } = game

  const bookmakers  = Object.keys(odds || {})
  const hasPitchers = home_pitcher || away_pitcher
  const modelKeys   = Object.keys(predictions || {})

  return (
    <div className="game-card">
      {/* Teams header */}
      <div className="teams-row">
        <div className="team">
          <span className="team-name">{away_team}</span>
          <span className="team-role">Away</span>
          {hasPitchers && <PitcherBadge name={away_pitcher} />}
        </div>
        <div className="vs-block">
          <span className="vs">@</span>
          <span className="game-time">{formatTime(commence_time)}</span>
        </div>
        <div className="team team-right">
          <span className="team-name">{home_team}</span>
          <span className="team-role">Home</span>
          {hasPitchers && <PitcherBadge name={home_pitcher} />}
        </div>
      </div>

      {/* Per-model predictions */}
      {modelKeys.length > 0 && (
        <div className={`predictions-section ${modelKeys.length > 1 ? 'multi-model' : ''}`}>
          {modelKeys.map(key => (
            <ModelPrediction
              key={key}
              modelKey={key}
              prediction={predictions[key]}
              homeTeam={home_team}
              awayTeam={away_team}
              bestOdds={best_odds}
            />
          ))}
        </div>
      )}

      {/* Odds table */}
      {bookmakers.length > 0 ? (
        <div className="odds-section">
          <table className="odds-table">
            <thead>
              <tr>
                <th>Sportsbook</th>
                <th>{away_team.split(' ').at(-1)}</th>
                <th>{home_team.split(' ').at(-1)}</th>
              </tr>
            </thead>
            <tbody>
              {bookmakers.map(book => (
                <OddsRow
                  key={book}
                  bookmaker={book}
                  homeTeam={home_team}
                  awayTeam={away_team}
                  prices={odds[book]}
                />
              ))}
              {best_odds && Object.keys(best_odds).length > 0 && (
                <tr className="best-row">
                  <td>Best Available</td>
                  <td className={`odds-cell ${best_odds[away_team] < 0 ? 'fav' : 'dog'}`}>
                    {formatAmericanOdds(best_odds[away_team])}
                  </td>
                  <td className={`odds-cell ${best_odds[home_team] < 0 ? 'fav' : 'dog'}`}>
                    {formatAmericanOdds(best_odds[home_team])}
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="no-odds">No odds available yet</p>
      )}
    </div>
  )
}
