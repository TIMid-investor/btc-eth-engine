# BTC Market Analysis — Reusable Agent Prompt

Use this with Claude Code's Agent tool (general-purpose subagent) to regenerate a fresh BTC analysis.

---

## Agent Prompt

```
Research the following for a comprehensive Bitcoin market analysis as of today's date:

1. **BTC Price Action**: Current price, recent price movement (last 7–30 days), today's specific move — how much did it go up or down, from what level, intraday range

2. **Volume Analysis**: Today's trading volume vs recent averages — is volume confirming the move or is it low/suspicious? Any notable volume spikes in the past week?

3. **Technical Levels**: Key support/resistance levels, 200-day MA location and distance from current price, any notable chart patterns (double bottom, head and shoulders, wedge, consolidation range). Include Williams Alligator, MACD, and RSI readings if available.

4. **On-Chain Data**:
   - Exchange reserves (BTC on exchanges — direction and historical context)
   - Whale wallet accumulation/distribution (1,000+ BTC wallets)
   - MVRV ratio (current reading and what it means vs cycle tops/bottoms)
   - SOPR (current reading — is it above or below 1.0)
   - Miner behavior: hash rate, average production cost vs spot price, are miners selling?
   - Long-term holder (LTH) behavior — accumulating or distributing?

5. **Macro Context**:
   - Current Fed funds rate and most recent FOMC decision/language
   - Latest CPI/PCE inflation readings (headline and core, YoY)
   - Latest US GDP reading — is growth slowing?
   - Stagflation risk assessment
   - Any major geopolitical events affecting risk sentiment (wars, trade wars, tariffs, sanctions)
   - Oil price and what's driving it

6. **Crypto Market Sentiment**:
   - Fear & Greed Index (current reading and how many consecutive days at this level)
   - Funding rates (positive = longs paying shorts = crowded longs; negative = shorts paying longs = crowded shorts)
   - Open interest — rising or falling?
   - Any major liquidation events recently?

7. **BTC Dominance**: Current % and direction. Is altseason starting or is this still BTC's market?

8. **Institutional Flows**:
   - Most recent day's US spot Bitcoin ETF net flows (IBIT, FBTC, total)
   - Weekly ETF flow trend
   - Any major institutional buys or sells announced?
   - Cumulative ETF AUM/inflows total

9. **Historical Cycle Comparisons**:
   - Current % drawdown from the most recent ATH and when that ATH occurred
   - How does this compare to prior cycle drawdowns (2017, 2021)?
   - Where are we in the 4-year halving cycle? What does cycle timing imply for bottoming?
   - Any analysts citing specific historical analogs?

10. **Expert & Analyst Views**:
    - Any notable on-chain analysts (Willy Woo, Glassnode, PlanB) with recent calls
    - Any major banks or crypto research desks with Q2/Q3 price targets
    - Contrarian vs consensus divide — what are the bears saying vs the bulls?

Search for the most recent data available. Be thorough and use specific numbers. Cite sources.

Deliver the report in this structure:
- Bottom Line Up Front (2–3 sentences: confirmed bottom/breakout or not?)
- Today's Move Analysis (is it a breakout? volume?)
- Technical Picture (key levels table + indicator readings)
- On-Chain: The Bullish Case
- Macro: The Bearish Case
- Are We at or Near a Bottom? (bull case vs bear case)
- Key Levels/Signals to Watch
- Summary Scorecard (factor | signal | bias table)
- Net Read (final synthesis paragraph)
```

---

## How to Use

In a Claude Code conversation, say:

> "Run the BTC analysis agent prompt from `projects/crypto/reports/btc_analysis_agent_prompt.md`"

Or paste the prompt directly into a conversation and ask Claude to run it via web search.
