# Reddit Sentiment Setup

Waiting on Reddit API access approval. Once approved:

## 1. Get credentials

Go to **reddit.com/prefs/apps** → "create another app"
- Type: **script**
- Name: anything (e.g. crypto-engine)
- Redirect URI: `http://localhost`

Copy the `client_id` (shown under the app name) and `client_secret`.

## 2. Set env vars

Add to `~/.zshrc`:

```bash
export REDDIT_CLIENT_ID=your_client_id
export REDDIT_CLIENT_SECRET=your_client_secret
export REDDIT_USER_AGENT="crypto-engine/1.0 by u/yourusername"
```

Then: `source ~/.zshrc`

## 3. Install dependencies

```bash
cd ~/crypto && source venv/bin/activate

pip install praw                                      # required
pip install vaderSentiment transformers torch         # better sentiment (downloads ~400MB FinBERT on first run)
pip install bertopic sentence-transformers umap-learn hdbscan  # optional deep topics
```

## 4. Run

```bash
python scripts/run_reddit.py --fetch        # fetch fresh data + show dashboard
python scripts/run_reddit.py                # use cached data
python scripts/run_reddit.py --no-finbert   # skip FinBERT, use VADER only (fast)
python scripts/run_reddit.py --fetch --bertopic  # deep topic discovery
```

Dashboard covers:
- **A. Volume & Attention** — post spikes, comment velocity, unique users
- **B. Sentiment Polarity** — bullish/bearish %, emotion breakdown (euphoria / FOMO / FUD / capitulation)
- **C. Narrative Phase** — cycle phase detection (capitulation → skepticism → recovery → optimism → euphoria)
