# Data Directory

## Overview

This directory contains the training data for the News Topic Classifier.

## File Structure

| File | Description |
|------|-------------|
| `sample_data.csv` | Synthetic news article snippets with category labels |

## CSV Format

The `sample_data.csv` file has two columns:

| Column | Type | Description |
|--------|------|-------------|
| `text` | string | 2–4 sentence news article snippet |
| `category` | string | One of: `sports`, `technology`, `politics`, `business`, `entertainment`, `health`, `science` |

### Example rows

```
text,category
"The home team clinched the championship title with a stunning 3-1 victory...",sports
"A Silicon Valley startup has unveiled a new AI chip...",technology
```

## Data Statistics

- **Total rows:** 180
- **Categories:** 7
- **Rows per category:** ≥ 25

## Data Collection Methodology

### Synthetic Data (current)

The current dataset (`sample_data.csv`) consists of synthetically generated news article
snippets that mimic real reporting style. Each snippet is 2–4 sentences covering a
plausible news story in one of the seven categories.

### Web Scraping (production approach)

For production-scale data collection, the `src/scraper.py` module implements a
web scraping pipeline:

1. **RSS Feed Parsing** — `NewsScraper.scrape_feed(rss_url)` parses standard RSS feeds
   to discover article URLs and metadata (title, description, link).

2. **Article Extraction** — `NewsScraper.scrape_article(url)` retrieves each article
   page and extracts:
   - **Title**: detected via common CSS selectors (`h1.article-title`, `h1`, `title`)
   - **Body**: detected via article container selectors (`article`, `div.article-body`,
     `div.entry-content`, `main`)

3. **Labelling** — `NewsScraper.collect_training_data(urls, category)` tags each
   scraped article with its known category.

4. **Persistence** — `NewsScraper.save_to_csv(data, path)` writes the collected
   articles to a CSV with `text` and `category` columns.

### Suggested RSS Sources

| Category | Feed URL |
|----------|---------|
| Sports | `http://rss.cnn.com/rss/edition_sport.rss` |
| Technology | `https://feeds.feedburner.com/TechCrunch` |
| Politics | `https://rss.nytimes.com/services/xml/rss/nyt/Politics.xml` |
| Business | `https://feeds.bloomberg.com/markets/news.rss` |
| Entertainment | `https://variety.com/feed/` |
| Health | `https://www.medicalnewstoday.com/rss` |
| Science | `https://www.sciencedaily.com/rss/top.xml` |

### Ethical Considerations

- Requests are rate-limited (default 1-second delay between requests).
- The scraper identifies itself with a descriptive `User-Agent` string.
- Only public articles are scraped; paywalled content is skipped.
- Check each site's `robots.txt` before scraping at scale.
