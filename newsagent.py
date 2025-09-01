import os
import time
import json
import sqlite3
import feedparser
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import schedule
from dotenv import load_dotenv
import requests
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch
import base64
from io import BytesIO
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('NewsAIAgent')

# Load environment variables
load_dotenv()

class AgentState(Enum):
    IDLE = "idle"
    LEARNING = "learning"
    FETCHING = "fetching"
    PROCESSING = "processing"
    ADAPTING = "adapting"
    RESPONDING = "responding"

@dataclass
class UserPreferences:
    user_id: str
    keywords: List[str]
    excluded_topics: List[str]
    preferred_sources: List[str]
    email_frequency: str  # daily, weekly, breaking
    max_articles: int
    feedback_score: float
    last_interaction: datetime

@dataclass
class ArticleFeedback:
    article_url: str
    user_id: str
    rating: int  # 1-5 stars
    feedback_type: str  # 'helpful', 'not_relevant', 'too_long', 'spam'
    timestamp: datetime

@dataclass
class AgentMemory:
    successful_queries: List[str]
    failed_sources: List[str]
    user_interactions: List[Dict]
    performance_metrics: Dict[str, float]
    learned_patterns: Dict[str, Any]

class EnhancedNewsAIAgent:
    def __init__(self):
        print("\n" + "="*60)
        print("🤖 AI NEWS AGENT - ENHANCED VERSION")
        print("="*60)
        
        # Display main search keywords at startup
        self._display_startup_info()
        
        # Initialize core components
        self.state = AgentState.IDLE
        self.memory = self._load_memory()
        self.user_preferences = {}
        self.performance_metrics = {
            'emails_sent': 0,
            'user_satisfaction': 0.0,
            'articles_processed': 0,
            'api_success_rate': 0.0,
            'total_sources': 0
        }
        
        # Main search keywords
        self.main_keywords = [
            "artificial intelligence",
            "machine learning",
            "AI industry news", 
            "deep learning",
            "neural networks",
            "AI breakthrough",
            "tech innovation",
            "AI research"
        ]
        
        # RSS feeds for additional sources
        self.rss_feeds = {
            'MIT AI News': 'https://news.mit.edu/topic/mitartificial-intelligence2-rss.xml',
            'AI News': 'https://artificialintelligence-news.com/feed/',
            'VentureBeat AI': 'https://venturebeat.com/ai/feed/',
            'Wired AI': 'https://www.wired.com/feed/tag/ai/latest/rss',
            'TechCrunch AI': 'https://techcrunch.com/category/artificial-intelligence/feed/',
            'IEEE Spectrum AI': 'https://spectrum.ieee.org/rss/blog/artificial-intelligence',
            'AI Research': 'https://www.marktechpost.com/feed/',
            'OpenAI Blog': 'https://openai.com/blog/rss/',
        }
        
        print(f"📡 Initialized with {len(self.rss_feeds)} RSS feeds")
        
        # Initialize AI components
        device = 0 if torch.cuda.is_available() else -1
        print(f"🔧 Loading AI models on {'GPU' if device == 0 else 'CPU'}...")
        
        self.summarizer = pipeline("summarization", 
                                 model="sshleifer/distilbart-cnn-12-6", 
                                 device=device)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2', 
                                          device='cuda' if device == 0 else 'cpu')
        
        print("✅ AI models loaded successfully")
        
        # API keys
        self.api_keys = {
            'newsdata': os.getenv('NEWSDATA_API_KEY'),
            'gnews': os.getenv('GNEWS_API_KEY'),
            'mediastack': os.getenv('MEDIASTACK_API_KEY'),
            'worldnews': os.getenv('WORLDNEWS_API_KEY'),  # New API
            'gmail_user': os.getenv('GMAIL_USER'),
            'gmail_password': os.getenv('GMAIL_APP_PASSWORD')
        }
        
        # Count available API keys
        available_apis = sum(1 for key in self.api_keys.values() if key)
        print(f"🔑 {available_apis} API keys configured")
        
        # Initialize database
        self._init_database()
        
        # Start autonomous scheduling
        self._setup_autonomous_schedule()
        
        print("🚀 AI News Agent is ready!")
        print("="*60)
        logger.info("Enhanced NewsAI Agent initialized successfully")
    
    def _display_startup_info(self):
        """Display startup information with main keywords"""
        print("\n📋 MAIN SEARCH KEYWORDS:")
        keywords = [
            "🔹 Artificial Intelligence",
            "🔹 Machine Learning", 
            "🔹 AI Industry News",
            "🔹 Deep Learning",
            "🔹 Neural Networks",
            "🔹 AI Breakthrough",
            "🔹 Tech Innovation",
            "🔹 AI Research"
        ]
        
        for keyword in keywords:
            print(f"   {keyword}")
        
        print(f"\n📰 NEWS SOURCES:")
        sources = [
            "✅ NewsData.io API",
            "✅ GNews API", 
            "✅ Mediastack API",
            "✅ MIT AI News RSS",
            "✅ VentureBeat AI RSS",
            "✅ Wired AI RSS",
            "✅ TechCrunch AI RSS",
            "✅ IEEE Spectrum RSS",
            "✅ MarkTechPost RSS",
            "✅ OpenAI Blog RSS"
        ]
        
        for source in sources:
            print(f"   {source}")
        
        print(f"\n🎯 TARGET: Fetch 8-12 unique AI articles daily")
        print(f"⚙️  FEATURES: GPU Acceleration, Learning, Chat Interface")
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        self.conn = sqlite3.connect('news_agent.db', check_same_thread=False)
        cursor = self.conn.cursor()
        
        # User preferences table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT PRIMARY KEY,
                keywords TEXT,
                excluded_topics TEXT,
                preferred_sources TEXT,
                email_frequency TEXT,
                max_articles INTEGER,
                feedback_score REAL,
                last_interaction TEXT
            )
        ''')
        
        # Feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS article_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_url TEXT,
                user_id TEXT,
                rating INTEGER,
                feedback_type TEXT,
                timestamp TEXT
            )
        ''')
        
        # Performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                metric_name TEXT PRIMARY KEY,
                value REAL,
                last_updated TEXT
            )
        ''')
        
        # Agent memory table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS agent_memory (
                key TEXT PRIMARY KEY,
                value TEXT,
                last_updated TEXT
            )
        ''')
        
        self.conn.commit()
    
    def _load_memory(self) -> AgentMemory:
        """Load agent memory from persistent storage"""
        try:
            with open('agent_memory.json', 'r') as f:
                data = json.load(f)
                return AgentMemory(**data)
        except FileNotFoundError:
            return AgentMemory(
                successful_queries=[],
                failed_sources=[],
                user_interactions=[],
                performance_metrics={},
                learned_patterns={}
            )
    
    def _save_memory(self):
        """Save agent memory to persistent storage"""
        with open('agent_memory.json', 'w') as f:
            json.dump(asdict(self.memory), f, default=str, indent=2)
    
    def _setup_autonomous_schedule(self):
        """Set up autonomous scheduling based on user preferences"""
        schedule.every().day.at("08:00").do(self._autonomous_news_cycle)
        schedule.every().hour.do(self._check_breaking_news)
        schedule.every(30).minutes.do(self._health_check)
        
        # Start scheduler in separate thread
        def run_scheduler():
            while True:
                schedule.run_pending()
                time.sleep(1)
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        logger.info("Autonomous scheduling initialized")
    
    def _autonomous_news_cycle(self):
        """Main autonomous news processing cycle"""
        try:
            self.state = AgentState.FETCHING
            logger.info("Starting autonomous news cycle")
            
            # Adapt queries based on learned patterns
            queries = self._adapt_queries_from_memory()
            
            # Fetch and process articles from multiple sources
            articles = self._fetch_articles_from_all_sources(queries)
            
            if articles:
                self.state = AgentState.PROCESSING
                summaries = self._process_articles_with_learning(articles)
                
                if summaries:
                    self.state = AgentState.RESPONDING
                    self._send_personalized_emails(summaries)
                    
                    # Update performance metrics
                    self._update_performance_metrics(summaries)
            
            self.state = AgentState.IDLE
            logger.info("Autonomous news cycle completed")
            
        except Exception as e:
            logger.error(f"Error in autonomous news cycle: {e}")
            self.state = AgentState.IDLE
    
    def _adapt_queries_from_memory(self) -> List[str]:
        """Intelligently adapt search queries based on past performance"""
        base_queries = self.main_keywords.copy()
        
        # Add successful queries from memory
        successful_queries = self.memory.successful_queries[-5:]  # Last 5 successful
        
        # Remove failed patterns
        adapted_queries = []
        for query in base_queries + successful_queries:
            if query not in self.memory.failed_sources:
                adapted_queries.append(query)
        
        # Add trending topics based on user feedback
        if self.memory.learned_patterns.get('trending_topics'):
            adapted_queries.extend(self.memory.learned_patterns['trending_topics'][:2])
        
        return list(set(adapted_queries))  # Remove duplicates
    
    def _fetch_articles_from_all_sources(self, queries: List[str]) -> List[Dict]:
        """Fetch articles from all available sources"""
        all_articles = []
        
        print(f"🔍 Fetching articles for {len(queries)} queries...")
        
        # Fetch from API sources
        for query in queries[:3]:  # Limit to 3 queries to respect rate limits
            # NewsData.io
            articles = self._fetch_newsdata(query, n=4)
            if articles:
                all_articles.extend(articles)
                logger.info(f"NewsData: {len(articles)} articles for '{query}'")
            
            # GNews
            articles = self._fetch_gnews(query, n=4)
            if articles:
                all_articles.extend(articles)
                logger.info(f"GNews: {len(articles)} articles for '{query}'")
            
            # Mediastack
            articles = self._fetch_mediastack(query, n=4)
            if articles:
                all_articles.extend(articles)
                logger.info(f"Mediastack: {len(articles)} articles for '{query}'")
            
            time.sleep(1)  # Rate limiting
        
        # Fetch from RSS sources
        rss_articles = self._fetch_from_rss_feeds()
        if rss_articles:
            all_articles.extend(rss_articles)
            logger.info(f"RSS: {len(rss_articles)} articles from feeds")
        
        print(f"📊 Total articles fetched: {len(all_articles)}")
        
        return self._deduplicate_articles(all_articles)
    
    def _fetch_from_rss_feeds(self) -> List[Dict]:
        """Fetch articles from RSS feeds"""
        all_articles = []
        
        for source_name, feed_url in self.rss_feeds.items():
            try:
                feed = feedparser.parse(feed_url)
                articles = []
                
                for entry in feed.entries[:3]:  # Limit to 3 per feed
                    article = {
                        'title': entry.get('title', ''),
                        'content': entry.get('summary', '') or entry.get('description', ''),
                        'url': entry.get('link', ''),
                        'source': source_name,
                        'published_date': entry.get('published', ''),
                        'image_url': self._extract_image_from_entry(entry)
                    }
                    
                    if len(article['content']) > 50:  # Filter out very short content
                        articles.append(article)
                
                all_articles.extend(articles)
                logger.info(f"RSS {source_name}: {len(articles)} articles")
                
            except Exception as e:
                logger.warning(f"Failed to fetch RSS from {source_name}: {e}")
        
        return all_articles
    
    def _extract_image_from_entry(self, entry) -> Optional[str]:
        """Extract image URL from RSS entry"""
        # Try media content
        if hasattr(entry, 'media_content'):
            for media in entry.media_content:
                if media.get('type', '').startswith('image/'):
                    return media.get('url')
        
        # Try enclosures
        if hasattr(entry, 'enclosures'):
            for enclosure in entry.enclosures:
                if enclosure.get('type', '').startswith('image/'):
                    return enclosure.get('href')
        
        # Try to extract from content/summary
        content = entry.get('summary', '') or entry.get('description', '')
        if content:
            img_match = re.search(r'<img[^>]+src=["\']([^"\']+)["\']', content)
            if img_match:
                return img_match.group(1)
        
        return None
    
    def _deduplicate_articles(self, articles: List[Dict]) -> List[Dict]:
        """Remove duplicate articles by URL and title similarity"""
        seen_urls = set()
        seen_titles = set()
        unique_articles = []
        
        for article in articles:
            url = article.get('url', '')
            title = article.get('title', '').lower().strip()
            
            # Skip if URL already seen
            if url in seen_urls:
                continue
            
            # Skip if very similar title already seen
            is_similar_title = any(
                self._similarity_ratio(title, seen_title) > 0.8 
                for seen_title in seen_titles
            )
            
            if not is_similar_title and len(article.get('content', '')) > 30:
                seen_urls.add(url)
                seen_titles.add(title)
                unique_articles.append(article)
        
        logger.info(f"Deduplicated: {len(unique_articles)} unique articles from {len(articles)}")
        return unique_articles
    
    def _similarity_ratio(self, a: str, b: str) -> float:
        """Calculate similarity ratio between two strings"""
        if not a or not b:
            return 0.0
        
        # Simple word-based similarity
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        
        if not words_a or not words_b:
            return 0.0
        
        intersection = words_a.intersection(words_b)
        union = words_a.union(words_b)
        
        return len(intersection) / len(union)
    
    def _process_articles_with_learning(self, articles: List[Dict]) -> List[Dict]:
        """Process articles and learn from the process"""
        self.state = AgentState.LEARNING
        
        # Filter articles based on learned preferences
        filtered_articles = self._filter_articles_by_learning(articles)
        
        print(f"🧠 Processing {len(filtered_articles)} articles with AI...")
        
        # Summarize in batches
        texts = [a['content'][:500] for a in filtered_articles]
        summaries = []
        
        batch_size = 5
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_summaries = self.summarizer(batch, max_length=130, min_length=30, do_sample=False)
            summaries.extend(batch_summaries)
            time.sleep(0.5)  # Small delay between batches
        
        paired = []
        for s, art in zip(summaries, filtered_articles):
            paired.append({
                'summary': s['summary_text'],
                'url': art['url'],
                'title': art['title'],
                'image_url': art.get('image_url'),
                'source': art['source'],
                'published_date': art.get('published_date'),
                'confidence_score': self._calculate_article_confidence(art)
            })
        
        # Semantic deduplication
        final_summaries = self._semantic_deduplication_with_learning(paired)
        
        # Learn patterns from successful articles
        self._learn_from_articles(final_summaries)
        
        print(f"✅ Final result: {len(final_summaries)} unique, high-quality summaries")
        
        return final_summaries[:12]  # Limit to 12 articles
    
    def _filter_articles_by_learning(self, articles: List[Dict]) -> List[Dict]:
        """Filter articles based on learned user preferences"""
        if not articles:
            return articles
        
        # Simple filtering for now - can be enhanced with ML
        filtered = []
        for article in articles:
            content = article.get('content', '').lower()
            title = article.get('title', '').lower()
            
            # Skip articles with failed patterns
            has_failed_pattern = any(
                pattern.lower() in content or pattern.lower() in title
                for pattern in self.memory.learned_patterns.get('failed_patterns', [])
            )
            
            if not has_failed_pattern:
                filtered.append(article)
        
        return filtered
    
    def _calculate_article_confidence(self, article: Dict) -> float:
        """Calculate confidence score for an article based on learned patterns"""
        score = 0.5  # Base score
        
        # Boost score for preferred sources
        if article['source'] in self.memory.learned_patterns.get('preferred_sources', []):
            score += 0.2
        
        # Boost score for successful keywords
        content_lower = article.get('content', '').lower()
        for keyword in self.memory.learned_patterns.get('successful_keywords', []):
            if keyword.lower() in content_lower:
                score += 0.1
        
        # Reduce score for previously failed patterns
        for failed_pattern in self.memory.learned_patterns.get('failed_patterns', []):
            if failed_pattern.lower() in content_lower:
                score -= 0.1
        
        return max(0.0, min(1.0, score))  # Clamp between 0 and 1
    
    def _semantic_deduplication_with_learning(self, paired: List[Dict]) -> List[Dict]:
        """Enhanced deduplication that learns optimal similarity thresholds"""
        if len(paired) <= 1:
            return paired
        
        # Adaptive threshold based on past performance
        base_threshold = 0.65
        learned_threshold = self.memory.learned_patterns.get('optimal_similarity_threshold', base_threshold)
        
        embeddings = self.embedder.encode([p['summary'] for p in paired], convert_to_tensor=True)
        final = []
        
        for i, p in enumerate(paired):
            is_unique = all(util.pytorch_cos_sim(embeddings[i], embeddings[j]) < learned_threshold 
                          for j in range(len(final)))
            if is_unique:
                final.append(p)
        
        # Learn from the deduplication process
        if len(final) < 5:  # Too few articles, threshold might be too strict
            new_threshold = min(learned_threshold + 0.05, 0.8)
            self.memory.learned_patterns['optimal_similarity_threshold'] = new_threshold
        elif len(final) > 15:  # Too many articles, threshold might be too loose
            new_threshold = max(learned_threshold - 0.05, 0.5)
            self.memory.learned_patterns['optimal_similarity_threshold'] = new_threshold
        
        return final
    
    def _learn_from_articles(self, articles: List[Dict]):
        """Learn patterns from processed articles"""
        # Extract successful sources
        sources = [art['source'] for art in articles if art.get('confidence_score', 0) > 0.7]
        if sources:
            if 'preferred_sources' not in self.memory.learned_patterns:
                self.memory.learned_patterns['preferred_sources'] = []
            self.memory.learned_patterns['preferred_sources'].extend(sources)
            # Keep only unique and recent sources
            self.memory.learned_patterns['preferred_sources'] = list(set(
                self.memory.learned_patterns['preferred_sources'][-15:]
            ))
        
        self._save_memory()
    
    def learn_from_feedback(self, feedback: ArticleFeedback):
        """Learn from user feedback to improve future recommendations"""
        self.state = AgentState.LEARNING
        
        # Store feedback in database
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO article_feedback (article_url, user_id, rating, feedback_type, timestamp)
            VALUES (?, ?, ?, ?, ?)
        ''', (feedback.article_url, feedback.user_id, feedback.rating, 
              feedback.feedback_type, feedback.timestamp.isoformat()))
        self.conn.commit()
        
        # Update learned patterns based on feedback
        if feedback.rating >= 4:  # Positive feedback
            self._extract_successful_patterns(feedback)
        elif feedback.rating <= 2:  # Negative feedback
            self._extract_failed_patterns(feedback)
        
        logger.info(f"Learned from feedback: {feedback.feedback_type} rating {feedback.rating}")
        self.state = AgentState.IDLE
    
    def _extract_successful_patterns(self, feedback: ArticleFeedback):
        """Extract patterns from positively rated articles"""
        if 'successful_keywords' not in self.memory.learned_patterns:
            self.memory.learned_patterns['successful_keywords'] = []
        
        # Simple pattern extraction - can be enhanced
        self.memory.learned_patterns['successful_keywords'].append("AI breakthrough")
        self._save_memory()
    
    def _extract_failed_patterns(self, feedback: ArticleFeedback):
        """Extract patterns from negatively rated articles"""
        if 'failed_patterns' not in self.memory.learned_patterns:
            self.memory.learned_patterns['failed_patterns'] = []
        
        # Simple pattern extraction - can be enhanced
        self.memory.learned_patterns['failed_patterns'].append("generic content")
        self._save_memory()
    
    def _send_personalized_emails(self, summaries: List[Dict]):
        """Send personalized emails based on user preferences"""
        try:
            # Generate beautiful HTML email
            html_content = self._generate_enhanced_email_html(summaries)
            
            # Send email
            msg = MIMEMultipart('alternative')
            msg['From'] = self.api_keys['gmail_user']
            msg['To'] = os.getenv('RECIPIENT_EMAIL')
            msg['Subject'] = f"🤖 AI News Daily - {datetime.now().strftime('%B %d, %Y')} ({len(summaries)} articles)"
            
            msg.attach(MIMEText(html_content, 'html'))
            
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(self.api_keys['gmail_user'], self.api_keys['gmail_password'])
            server.send_message(msg)
            server.quit()
            
            self.performance_metrics['emails_sent'] += 1
            logger.info(f"Enhanced email sent with {len(summaries)} articles")
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
    
    def _generate_enhanced_email_html(self, summaries: List[Dict]) -> str:
        """Generate beautiful enhanced HTML email"""
        greeting, timestamp = self._get_greeting()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #F5E6D3 0%, #E8D5B7 50%, #D4B896 100%);
                    color: #5D4037;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                }}
                .container {{
                    max-width: 700px;
                    margin: 0 auto;
                    background: #FFFEF7;
                    border-radius: 15px;
                    box-shadow: 0 10px 30px rgba(139, 101, 80, 0.2);
                    overflow: hidden;
                    border: 2px solid #D7CCC8;
                }}
                .header {{
                    background: linear-gradient(45deg, #8D6E63, #A1887F);
                    color: #FFFEF7;
                    padding: 30px;
                    text-align: center;
                    position: relative;
                }}
                .header h1 {{
                    margin: 0;
                    font-size: 26px;
                    font-weight: 400;
                }}
                .timestamp {{
                    font-size: 14px;
                    opacity: 0.9;
                    margin-top: 8px;
                }}
                .stats {{
                    background: #FFF8E1;
                    padding: 15px 30px;
                    border-bottom: 2px solid #D7CCC8;
                    font-size: 14px;
                    color: #6D4C41;
                }}
                .highlights {{
                    background: linear-gradient(to right, #FFF8E1, #FFFEF7);
                    padding: 25px;
                    border-bottom: 2px solid #D7CCC8;
                }}
                .highlights h2 {{
                    color: #6D4C41;
                    margin: 0 0 20px 0;
                    font-size: 20px;
                    font-weight: 500;
                }}
                .highlight-item {{
                    background: #FFFEF7;
                    padding: 18px;
                    margin-bottom: 15px;
                    border-radius: 10px;
                    border-left: 4px solid #A1887F;
                    box-shadow: 0 3px 8px rgba(139, 101, 80, 0.1);
                }}
                .article {{
                    padding: 25px;
                    border-bottom: 1px solid #EFEBE9;
                    background: #FFFEF7;
                }}
                .article:nth-child(even) {{
                    background: #FFF8E1;
                }}
                .article-meta {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 15px;
                    font-size: 12px;
                    color: #8D6E63;
                }}
                .source {{
                    background: linear-gradient(45deg, #8D6E63, #A1887F);
                    color: #FFFEF7;
                    padding: 4px 10px;
                    border-radius: 15px;
                    font-size: 11px;
                    font-weight: 500;
                }}
                .summary {{
                    font-size: 15px;
                    line-height: 1.7;
                    margin-bottom: 15px;
                    color: #5D4037;
                }}
                .read-more {{
                    background: linear-gradient(45deg, #8D6E63, #A1887F);
                    color: #FFFEF7;
                    padding: 8px 16px;
                    text-decoration: none;
                    border-radius: 20px;
                    font-size: 13px;
                    font-weight: 500;
                    transition: all 0.3s ease;
                }}
                .footer {{
                    background: #F5E6D3;
                    padding: 20px;
                    text-align: center;
                    color: #6D4C41;
                    font-size: 12px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🤖 {greeting}</h1>
                    <div class="timestamp">{timestamp}</div>
                </div>
                
                <div class="stats">
                    📊 <strong>{len(summaries)} articles</strong> from <strong>{len(set(s['source'] for s in summaries))}</strong> sources | 
                    🧠 Processed with AI | 
                    ⚡ GPU Accelerated
                </div>
                
                <div class="highlights">
                    <h2>📌 Top Headlines</h2>
        """
        
        # Add top 3 highlights
        for i, summary in enumerate(summaries[:3], 1):
            html += f"""
                    <div class="highlight-item">
                        <strong>{i}.</strong> {summary['summary'][:120]}...
                        <br><br><a href="{summary['url']}" class="read-more">Read More</a>
                    </div>
            """
        
        html += f"""
                </div>
                
                <div class="content">
        """
        
        # Add all articles
        for i, summary in enumerate(summaries, 1):
            published_date = summary.get('published_date', 'Unknown date')
            if published_date and len(published_date) > 10:
                try:
                    date_obj = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
                    published_date = date_obj.strftime('%b %d, %Y')
                except:
                    published_date = published_date[:10]
            
            html += f"""
                    <div class="article">
                        <div class="article-meta">
                            <span class="source">{summary['source']}</span>
                            <span>{published_date}</span>
                        </div>
                        <div class="summary"><strong>#{i}</strong> {summary['summary']}</div>
                        <a href="{summary['url']}" class="read-more">Read Full Article →</a>
                    </div>
            """
        
        html += """
                </div>
                
                <div class="footer">
                    🤖 Generated by Enhanced AI News Agent<br>
                    ⚙️ Powered by GPUs, Transformers & Machine Learning
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _get_greeting(self):
        """Get time-appropriate greeting"""
        hour = datetime.now().hour
        part = "morning" if hour < 12 else "afternoon" if hour < 17 else "evening"
        timestamp = datetime.now().strftime("%A, %B %d, %Y %I:%M %p")
        return f"Good {part}, here are today's AI news highlights!", timestamp
    
    def _update_performance_metrics(self, summaries: List[Dict]):
        """Update performance metrics"""
        self.performance_metrics['articles_processed'] += len(summaries)
        self.performance_metrics['total_sources'] = len(set(s['source'] for s in summaries))
        
        # Calculate success rate
        successful_sources = len(set(s['source'] for s in summaries if s.get('source')))
        total_attempted_sources = len(self.rss_feeds) + 3  # 3 API sources
        self.performance_metrics['api_success_rate'] = (successful_sources / total_attempted_sources) * 100
    
    def _check_breaking_news(self):
        """Check for breaking news and send alerts"""
        try:
            urgent_queries = ["AI breakthrough", "major tech announcement", "OpenAI", "Google AI"]
            articles = self._fetch_articles_from_all_sources(urgent_queries[:2])
            
            if len(articles) > 8:  # Threshold for breaking news
                logger.info("Potential breaking news detected!")
                
        except Exception as e:
            logger.error(f"Error checking breaking news: {e}")
    
    def _health_check(self):
        """Periodic health check of the agent"""
        logger.info(f"Agent health check - State: {self.state.value}")
        
        # Check API connectivity
        missing_keys = [name for name, key in self.api_keys.items() if not key]
        if missing_keys:
            logger.warning(f"Missing API keys: {missing_keys}")
    
    def chat_interface(self) -> str:
        """Interactive chat interface for real-time user interaction"""
        return f"""
🤖 ENHANCED AI NEWS AGENT - INTERACTIVE CHAT

📋 ACTIVE SEARCH KEYWORDS:
   {' | '.join(self.main_keywords[:4])}
   
📰 ACTIVE SOURCES: {len(self.rss_feeds)} RSS feeds + 3 APIs

💬 AVAILABLE COMMANDS:
1. 'fetch [topic]'     - Get latest news on any topic
2. 'feedback [1-5] [url]' - Rate an article to improve AI
3. 'sources'           - Show all active news sources  
4. 'keywords'          - Show current search keywords
5. 'status'            - Check agent performance stats
6. 'help'              - Show this help message
7. 'quit'              - Exit the agent

💡 EXAMPLES:
   fetch quantum computing
   feedback 5 https://example.com/article
   sources

🚀 The agent runs autonomously and learns from your feedback!
        """
    
    def process_chat_command(self, command: str, user_id: str) -> str:
        """Process chat commands from users"""
        parts = command.lower().split()
        
        if not parts:
            return "Please enter a command. Type 'help' for available commands."
        
        cmd = parts[0]
        
        if cmd == 'fetch':
            if len(parts) < 2:
                return "Please specify a topic. Example: 'fetch AI robotics'"
            topic = ' '.join(parts[1:])
            return self._handle_fetch_command(topic, user_id)
        
        elif cmd == 'feedback':
            if len(parts) < 3:
                return "Please provide rating and URL. Example: 'feedback 5 https://example.com'"
            try:
                rating = int(parts[1])
                url = parts[2]
                return self._handle_feedback_command(rating, url, user_id)
            except ValueError:
                return "Rating must be a number between 1-5"
        
        elif cmd == 'sources':
            return self._show_sources()
            
        elif cmd == 'keywords':
            return self._show_keywords()
        
        elif cmd == 'status':
            return self._get_agent_status()
        
        elif cmd == 'help':
            return self.chat_interface()
        
        else:
            return f"Unknown command: '{cmd}'. Type 'help' for available commands."
    
    def _show_sources(self) -> str:
        """Show all active news sources"""
        result = "📰 ACTIVE NEWS SOURCES:\n\n"
        
        result += "🔗 API SOURCES:\n"
        api_sources = [
            "• NewsData.io (Global News)",
            "• GNews (60k+ sources)", 
            "• Mediastack (News Archive)",
        ]
        for source in api_sources:
            result += f"  {source}\n"
        
        result += f"\n📡 RSS FEEDS ({len(self.rss_feeds)}):\n"
        for name, url in self.rss_feeds.items():
            result += f"  • {name}\n"
        
        result += f"\n📊 TOTAL: {len(api_sources) + len(self.rss_feeds)} sources"
        return result
    
    def _show_keywords(self) -> str:
        """Show current search keywords"""
        result = "🔍 ACTIVE SEARCH KEYWORDS:\n\n"
        
        result += "🎯 PRIMARY KEYWORDS:\n"
        for i, keyword in enumerate(self.main_keywords, 1):
            result += f"  {i}. {keyword.title()}\n"
        
        # Show learned patterns if available
        if self.memory.learned_patterns.get('successful_keywords'):
            result += "\n🧠 LEARNED KEYWORDS (from feedback):\n"
            for keyword in self.memory.learned_patterns['successful_keywords'][-5:]:
                result += f"  • {keyword}\n"
        
        return result
    
    def _handle_fetch_command(self, topic: str, user_id: str) -> str:
        """Handle real-time fetch requests"""
        try:
            self.state = AgentState.FETCHING
            print(f"🔍 Fetching news for: {topic}")
            
            articles = self._fetch_articles_from_all_sources([topic])
            
            if articles:
                self.state = AgentState.PROCESSING
                summaries = self._process_articles_with_learning(articles[:8])
                
                result = f"📰 Found {len(summaries)} articles on '{topic}':\n\n"
                for i, summary in enumerate(summaries[:5], 1):
                    result += f"{i}. {summary['summary'][:120]}...\n"
                    result += f"   📍 {summary['source']} | {summary['url']}\n\n"
                
                if len(summaries) > 5:
                    result += f"... and {len(summaries) - 5} more articles.\n"
                
                self.state = AgentState.IDLE
                return result
            else:
                return f"❌ Sorry, couldn't find recent articles on '{topic}'. Try a different topic."
                
        except Exception as e:
            logger.error(f"Error in fetch command: {e}")
            return "❌ Sorry, there was an error fetching articles. Please try again."
    
    def _handle_feedback_command(self, rating: int, url: str, user_id: str) -> str:
        """Handle user feedback"""
        if rating < 1 or rating > 5:
            return "❌ Rating must be between 1 and 5"
        
        feedback = ArticleFeedback(
            article_url=url,
            user_id=user_id,
            rating=rating,
            feedback_type="user_rating",
            timestamp=datetime.now()
        )
        
        self.learn_from_feedback(feedback)
        
        stars = "⭐" * rating
        return f"✅ Thank you for your feedback! Rated {stars} ({rating}/5)\n📖 The AI agent learned from your input and will improve future recommendations."
    
    def _get_agent_status(self) -> str:
        """Get current agent status and performance metrics"""
        return f"""
🤖 ENHANCED AI NEWS AGENT STATUS

📊 PERFORMANCE METRICS:
   Current State: {self.state.value.title()}
   Emails Sent: {self.performance_metrics['emails_sent']}
   Articles Processed: {self.performance_metrics['articles_processed']}
   Success Rate: {self.performance_metrics['api_success_rate']:.1f}%
   Active Sources: {self.performance_metrics.get('total_sources', 0)}

🧠 MEMORY & LEARNING:
   Successful Queries: {len(self.memory.successful_queries)}
   Failed Sources: {len(self.memory.failed_sources)}
   Learned Patterns: {len(self.memory.learned_patterns)}
   User Interactions: {len(self.memory.user_interactions)}

🔧 SYSTEM INFO:
   AI Device: {'GPU (CUDA)' if torch.cuda.is_available() else 'CPU'}
   RSS Feeds: {len(self.rss_feeds)}
   API Keys: {sum(1 for key in self.api_keys.values() if key)}
   
📅 Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

🚀 Status: Agent is running autonomously and learning!
        """
    
    # API fetch methods (same as before but with better error handling)
    def _fetch_newsdata(self, query, n=6):
        try:
            resp = requests.get(
                "https://newsdata.io/api/1/news",
                params={'apikey': self.api_keys['newsdata'], 'q': query, 'language': 'en', 'size': n},
                timeout=10
            )
            resp.raise_for_status()
            articles = resp.json().get('results', [])
            return [{
                'title': a.get('title', ''),
                'content': a.get('content', '') or a.get('description', ''),
                'url': a.get('link', ''),
                'image_url': a.get('image_url'),
                'source': a.get('source_id', 'NewsData'),
                'published_date': a.get('pubDate', '')
            } for a in articles[:n]]
        except Exception as e:
            logger.warning(f"NewsData API failed: {e}")
            return []
    
    def _fetch_gnews(self, query, n=6):
        try:
            resp = requests.get(
                "https://gnews.io/api/v4/search",
                params={'q': query, 'lang': 'en', 'max': n, 'apikey': self.api_keys['gnews']},
                timeout=10
            )
            resp.raise_for_status()
            articles = resp.json().get('articles', [])
            return [{
                'title': a.get('title', ''),
                'content': a.get('description', ''),
                'url': a.get('url', ''),
                'image_url': a.get('image'),
                'source': a.get('source', {}).get('name', 'GNews'),
                'published_date': a.get('publishedAt', '')
            } for a in articles[:n]]
        except Exception as e:
            logger.warning(f"GNews API failed: {e}")
            return []
    
    def _fetch_mediastack(self, query, n=6):
        try:
            resp = requests.get(
                "http://api.mediastack.com/v1/news",
                params={'access_key': self.api_keys['mediastack'], 'keywords': query, 'limit': n},
                timeout=10
            )
            resp.raise_for_status()
            articles = resp.json().get('data', [])
            return [{
                'title': a.get('title', ''),
                'content': a.get('description', ''),
                'url': a.get('url', ''),
                'image_url': a.get('image'),
                'source': a.get('source', 'Mediastack'),
                'published_date': a.get('published_at', '')
            } for a in articles[:n]]
        except Exception as e:
            logger.warning(f"Mediastack API failed: {e}")
            return []
    
    def run_agent(self):
        """Main agent loop - keeps the agent running"""
        logger.info("🚀 Enhanced AI News Agent started and running autonomously")
        print(self.chat_interface())
        
        while True:
            try:
                user_input = input("\n> ").strip()
                if user_input.lower() in ['quit', 'exit']:
                    break
                
                if user_input:
                    response = self.process_chat_command(user_input, "default_user")
                    print(f"\n{response}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                print("❌ Sorry, there was an error. Please try again.")
        
        print("\n👋 AI News Agent shutting down. Thanks for using the enhanced version!")
        logger.info("Enhanced AI News Agent shutting down")

def main():
    """Initialize and run the Enhanced AI News Agent"""
    try:
        agent = EnhancedNewsAIAgent()
        agent.run_agent()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Agent shutting down...")
    except Exception as e:
        print(f"\n❌ Error starting agent: {e}")
        logger.error(f"Fatal error: {e}")

if __name__ == "__main__":
    main()