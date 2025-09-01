# 🤖 AI News Agent

An autonomous AI-powered news aggregation system that fetches, summarizes, and delivers personalized AI news from multiple sources using advanced NLP models.

## 🚀 Features

- **Autonomous Operation**: Runs daily news cycles automatically with intelligent scheduling
- **Multi-Source Aggregation**: Fetches news from 11+ sources (APIs + RSS feeds)
- **AI-Powered Processing**: Uses transformer models for summarization and semantic deduplication
- **Learning Capabilities**: Adapts based on user feedback and performance patterns
- **Interactive Chat Interface**: Real-time commands for fetching news and providing feedback
- **Beautiful Email Reports**: Generates HTML emails with embedded images and PDF attachments
- **GPU Acceleration**: Leverages CUDA for faster AI processing when available

## 📰 News Sources

### API Sources
- NewsData.io - Global news aggregation
- GNews - 60,000+ news sources
- Mediastack - News archive and search
- WorldNews API - International news coverage

### RSS Feeds
- MIT AI News
- VentureBeat AI
- Wired AI Coverage
- TechCrunch AI
- IEEE Spectrum AI
- MarkTechPost
- OpenAI Blog
- AI News Aggregator

## 🔧 Technologies Used

- **Python 3.8+**
- **PyTorch** - Deep learning framework
- **Transformers** - Hugging Face NLP models
- **Sentence Transformers** - Semantic similarity analysis
- **SQLite** - Persistent storage for learning and preferences
- **Schedule** - Task automation and scheduling
- **SMTP** - Email delivery system
- **Feedparser** - RSS feed processing
- **ReportLab** - PDF generation

## 📋 Requirements

- Python 3.8 or higher
- GPU with CUDA support (optional, for acceleration)
- Email account with app password for Gmail SMTP
- API keys for news sources

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/ai-news-agent.git
cd ai-news-agent
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
Copy `.env.example` to `.env` and fill in your API keys:
```bash
cp .env.example .env
```

Edit `.env` with your actual keys:
```
NEWSDATA_API_KEY=your_newsdata_key
GNEWS_API_KEY=your_gnews_key
MEDIASTACK_API_KEY=your_mediastack_key
WORLDNEWS_API_KEY=your_worldnews_key
GMAIL_USER=your_email@gmail.com
GMAIL_APP_PASSWORD=your_app_password
RECIPIENT_EMAIL=recipient@example.com
```

### 4. Run the Agent
```bash
python newsagent.py
```

## 💬 Interactive Commands

Once running, you can use these commands in the chat interface:

- `fetch [topic]` - Get latest news on any topic
- `feedback [1-5] [url]` - Rate articles to improve AI learning
- `sources` - View all active news sources
- `keywords` - Show current search keywords
- `status` - Check agent performance and statistics
- `help` - Display available commands
- `quit` - Exit the agent

## 📊 Example Usage

```
> fetch quantum computing
📰 Found 8 articles on 'quantum computing':

1. Google announces breakthrough in quantum error correction...
   📍 MIT AI News | https://news.mit.edu/...

2. IBM reveals new quantum processor with 1000+ qubits...
   📍 TechCrunch AI | https://techcrunch.com/...

> feedback 5 https://news.mit.edu/quantum-article
✅ Thank you for your feedback! Rated ⭐⭐⭐⭐⭐ (5/5)
📖 The AI agent learned from your input and will improve future recommendations.

> status
🤖 ENHANCED AI NEWS AGENT STATUS
📊 PERFORMANCE METRICS:
   Current State: Idle
   Emails Sent: 15
   Articles Processed: 147
   Success Rate: 85.2%
```

## 🤖 How It Works

1. **Autonomous Scheduling**: Agent runs daily at 8 AM, fetching news automatically
2. **Multi-Source Fetching**: Queries 11+ news sources with intelligent fallback
3. **AI Processing**: Summarizes articles using DistilBART and deduplicates with sentence embeddings
4. **Learning System**: Adapts queries and preferences based on user feedback
5. **Email Delivery**: Sends beautiful HTML emails with summaries and images
6. **Interactive Chat**: Provides real-time interaction for custom requests

## 📈 Performance Features

- **GPU Acceleration**: Automatic CUDA detection for faster processing
- **Batch Processing**: Efficient handling of multiple articles
- **Intelligent Deduplication**: Semantic similarity analysis prevents duplicate content
- **Adaptive Thresholds**: Learning optimal similarity thresholds over time
- **Error Recovery**: Robust handling of API failures with source switching

## 🎯 Key Search Keywords

- Artificial Intelligence
- Machine Learning
- AI Industry News
- Deep Learning
- Neural Networks
- AI Breakthrough
- Tech Innovation
- AI Research

## 📝 Configuration

The agent stores its learning and preferences in:
- `agent_memory.json` - Learned patterns and successful queries
- `news_agent.db` - SQLite database for user feedback and performance metrics

## 🔐 Security Notes

- Never commit your `.env` file with real API keys
- Use app passwords for Gmail SMTP authentication
- Keep your API keys secure and rotate them regularly

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Hugging Face for transformer models
- OpenAI for inspiration in AI agent design
- News API providers for data access
- Python community for excellent libraries

## 📞 Support

For questions or issues, please open a GitHub issue or contact [holagundiakash86@gmail.com].

---

**Built with ❤️ using Python, PyTorch, and cutting-edge NLP technologies**
