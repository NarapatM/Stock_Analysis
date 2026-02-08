import streamlit as st
import feedparser
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
import re
import numpy as np
from deep_translator import GoogleTranslator

st.set_page_config(
    page_title="AI Stock News Analyzer Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
        font-size: 1.1rem;
    }
    
    .news-card {
        background: linear-gradient(145deg, #2d2d3a, #252530);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border-left: 5px solid;
        box-shadow: 0 8px 20px rgba(0,0,0,0.2);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .news-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 30px rgba(0,0,0,0.3);
    }
    
    .card-good { border-left-color: #10b981; }
    .card-normal { border-left-color: #3b82f6; }
    .card-risk { border-left-color: #f59e0b; }
    .card-high-risk { border-left-color: #ef4444; }
    
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1.2rem;
        border-radius: 25px;
        font-weight: 700;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
        margin-bottom: 1rem;
        text-transform: uppercase;
    }
    
    .badge-good { background: linear-gradient(135deg, #10b981, #059669); color: white; }
    .badge-normal { background: linear-gradient(135deg, #3b82f6, #2563eb); color: white; }
    .badge-risk { background: linear-gradient(135deg, #f59e0b, #d97706); color: white; }
    .badge-high-risk { background: linear-gradient(135deg, #ef4444, #dc2626); color: white; }
    
    .narrative-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .confidence-indicator {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.75rem;
        font-weight: 600;
        background: linear-gradient(135deg, #8b5cf6, #6d28d9);
        color: white;
    }
    
    .news-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #f0f0f0;
        margin-bottom: 1rem;
        line-height: 1.5;
    }
    
    .news-points {
        background: rgba(255,255,255,0.05);
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .news-points ul {
        margin: 0;
        padding-left: 1.5rem;
    }
    
    .news-points li {
        margin-bottom: 0.5rem;
        color: #d1d5db;
        line-height: 1.6;
    }
    
    .source-link {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        color: #8b5cf6;
        text-decoration: none;
        font-weight: 500;
        transition: color 0.2s;
    }
    
    .source-link:hover { color: #a78bfa; }
    
    .stat-card {
        background: linear-gradient(145deg, #2d2d3a, #252530);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .stat-number {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .stat-label {
        color: #9ca3af;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 1px solid rgba(255,255,255,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- ADVANCED NLP ANALYSIS ---
def classify_narrative(headline):
    """Classify news narrative type"""
    headline_lower = headline.lower()
    
    narratives = []
    
    # Growth Story
    if any(word in headline_lower for word in ['growth', 'expansion', 'revenue', 'earnings beat', 'partnership', 'contract', 'deal']):
        narratives.append(('📈 Growth', '#10b981'))
    
    # Turnaround / Survival
    if any(word in headline_lower for word in ['turnaround', 'recovery', 'restructure', 'survival', 'comeback']):
        narratives.append(('🔄 Turnaround', '#3b82f6'))
    
    # Legal / Regulatory
    if any(word in headline_lower for word in ['lawsuit', 'investigation', 'sec', 'regulatory', 'compliance', 'fraud']):
        narratives.append(('⚖️ Legal', '#ef4444'))
    
    # Speculative / Hype
    if any(word in headline_lower for word in ['meme', 'short squeeze', 'retail', 'viral', 'reddit', 'hype']):
        narratives.append(('🚀 Speculative', '#f59e0b'))
    
    # Macro / Fed / Rates
    if any(word in headline_lower for word in ['fed', 'rate', 'inflation', 'macro', 'economy', 'recession']):
        narratives.append(('🌍 Macro', '#6366f1'))
    
    # Crisis / Distress
    if any(word in headline_lower for word in ['crisis', 'bankruptcy', 'collapse', 'emergency', 'disaster']):
        narratives.append(('🚨 Crisis', '#dc2626'))
    
    # M&A / Corporate Action
    if any(word in headline_lower for word in ['merger', 'acquisition', 'buyout', 'takeover', 'spinoff']):
        narratives.append(('🤝 M&A', '#8b5cf6'))
    
    # Analyst Action
    if any(word in headline_lower for word in ['upgrade', 'downgrade', 'target', 'analyst', 'rating']):
        narratives.append(('📊 Analyst', '#06b6d4'))
    
    if not narratives:
        narratives.append(('📰 General', '#6b7280'))
    
    return narratives

def calculate_confidence(headline, matched_signals):
    """Calculate confidence score of analysis"""
    confidence = 50  # Base
    
    # More signals = higher confidence
    signal_count = len(matched_signals)
    if signal_count >= 5:
        confidence += 30
    elif signal_count >= 3:
        confidence += 20
    elif signal_count >= 1:
        confidence += 10
    
    # Hedging words reduce confidence
    hedging_words = ['may', 'might', 'could', 'possibly', 'potentially', 'reportedly', 'allegedly']
    hedging_count = sum(1 for word in hedging_words if word in headline.lower())
    confidence -= (hedging_count * 8)
    
    # Clear strong signals increase confidence
    strong_signals = ['bankruptcy', 'approved', 'beats', 'record', 'criminal']
    if any(sig in headline.lower() for sig in strong_signals):
        confidence += 15
    
    # Question marks reduce confidence
    if '?' in headline:
        confidence -= 15
    
    return max(10, min(95, confidence))

def analyze_sentiment(headline, ticker):
    """Advanced NLP-based sentiment analysis with confidence"""
    
    headline_lower = headline.lower()
    
    # Multi-tier keywords
    exceptional_positive = {
        'record earnings': 20, 'beats earnings': 18, 'crushes expectations': 20,
        'massive surge': 19, 'breakthrough deal': 20, 'blockbuster': 18,
        'skyrockets': 19, 'doubles revenue': 20
    }
    
    exceptional_negative = {
        'bankruptcy': -20, 'fraud investigation': -20, 'criminal charges': -20,
        'massive layoffs': -18, 'ceo arrested': -20, 'sec investigation': -19,
        'class action lawsuit': -18, 'accounting fraud': -20
    }
    
    strong_positive = {
        'beats': 15, 'exceeds': 14, 'surge': 13, 'soar': 13, 'rally': 12,
        'acquisition': 14, 'merger approved': 15, 'fda approval': 15,
        'upgraded': 14, 'outperforms': 13, 'record': 14
    }
    
    strong_negative = {
        'plunges': -15, 'crashes': -14, 'bankruptcy warning': -15,
        'investigation': -13, 'lawsuit filed': -14, 'halted trading': -15,
        'scandal': -14, 'massive loss': -15
    }
    
    moderate_positive = {
        'partnership': 10, 'growth': 9, 'expansion': 9, 'innovation': 8,
        'strong': 8, 'win': 9, 'success': 8, 'positive': 8,
        'contract': 9, 'revenue increase': 10, 'profit': 9
    }
    
    moderate_negative = {
        'downgrade': -12, 'cuts outlook': -10, 'disappointing': -9,
        'concern': -8, 'warning': -9, 'decline': -8, 'loss': -9,
        'slump': -10, 'struggle': -8, 'challenge': -7
    }
    
    mild_positive = {
        'improve': 6, 'optimistic': 6, 'hopeful': 5, 'recovery': 6,
        'gain': 5, 'rise': 5, 'increase': 5
    }
    
    mild_negative = {
        'miss': -9, 'below expectations': -10, "won't save": -12,
        'worry': -6, 'caution': -6, 'delay': -6, 'fall': -7,
        'drop': -7, 'weaken': -6, 'pressure': -5
    }
    
    score = 50
    matched_signals = []
    
    all_keywords = [
        (exceptional_positive, "++"), (exceptional_negative, "--"),
        (strong_positive, "+"), (strong_negative, "-"),
        (moderate_positive, "+"), (moderate_negative, "-"),
        (mild_positive, "+"), (mild_negative, "-")
    ]
    
    for keyword_dict, symbol in all_keywords:
        for phrase, weight in keyword_dict.items():
            if phrase in headline_lower:
                score += weight
                matched_signals.append(f"{symbol}{phrase}")
    
    # Contextual adjustments
    if "even" in headline_lower and ("beat" in headline_lower or "good" in headline_lower):
        if any(word in headline_lower for word in ["won't", "can't", "not enough"]):
            score -= 15
            matched_signals.append("-contrarian")
    
    score = max(0, min(100, score))
    
    if score >= 75:
        status = "GOOD"
    elif score >= 50:
        status = "NORMAL"
    elif score >= 25:
        status = "RISK"
    else:
        status = "HIGH RISK"
    
    # Calculate confidence
    confidence = calculate_confidence(headline, matched_signals)
    
    # Classify narrative
    narratives = classify_narrative(headline)
    
    # Generate insights
    points = generate_insights(headline, status, score, matched_signals, confidence)
    
    return status, points, score, confidence, narratives, matched_signals

def generate_insights(headline, status, score, signals, confidence):
    """Generate intelligent insights"""
    
    headline_lower = headline.lower()
    
    # Point 1: What happened
    if status == "GOOD":
        if 'earnings' in headline_lower or 'beat' in headline_lower:
            point1 = "Strong positive earnings signal - company exceeded market expectations"
        elif any(word in headline_lower for word in ['partnership', 'deal', 'acquisition']):
            point1 = "Strategic business development indicates growth trajectory"
        else:
            point1 = "Multiple positive indicators detected suggesting bullish sentiment"
    elif status == "RISK":
        if 'downgrade' in headline_lower:
            point1 = "Analyst downgrade reflects concern about near-term fundamentals"
        elif "won't save" in headline_lower:
            point1 = "Bearish outlook persists despite positive factors - structural issues"
        else:
            point1 = "Warning signals detected that may pressure stock performance"
    elif status == "HIGH RISK":
        if any(word in headline_lower for word in ['lawsuit', 'investigation', 'fraud']):
            point1 = "Critical legal/regulatory risk with material impact potential"
        else:
            point1 = "Severe negative signals indicate high-risk situation"
    else:
        point1 = "Mixed signals - headline lacks strong directional momentum"
    
    # Point 2: Market impact with confidence
    impact_strength = "High" if confidence >= 70 else "Moderate" if confidence >= 50 else "Low"
    
    if score >= 75:
        point2 = f"{impact_strength} confidence of positive market reaction (+{score-50}% potential upside)"
    elif score >= 50:
        point2 = f"{impact_strength} confidence of minimal market impact (±{abs(score-50)}% range expected)"
    else:
        point2 = f"{impact_strength} confidence of negative reaction (-{50-score}% downside risk)"
    
    # Point 3: Action with confidence qualifier
    if confidence >= 70:
        conf_text = "High confidence"
    elif confidence >= 50:
        conf_text = "Moderate confidence"
    else:
        conf_text = "Low confidence - await confirmation"
    
    if score >= 80:
        point3 = f"{conf_text}: Strong buy signal - consider increasing position"
    elif score >= 65:
        point3 = f"{conf_text}: Hold with positive bias - maintain or add modestly"
    elif score >= 50:
        point3 = f"{conf_text}: Hold - monitor for clearer signals"
    elif score >= 35:
        point3 = f"{conf_text}: Reduce exposure - set protective stop-losses"
    else:
        point3 = f"{conf_text}: Exit or avoid - significant downside risk"
    
    return [f"- {point1}", f"- {point2}", f"- {point3}"]

# --- VISUALIZATION FUNCTIONS ---
def create_gauge_chart(sentiment_score, status_text):
    """Create sentiment gauge"""
    if sentiment_score >= 75:
        color = "#10b981"
    elif sentiment_score >= 50:
        color = "#3b82f6"
    elif sentiment_score >= 25:
        color = "#f59e0b"
    else:
        color = "#ef4444"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = sentiment_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {
            'text': f"<b>{status_text}</b><br><span style='font-size:14px'>Sentiment Score</span>", 
            'font': {'size': 24, 'color': 'white'}
        },
        delta = {
            'reference': 50, 
            'increasing': {'color': "#10b981"}, 
            'decreasing': {'color': "#ef4444"}
        },
        number = {'suffix': "/100", 'font': {'size': 40}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "white"},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "rgba(255,255,255,0.1)",
            'borderwidth': 2,
            'bordercolor': "white",
            'steps': [
                {'range': [0, 25], 'color': 'rgba(239, 68, 68, 0.3)'},
                {'range': [25, 50], 'color': 'rgba(245, 158, 11, 0.3)'},
                {'range': [50, 75], 'color': 'rgba(59, 130, 246, 0.3)'},
                {'range': [75, 100], 'color': 'rgba(16, 185, 129, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': sentiment_score
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor = "rgba(0,0,0,0)",
        font = {'color': "white", 'family': "Arial"},
        height = 300,
        margin = dict(l=20, r=20, t=60, b=20)
    )
    
    return fig

def create_stock_chart_with_sentiment(ticker, results):
    """Create stock price chart with sentiment overlay"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1mo")
        
        if hist.empty:
            return None
        
        # Create figure with secondary y-axis
        fig = go.Figure()
        
        # Add candlestick
        fig.add_trace(go.Candlestick(
            x=hist.index,
            open=hist['Open'],
            high=hist['High'],
            low=hist['Low'],
            close=hist['Close'],
            name='Price',
            increasing_line_color='#10b981',
            decreasing_line_color='#ef4444'
        ))
        
        # Add volume
        fig.add_trace(go.Bar(
            x=hist.index,
            y=hist['Volume'],
            name='Volume',
            yaxis='y2',
            marker_color='rgba(59, 130, 246, 0.3)'
        ))
        
        # Add sentiment markers if available
        if results:
            avg_sentiment = np.mean([r['score'] for r in results])
            latest_date = hist.index[-1]
            
            # Add sentiment indicator
            fig.add_trace(go.Scatter(
                x=[latest_date],
                y=[hist['Close'].iloc[-1]],
                mode='markers+text',
                name='Sentiment',
                marker=dict(
                    size=20,
                    color='#8b5cf6',
                    symbol='diamond',
                    line=dict(color='white', width=2)
                ),
                text=[f'Sentiment: {avg_sentiment:.0f}'],
                textposition='top center',
                textfont=dict(size=12, color='white')
            ))
        
        fig.update_layout(
            title=dict(
                text=f'<b>{ticker} Price Chart (30 Days)</b>',
                font=dict(size=20, color='white')
            ),
            yaxis=dict(title='Price ($)', side='left', color='white'),
            yaxis2=dict(title='Volume', side='right', overlaying='y', color='rgba(59, 130, 246, 0.7)'),
            xaxis=dict(
                title='Date',
                rangeslider=dict(visible=False),
                color='white'
            ),
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400,
            margin=dict(l=50, r=50, t=50, b=50),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(0,0,0,0.5)'
            )
        )
        
        return fig
    except Exception as e:
        return None

def create_wordcloud(headlines):
    """Create word cloud from headlines"""
    try:
        # Combine all headlines
        text = ' '.join(headlines)
        
        # Remove common stop words
        stop_words = set(['stock', 'stocks', 'share', 'shares', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'be'])
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='#1e1e2e',
            colormap='viridis',
            stopwords=stop_words,
            max_words=50,
            relative_scaling=0.5,
            min_font_size=10
        ).generate(text)
        
        # Create matplotlib figure
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        fig.patch.set_facecolor('#1e1e2e')
        plt.tight_layout(pad=0)
        
        return fig
    except:
        return None

def get_status_config(status):
    """Get UI config"""
    configs = {
        "GOOD": {"badge_class": "badge-good", "card_class": "card-good", "icon": "✅", "text": "Good News"},
        "NORMAL": {"badge_class": "badge-normal", "card_class": "card-normal", "icon": "ℹ️", "text": "Normal"},
        "RISK": {"badge_class": "badge-risk", "card_class": "card-risk", "icon": "⚠️", "text": "Risk"},
        "HIGH RISK": {"badge_class": "badge-high-risk", "card_class": "card-high-risk", "icon": "🚨", "text": "High Risk"}
    }
    return configs.get(status, configs["NORMAL"])

def translate_to_thai(text, max_retries=2):
    """Translate text to Thai with retry mechanism using deep-translator"""
    if not text or len(text.strip()) == 0:
        return text
    
    for attempt in range(max_retries):
        try:
            # ใช้ deep_translator แทน googletrans
            return GoogleTranslator(source='auto', target='th').translate(text)
        except Exception as e:
            if attempt < max_retries - 1:
                continue
            else:
                return text  # Return original if translation fails
    return text

def translate_results(results, translate_titles=True, translate_points=True):
    """Translate analysis results to Thai"""
    translated_results = []
    
    for result in results:
        translated_result = result.copy()
        
        # Translate title
        if translate_titles:
            translated_result['title_th'] = translate_to_thai(result['title'])
        
        # Translate analysis points
        if translate_points:
            translated_points = []
            for point in result['points']:
                # Keep the "- " prefix and translate the rest
                text_to_translate = point[2:] if point.startswith('- ') else point
                translated_text = translate_to_thai(text_to_translate)
                translated_points.append(f"- {translated_text}")
            translated_result['points_th'] = translated_points
        
        translated_results.append(translated_result)
    
    return translated_results

# --- MAIN APP ---
st.markdown("""
<div class="main-header">
    <h1>📊 AI Stock News Analyzer Pro</h1>
    <p>Professional Market Intelligence • Advanced NLP • Real-time Data</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
    st.title("ℹ️ Features")
    
    st.success("✅ Professional Analytics Suite")
    
    # Language Toggle
    st.markdown("---")
    st.subheader("🌐 Language")
    use_thai = st.toggle("แสดงภาษาไทย (Show Thai)", value=False, help="Toggle to translate news and analysis to Thai")
    
    st.markdown("---")
    
    st.markdown("""
    **AI Analysis:**
    - Multi-tier NLP engine
    - Narrative classification
    - Confidence scoring
    - Context detection
    
    **Market Data:**
    - Real-time price charts
    - Volume analysis
    - Sentiment correlation
    - Technical indicators
    
    **Visualizations:**
    - Word cloud themes
    - Interactive dashboards
    - Trend analysis
    - Export reports
    """)
    
    st.markdown("---")
    st.caption("💡 Try: NVDA, TSLA, AAPL, RKLB")

# Main Input
col1, col2 = st.columns([3, 1])
with col1:
    ticker = st.text_input(
        "🎯 Enter Stock Symbol",
        value="RKLB",
        placeholder="e.g., AAPL, TSLA, NVDA"
    ).upper()

with col2:
    st.write("")
    st.write("")
    scan_button = st.button("🔍 Analyze", type="primary", use_container_width=True)

# Analysis
if scan_button:
    with st.spinner(f"🔍 Analyzing {ticker}..."):
        # Fetch news
        rss_url = f"https://news.google.com/rss/search?q={ticker}+Stock+when:7d&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(rss_url)
        
        if not feed.entries:
            st.warning(f"📭 No recent news found for {ticker}")
        else:
            news_items = feed.entries[:5]
            
            st.success("✨ Advanced Analysis Complete!")
            
            # Analyze news
            results = []
            headlines = []
            progress_bar = st.progress(0)
            
            for idx, item in enumerate(news_items):
                status, points, score, confidence, narratives, signals = analyze_sentiment(item.title, ticker)
                results.append({
                    'title': item.title,
                    'link': item.link,
                    'status': status,
                    'points': points,
                    'score': score,
                    'confidence': confidence,
                    'narratives': narratives,
                    'signals': signals,
                    'published': item.get('published', 'N/A')
                })
                headlines.append(item.title)
                progress_bar.progress((idx + 1) / len(news_items))
            
            progress_bar.empty()
            
            # Translate results if Thai is enabled
            if use_thai:
                with st.spinner("🌐 กำลังแปลเป็นภาษาไทย..."):
                    results = translate_results(results, translate_titles=True, translate_points=True)
            
            # Calculate metrics
            avg_score = np.mean([r['score'] for r in results])
            avg_confidence = np.mean([r['confidence'] for r in results])
            
            if avg_score >= 75:
                overall_status = "GOOD"
            elif avg_score >= 50:
                overall_status = "NORMAL"
            elif avg_score >= 25:
                overall_status = "RISK"
            else:
                overall_status = "HIGH RISK"
            
            # === DASHBOARD LAYOUT ===
            st.markdown(f"## 📊 {'แดชบอร์ดข้อมูลตลาด' if use_thai else 'Market Intelligence Dashboard'}")
            
            # Row 1: Gauge + Stock Chart
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"### {'ภาพรวมความรู้สึก' if use_thai else 'Sentiment Overview'}")
                gauge_fig = create_gauge_chart(avg_score, overall_status)
                st.plotly_chart(gauge_fig, use_container_width=True)
                
                st.metric(
                    label="ความมั่นใจเฉลี่ย" if use_thai else "Average Confidence",
                    value=f"{avg_confidence:.0f}%",
                    delta=f"{avg_confidence-70:.0f}% vs {'มาตรฐาน' if use_thai else 'baseline'}"
                )
            
            with col2:
                st.markdown(f"### {'ราคาและปริมาณ' if use_thai else 'Price & Volume'}")
                stock_chart = create_stock_chart_with_sentiment(ticker, results)
                if stock_chart:
                    st.plotly_chart(stock_chart, use_container_width=True)
                else:
                    st.info("📊 " + ("ไม่มีข้อมูลราคาสำหรับสัญลักษณ์นี้" if use_thai else "Price data unavailable for this symbol"))
            
            # Row 2: Stats
            st.markdown(f"### 📈 {'สถิติสรุป' if use_thai else 'Summary Statistics'}")
            stat_cols = st.columns(4)
            
            status_counts = {}
            for r in results:
                status_counts[r['status']] = status_counts.get(r['status'], 0) + 1
            
            with stat_cols[0]:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number" style="color: #10b981;">{status_counts.get('GOOD', 0)}</div>
                    <div class="stat-label">Good News</div>
                </div>
                """, unsafe_allow_html=True)
            
            with stat_cols[1]:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number" style="color: #3b82f6;">{status_counts.get('NORMAL', 0)}</div>
                    <div class="stat-label">Normal</div>
                </div>
                """, unsafe_allow_html=True)
            
            with stat_cols[2]:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number" style="color: #f59e0b;">{status_counts.get('RISK', 0)}</div>
                    <div class="stat-label">Risk</div>
                </div>
                """, unsafe_allow_html=True)
            
            with stat_cols[3]:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number" style="color: #ef4444;">{status_counts.get('HIGH RISK', 0)}</div>
                    <div class="stat-label">High Risk</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Row 3: Word Cloud
            st.markdown(f"### 🔤 {'การวิเคราะห์หัวข้อข่าว (Word Cloud)' if use_thai else 'News Theme Analysis (Word Cloud)'}")
            wordcloud_fig = create_wordcloud(headlines)
            if wordcloud_fig:
                st.pyplot(wordcloud_fig)
            
            st.markdown("<hr>", unsafe_allow_html=True)
            
            # Row 4: Detailed Analysis
            st.markdown(f"### 📰 {'การวิเคราะห์เชิงลึกด้วย AI' if use_thai else 'Detailed AI Analysis'}")
            
            for idx, result in enumerate(results, 1):
                config = get_status_config(result['status'])
                
                # Narrative badges
                narrative_html = ''.join([
                    f'<span class="narrative-badge" style="background-color: {color};">{name}</span>'
                    for name, color in result['narratives']
                ])
                
                # Confidence indicator
                conf_color = "#10b981" if result['confidence'] >= 70 else "#f59e0b" if result['confidence'] >= 50 else "#ef4444"
                
                # Use Thai or English based on toggle
                display_title = result.get('title_th', result['title']) if use_thai else result['title']
                display_points = result.get('points_th', result['points']) if use_thai else result['points']
                
                st.markdown(f"""
                <div class="news-card {config['card_class']}">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
                        <span class="status-badge {config['badge_class']}">
                            {config['icon']} {config['text']} • Score: {result['score']}/100
                        </span>
                        <span class="confidence-indicator" style="background: {conf_color};">
                            🎯 {'ความมั่นใจ' if use_thai else 'Confidence'}: {result['confidence']}%
                        </span>
                    </div>
                    <div style="margin: 0.5rem 0;">
                        {narrative_html}
                    </div>
                    <div class="news-title">
                        {idx}. {display_title}
                    </div>
                    <div class="news-points">
                        <strong>🤖 {'การวิเคราะห์เชิงลึก' if use_thai else 'Professional Analysis'}:</strong>
                        <ul>
                            {''.join(f'<li>{point[2:]}</li>' for point in display_points)}
                        </ul>
                    </div>
                    <div style="font-size: 0.85rem; color: #9ca3af; margin-top: 0.5rem;">
                        <strong>{'สัญญาณที่ตรวจพบ' if use_thai else 'Detected signals'}:</strong> {', '.join(result['signals'][:5]) if result['signals'] else 'baseline analysis'}
                    </div>
                    <a href="{result['link']}" target="_blank" class="source-link">
                        🔗 {'อ่านข่าวฉบับเต็ม' if use_thai else 'Read full article'} →
                    </a>
                </div>
                """, unsafe_allow_html=True)
            
            # Export
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown(f"### 💾 {'ส่งออกรายงาน' if use_thai else 'Export Report'}")
            
            df = pd.DataFrame([{
                'Ticker': ticker,
                'News Title': r.get('title_th', r['title']) if use_thai else r['title'],
                'Status': r['status'],
                'Sentiment Score': r['score'],
                'Confidence': r['confidence'],
                'Narratives': ', '.join([n[0] for n in r['narratives']]),
                'Analysis': ' | '.join(r.get('points_th', r['points']) if use_thai else r['points']),
                'Signals': ', '.join(r['signals']),
                'Source': r['link'],
                'Date': r['published']
            } for r in results])
            
            csv = df.to_csv(index=False)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button(
                    label="📥 " + ("ดาวน์โหลดรายงาน CSV" if use_thai else "Download CSV Report"),
                    data=csv,
                    file_name=f"{ticker}_pro_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            success_msg = f"✅ {'การวิเคราะห์เสร็จสมบูรณ์' if use_thai else 'Professional Analysis Complete'}! • {len(results)} {'ข่าว' if use_thai else 'articles'} • {'ความมั่นใจเฉลี่ย' if use_thai else 'Avg Confidence'}: {avg_confidence:.0f}%"
            st.success(success_msg)

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #6b7280; padding: 2rem 0;'>
    <p>📊 <strong>AI Stock News Analyzer Pro</strong> • Advanced NLP • Market Intelligence</p>
    <p style='font-size: 0.85rem;'>✅ Real-time Data • Word Cloud • Narrative Analysis • Confidence Scoring</p>
    <p style='font-size: 0.8rem; margin-top: 1rem;'>⚠️ For informational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)