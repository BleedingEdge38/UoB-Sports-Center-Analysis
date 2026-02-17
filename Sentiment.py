import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Text processing libraries
import re
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from collections import Counter, defaultdict
import string

# Advanced NLP libraries
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from wordcloud import WordCloud

# Download required NLTK data
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Date processing
from datetime import datetime, timedelta
import calendar

# Statistical analysis
from scipy import stats
from scipy.stats import pearsonr

# Set style for visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("✅ All libraries imported successfully!")

# Load the qualitative feedback data
def load_feedback_data():
    """Load and preprocess the qualitative feedback data"""
    try:
        # Try to load preprocessed pickle file first
        feedback_df = pd.read_pickle('feedback_processed.pkl')
        print("✅ Loaded preprocessed feedback data")
    except:
        # Load from Excel if pickle doesn't exist
        feedback_df = pd.read_excel('F:/UoB Study/Capstone Project/Final Project/Datasets/Qualitative_Feedback_25_G24.xlsx')
        print("✅ Loaded raw feedback data from Excel")
        
        # Basic preprocessing
        feedback_df['Date Received2'] = pd.to_datetime(feedback_df['Date Received2'])
        feedback_df['year'] = feedback_df['Date Received2'].dt.year
        feedback_df['month'] = feedback_df['Date Received2'].dt.month
        feedback_df['quarter'] = feedback_df['Date Received2'].dt.quarter
        feedback_df['weekday'] = feedback_df['Date Received2'].dt.day_name()
        
        # Clean feedback text
        feedback_df['Feedback Received'] = feedback_df['Feedback Received'].fillna('')
        feedback_df = feedback_df[feedback_df['Feedback Received'].str.len() > 10]  # Filter out very short feedback
        
        # Save preprocessed data
        feedback_df.to_pickle('feedback_processed.pkl')
    
    return feedback_df

# Load NPS data for correlation analysis
def load_nps_data():
    """Load NPS data for validation"""
    try:
        nps_df = pd.read_csv('F:/UoB Study/Capstone Project/Final Project/Datasets/NPS_Updated_25_G24.csv')
        nps_df['Response Date'] = pd.to_datetime(nps_df['Response Date'], format='%d/%m/%Y')
        return nps_df
    except:
        print("⚠️ NPS data not found")
        return None

# Load the data
feedback_df = load_feedback_data()
nps_df = load_nps_data()

print(f"📊 Loaded {len(feedback_df)} feedback records")
print(f"📅 Date range: {feedback_df['Date Received2'].min()} to {feedback_df['Date Received2'].max()}")

# Text Preprocessing

class SentimentAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        self.stop_words = set(stopwords.words('english'))
        
        # Define facility categories for aspect-based analysis
        self.facility_keywords = {
            'pool': ['pool', 'swimming', 'swim', 'water', 'lane', 'depth', 'chlorine', 'poolside'],
            'gym': ['gym', 'equipment', 'weights', 'cardio', 'machines', 'treadmill', 'exercise'],
            'showers': ['shower', 'showers', 'water', 'temperature', 'pressure', 'cubicle'],
            'changing_rooms': ['changing', 'locker', 'lockers', 'cubicle', 'changing room'],
            'classes': ['class', 'classes', 'instructor', 'yoga', 'pilates', 'spin', 'body pump'],
            'reception': ['reception', 'staff', 'booking', 'front desk', 'customer service'],
            'facilities_general': ['facilities', 'building', 'cleanliness', 'maintenance', 'clean'],
            'parking': ['parking', 'car park', 'spaces', 'disabled parking'],
            'technology': ['app', 'wifi', 'booking system', 'online', 'website', 'digital'],
            'pricing': ['price', 'cost', 'membership', 'fee', 'expensive', 'value', 'money']
        }
        
        # Intensity indicators for scaled sentiment
        self.intensity_patterns = {
            'highly_negative': ['exceedingly', 'absolutely', 'extremely', 'completely', 'utterly', 
                              'appalling', 'disgusting', 'terrible', 'awful', 'hopeless'],
            'negative': ['poor', 'bad', 'disappointing', 'unsatisfactory', 'inadequate', 
                        'needs improvement', 'not good', 'lacking'],
            'positive': ['good', 'nice', 'satisfied', 'decent', 'adequate', 'helpful', 'pleased'],
            'highly_positive': ['excellent', 'fantastic', 'outstanding', 'brilliant', 'amazing', 
                              'wonderful', 'superb', 'exceptional', 'perfect']
        }
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def get_aspect_sentiment(self, text, aspect_keywords):
        """Extract sentiment for specific aspects"""
        text_lower = text.lower()
        aspect_sentences = []
        
        # Find sentences containing aspect keywords
        sentences = sent_tokenize(text)
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in aspect_keywords):
                aspect_sentences.append(sentence)
        
        if not aspect_sentences:
            return None
        
        # Calculate sentiment for aspect-related sentences
        aspect_text = ' '.join(aspect_sentences)
        sentiment = self.sia.polarity_scores(aspect_text)
        
        return {
            'compound': sentiment['compound'],
            'positive': sentiment['pos'],
            'negative': sentiment['neg'],
            'neutral': sentiment['neu'],
            'text': aspect_text
        }
    
    def get_intensity_score(self, text):
        """Get intensity-scaled sentiment score (-2 to +2)"""
        text_lower = text.lower()
        
        # Check for highly negative patterns
        for pattern in self.intensity_patterns['highly_negative']:
            if pattern in text_lower:
                return -2
        
        # Check for highly positive patterns
        for pattern in self.intensity_patterns['highly_positive']:
            if pattern in text_lower:
                return 2
        
        # Get basic sentiment
        sentiment = self.sia.polarity_scores(text)
        compound = sentiment['compound']
        
        # Scale to intensity levels
        if compound <= -0.5:
            return -1
        elif compound >= 0.5:
            return 1
        else:
            return 0
    
    def analyze_feedback(self, text):
        """Comprehensive sentiment analysis for a single feedback"""
        if pd.isna(text) or len(text.strip()) == 0:
            return None
        
        # Basic sentiment analysis
        sentiment = self.sia.polarity_scores(text)
        
        # Aspect-based sentiment
        aspects = {}
        for aspect, keywords in self.facility_keywords.items():
            aspect_sentiment = self.get_aspect_sentiment(text, keywords)
            if aspect_sentiment:
                aspects[aspect] = aspect_sentiment
        
        # Intensity score
        intensity = self.get_intensity_score(text)
        
        # TextBlob for additional validation
        blob = TextBlob(text)
        textblob_sentiment = blob.sentiment
        
        return {
            'compound': sentiment['compound'],
            'positive': sentiment['pos'],
            'negative': sentiment['neg'],
            'neutral': sentiment['neu'],
            'intensity_score': intensity,
            'textblob_polarity': textblob_sentiment.polarity,
            'textblob_subjectivity': textblob_sentiment.subjectivity,
            'aspects': aspects,
            'word_count': len(text.split()),
            'char_count': len(text)
        }

# Initialize analyzer
analyzer = SentimentAnalyzer()

# Process all feedback
print("🔄 Processing sentiment analysis...")
sentiment_results = []

for idx, row in feedback_df.iterrows():
    result = analyzer.analyze_feedback(row['Feedback Received'])
    if result:
        result['feedback_id'] = idx
        result['unique_id'] = row['Unique ID']
        result['date'] = row['Date Received2']
        result['form_type'] = row['Form of feedback']
        sentiment_results.append(result)

# Convert to DataFrame
sentiment_df = pd.DataFrame(sentiment_results)
print(f"✅ Processed {len(sentiment_df)} feedback entries")

# Aspect-based sentiment heatmap

def create_aspect_sentiment_heatmap():
    """Create heatmap showing sentiment by aspect and time period"""
    
    # Extract aspect sentiments
    aspect_data = []
    for idx, row in sentiment_df.iterrows():
        for aspect, sentiment_info in row['aspects'].items():
            aspect_data.append({
                'date': row['date'],
                'aspect': aspect.replace('_', ' ').title(),
                'sentiment': sentiment_info['compound'],
                'month_year': row['date'].strftime('%Y-%m')
            })
    
    if not aspect_data:
        print("⚠️ No aspect data found")
        return
    
    aspect_df = pd.DataFrame(aspect_data)
    
    # Create monthly averages
    monthly_aspect = aspect_df.groupby(['month_year', 'aspect'])['sentiment'].mean().reset_index()
    pivot_data = monthly_aspect.pivot(index='month_year', columns='aspect', values='sentiment')
    
    # Create heatmap
    plt.figure(figsize=(16, 10))
    sns.heatmap(pivot_data.T, 
                annot=True, 
                cmap='RdYlGn', 
                center=0,
                fmt='.2f',
                cbar_kws={'label': 'Sentiment Score'})
    
    plt.title('Aspect-Based Sentiment Analysis Over Time\nUniversity of Birmingham Sports & Fitness Centre', 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Month-Year', fontsize=12)
    plt.ylabel('Facility Aspects', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('aspect_sentiment_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

create_aspect_sentiment_heatmap()

# Intensity Scale Sentiment Analysis
def create_intensity_sentiment_distribution():
    """Create distribution of intensity-scaled sentiment scores"""
    
    intensity_labels = {-2: 'Highly Negative', -1: 'Negative', 0: 'Neutral', 
                       1: 'Positive', 2: 'Highly Positive'}
    
    # Count intensity scores
    intensity_counts = sentiment_df['intensity_score'].value_counts().sort_index()
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar chart
    colors = ['#d32f2f', '#f57c00', '#757575', '#388e3c', '#1976d2']
    bars = ax1.bar(range(len(intensity_counts)), intensity_counts.values, 
                   color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    
    ax1.set_xlabel('Sentiment Intensity Level', fontsize=12)
    ax1.set_ylabel('Number of Feedback Entries', fontsize=12)
    ax1.set_title('Distribution of Sentiment Intensity Levels', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(intensity_counts)))
    ax1.set_xticklabels([intensity_labels[i] for i in intensity_counts.index], rotation=45)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    ax2.pie(intensity_counts.values, 
            labels=[intensity_labels[i] for i in intensity_counts.index],
            colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Sentiment Intensity Distribution', fontsize=14, fontweight='bold')
    
    plt.suptitle('Sentiment Intensity Analysis - Sports & Fitness Centre', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('intensity_sentiment_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    total_feedback = len(sentiment_df)
    print(f"\n📊 Sentiment Intensity Statistics:")
    print(f"Total Feedback Analyzed: {total_feedback}")
    for score, count in intensity_counts.items():
        percentage = (count / total_feedback) * 100
        print(f"{intensity_labels[score]}: {count} ({percentage:.1f}%)")

create_intensity_sentiment_distribution()

# Word Cloud Visualization

def create_word_clouds_and_topics():
    """Create word clouds and perform topic modeling with improved visualizations"""
    
    # Separate feedback by sentiment
    positive_feedback = sentiment_df[sentiment_df['intensity_score'] >= 1]['feedback_id']
    negative_feedback = sentiment_df[sentiment_df['intensity_score'] <= -1]['feedback_id']
    
    # Get original feedback text
    positive_texts = feedback_df.loc[positive_feedback, 'Feedback Received'].str.cat(sep=' ')
    negative_texts = feedback_df.loc[negative_feedback, 'Feedback Received'].str.cat(sep=' ')
    
    # Create individual word clouds with proper formatting
    create_individual_word_clouds(positive_texts, negative_texts)
    
    # Perform topic modeling
    topics_words = perform_topic_modeling()
    
    # Create complaint frequency analysis
    create_complaint_frequency_analysis(negative_feedback)
    
    # Create temporal word clouds
    create_temporal_word_clouds()
    
    return topics_words

def create_individual_word_clouds(positive_texts, negative_texts):
    """Create separate word cloud visualizations"""
    
    # 1. Positive Feedback Word Cloud
    if positive_texts:
        plt.figure(figsize=(16, 10))
        positive_wordcloud = WordCloud(
            width=1600, height=800, 
            background_color='white',
            colormap='Greens',
            max_words=150,
            stopwords=analyzer.stop_words,
            collocations=False,
            relative_scaling=0.5,
            min_font_size=10,
            max_font_size=100
        ).generate(positive_texts)
        
        plt.imshow(positive_wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Positive Feedback Word Cloud\nSports & Fitness Centre', 
                 fontsize=20, fontweight='bold', pad=30, color='darkgreen')
        plt.tight_layout()
        plt.savefig('positive_feedback_wordcloud.png', 
                   dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.show()
        plt.close()

    # 2. Negative Feedback Word Cloud
    if negative_texts:
        plt.figure(figsize=(16, 10))
        negative_wordcloud = WordCloud(
            width=1600, height=800, 
            background_color='white',
            colormap='Reds',
            max_words=150,
            stopwords=analyzer.stop_words,
            collocations=False,
            relative_scaling=0.5,
            min_font_size=10,
            max_font_size=100
        ).generate(negative_texts)
        
        plt.imshow(negative_wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Negative Feedback Word Cloud\nSports & Fitness Centre', 
                 fontsize=20, fontweight='bold', pad=30, color='darkred')
        plt.tight_layout()
        plt.savefig('negative_feedback_wordcloud.png', 
                   dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.show()
        plt.close()

def perform_topic_modeling():
    """Perform topic modeling and create visualization"""
    
    # Topic modeling preparation
    all_texts = feedback_df['Feedback Received'].dropna().tolist()
    
    # Clean texts for topic modeling
    def clean_for_topics(text):
        text = analyzer.clean_text(text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in analyzer.stop_words and len(word) > 2]
        return ' '.join(tokens)
    
    cleaned_texts = [clean_for_topics(text) for text in all_texts if len(text) > 20]
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(max_features=100, max_df=0.8, min_df=2)
    tfidf_matrix = vectorizer.fit_transform(cleaned_texts)
    
    # LDA Topic Modeling
    n_topics = 6
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=10)
    lda.fit(tfidf_matrix)
    
    # Get top words for each topic
    feature_names = vectorizer.get_feature_names_out()
    topics_words = []
    
    for topic_idx, topic in enumerate(lda.components_):
        top_words_idx = topic.argsort()[-10:][::-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topics_words.append(top_words)
    
    # Create topic visualization
    plt.figure(figsize=(16, 12))
    
    # Create a table-like visualization for topics
    topic_labels = [f"Topic {i+1}" for i in range(n_topics)]
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table data
    table_data = []
    for i, words in enumerate(topics_words):
        table_data.append([f'Topic {i+1}', ', '.join(words[:8])])
    
    # Create table
    table = ax.table(cellText=table_data,
                    colLabels=['Topic', 'Top Keywords'],
                    cellLoc='left',
                    loc='center',
                    colWidths=[0.15, 0.85])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    
    # Style the table
    for i in range(len(table_data) + 1):
        for j in range(2):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor('#40466e')
                cell.set_text_props(weight='bold', color='white')
            else:
                if i % 2 == 0:
                    cell.set_facecolor('#f1f1f2')
                else:
                    cell.set_facecolor('white')
                cell.set_text_props(wrap=True)
    
    plt.title('Topic Analysis - Key Themes in Customer Feedback\nSports & Fitness Centre', 
             fontsize=18, fontweight='bold', pad=30)
    plt.tight_layout()
    plt.savefig('topic_analysis.png', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.show()
    plt.close()
    
    return topics_words

def create_complaint_frequency_analysis(negative_feedback):
    """Create detailed complaint frequency analysis"""
    
    # Extract complaint words
    complaint_words = []
    complaint_categories = defaultdict(list)
    
    for idx in negative_feedback:
        if idx in feedback_df.index:
            text = feedback_df.loc[idx, 'Feedback Received']
            words = analyzer.clean_text(text).split()
            complaint_words.extend([w for w in words if w not in analyzer.stop_words and len(w) > 3])
            
            # Categorize complaints
            text_lower = text.lower()
            for category, keywords in analyzer.facility_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    complaint_categories[category.replace('_', ' ').title()].append(text)
    
    # Create complaint word frequency chart
    if complaint_words:
        complaint_freq = Counter(complaint_words).most_common(20)
        words, counts = zip(*complaint_freq)
        
        plt.figure(figsize=(14, 10))
        bars = plt.barh(range(len(words)), counts, color='lightcoral', alpha=0.8, edgecolor='darkred')
        plt.yticks(range(len(words)), words)
        plt.xlabel('Frequency', fontsize=12, fontweight='bold')
        plt.ylabel('Keywords', fontsize=12, fontweight='bold')
        plt.title('Most Frequent Keywords in Negative Feedback\nSports & Fitness Centre', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{int(width)}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('complaint_frequency_analysis.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.show()
        plt.close()
    
    # Create complaint categories chart
    if complaint_categories:
        categories = list(complaint_categories.keys())
        category_counts = [len(complaints) for complaints in complaint_categories.values()]
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(categories, category_counts, 
                      color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', '#dda0dd'])
        
        plt.xlabel('Complaint Categories', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Complaints', fontsize=12, fontweight='bold')
        plt.title('Complaint Distribution by Category\nSports & Fitness Centre', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('complaint_categories.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.show()
        plt.close()

def create_temporal_word_clouds():
    """Create word clouds for different time periods"""
    
    # Group feedback by year
    feedback_df['year'] = feedback_df['Date Received2'].dt.year
    years = feedback_df['year'].unique()
    
    if len(years) > 1:
        # Create word clouds for each year
        for year in sorted(years):
            year_feedback = feedback_df[feedback_df['year'] == year]['Feedback Received']
            year_text = year_feedback.str.cat(sep=' ')
            
            if len(year_text) > 100:  # Only create if sufficient text
                plt.figure(figsize=(14, 8))
                wordcloud = WordCloud(
                    width=1400, height=700,
                    background_color='white',
                    colormap='viridis',
                    max_words=100,
                    stopwords=analyzer.stop_words,
                    collocations=False,
                    relative_scaling=0.5
                ).generate(year_text)
                
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title(f'Customer Feedback Word Cloud - {year}\nSports & Fitness Centre', 
                         fontsize=16, fontweight='bold', pad=20)
                plt.tight_layout()
                plt.savefig(f'wordcloud_{year}.png', dpi=300, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
                plt.show()
                plt.close()

def create_sentiment_word_comparison():
    """Create a side-by-side comparison of positive vs negative words"""
    
    # Get most common words from positive and negative feedback
    positive_feedback_ids = sentiment_df[sentiment_df['intensity_score'] >= 1]['feedback_id']
    negative_feedback_ids = sentiment_df[sentiment_df['intensity_score'] <= -1]['feedback_id']
    
    positive_words = []
    negative_words = []
    
    # Extract words from positive feedback
    for idx in positive_feedback_ids:
        if idx in feedback_df.index:
            text = feedback_df.loc[idx, 'Feedback Received']
            words = analyzer.clean_text(text).split()
            positive_words.extend([w for w in words if w not in analyzer.stop_words and len(w) > 3])
    
    # Extract words from negative feedback
    for idx in negative_feedback_ids:
        if idx in feedback_df.index:
            text = feedback_df.loc[idx, 'Feedback Received']
            words = analyzer.clean_text(text).split()
            negative_words.extend([w for w in words if w not in analyzer.stop_words and len(w) > 3])
    
    # Get top words
    positive_freq = Counter(positive_words).most_common(15)
    negative_freq = Counter(negative_words).most_common(15)
    
    # Create comparison chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
    
    # Positive words
    if positive_freq:
        pos_words, pos_counts = zip(*positive_freq)
        bars1 = ax1.barh(range(len(pos_words)), pos_counts, color='lightgreen', alpha=0.8)
        ax1.set_yticks(range(len(pos_words)))
        ax1.set_yticklabels(pos_words)
        ax1.set_xlabel('Frequency', fontweight='bold')
        ax1.set_title('Most Frequent Words in\nPositive Feedback', fontsize=14, fontweight='bold', color='darkgreen')
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars1):
            width = bar.get_width()
            ax1.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{int(width)}', ha='left', va='center', fontweight='bold')
    
    # Negative words
    if negative_freq:
        neg_words, neg_counts = zip(*negative_freq)
        bars2 = ax2.barh(range(len(neg_words)), neg_counts, color='lightcoral', alpha=0.8)
        ax2.set_yticks(range(len(neg_words)))
        ax2.set_yticklabels(neg_words)
        ax2.set_xlabel('Frequency', fontweight='bold')
        ax2.set_title('Most Frequent Words in\nNegative Feedback', fontsize=14, fontweight='bold', color='darkred')
        ax2.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, bar in enumerate(bars2):
            width = bar.get_width()
            ax2.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{int(width)}', ha='left', va='center', fontweight='bold')
    
    plt.suptitle('Sentiment-Based Word Frequency Comparison\nSports & Fitness Centre', 
                 fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    plt.savefig('sentiment_word_comparison.png', dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.show()
    plt.close()

# Execute the improved word cloud analysis
print("🔄 Creating improved word cloud and topic analysis...")
topics = create_word_clouds_and_topics()

# Create additional comparison analysis
create_sentiment_word_comparison()

print("✅ Improved visualizations created successfully!")
print("📁 Generated Files:")
print("   • positive_feedback_wordcloud.png")
print("   • negative_feedback_wordcloud.png")
print("   • topic_analysis.png")
print("   • complaint_frequency_analysis.png")
print("   • complaint_categories.png")
print("   • sentiment_word_comparison.png")
print("   • wordcloud_[year].png (for each year in data)")

#Sentiment Dashboard
def create_sentiment_dashboard_summary():
    """Create a comprehensive dashboard summary"""
    
    # Calculate key metrics
    total_feedback = len(sentiment_df)
    avg_sentiment = sentiment_df['compound'].mean()
    
    positive_count = len(sentiment_df[sentiment_df['intensity_score'] > 0])
    negative_count = len(sentiment_df[sentiment_df['intensity_score'] < 0])
    neutral_count = len(sentiment_df[sentiment_df['intensity_score'] == 0])
    
    # Create comprehensive summary visualization
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=[
            'Overall Sentiment Distribution', 'Sentiment Trend (3-Month Rolling)',
            'Top Positive Aspects', 'Top Negative Aspects',
            'Monthly Feedback Volume', 'Sentiment by Feedback Type',
            'Word Frequency Analysis', 'Sentiment Intensity Heatmap',
            'Key Performance Indicators'
        ],
        specs=[[{"type": "pie"}, {"type": "scatter"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "scatter"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "heatmap"}, {"type": "indicator"}]]
    )
    
    # 1. Overall sentiment pie chart
    sentiment_labels = ['Positive', 'Neutral', 'Negative']
    sentiment_values = [positive_count, neutral_count, negative_count]
    sentiment_colors = ['#2E8B57', '#FFD700', '#DC143C']
    
    fig.add_trace(
        go.Pie(labels=sentiment_labels, values=sentiment_values, 
               marker_colors=sentiment_colors, name="Sentiment"),
        row=1, col=1
    )
    
    # 2. Sentiment trend
    monthly_sentiment['rolling_avg'] = monthly_sentiment['compound'].rolling(window=3, center=True).mean()
    
    fig.add_trace(
        go.Scatter(x=monthly_sentiment['year_month_str'], 
                  y=monthly_sentiment['rolling_avg'],
                  mode='lines+markers',
                  name='3-Month Avg',
                  line=dict(color='blue', width=3)),
        row=1, col=2
    )
    
    # 3. Top positive aspects
    positive_aspects = []
    for idx, row in sentiment_df[sentiment_df['intensity_score'] > 0].iterrows():
        for aspect, sentiment_info in row['aspects'].items():
            if sentiment_info['compound'] > 0.1:
                positive_aspects.append(aspect.replace('_', ' ').title())
    
    if positive_aspects:
        pos_aspect_counts = Counter(positive_aspects).most_common(5)
        pos_aspects, pos_counts = zip(*pos_aspect_counts)
        
        fig.add_trace(
            go.Bar(x=list(pos_aspects), y=list(pos_counts),
                  marker_color='green', name='Positive Mentions'),
            row=1, col=3
        )
    
    # 4. Top negative aspects
    negative_aspects = []
    for idx, row in sentiment_df[sentiment_df['intensity_score'] < 0].iterrows():
        for aspect, sentiment_info in row['aspects'].items():
            if sentiment_info['compound'] < -0.1:
                negative_aspects.append(aspect.replace('_', ' ').title())
    
    if negative_aspects:
        neg_aspect_counts = Counter(negative_aspects).most_common(5)
        neg_aspects, neg_counts = zip(*neg_aspect_counts)
        
        fig.add_trace(
            go.Bar(x=list(neg_aspects), y=list(neg_counts),
                  marker_color='red', name='Negative Mentions'),
            row=2, col=1
        )
    
    # 5. Monthly feedback volume
    monthly_volume = sentiment_df.groupby('year_month').size().reset_index(name='count')
    monthly_volume['year_month_str'] = monthly_volume['year_month'].astype(str)
    
    fig.add_trace(
        go.Scatter(x=monthly_volume['year_month_str'], 
                  y=monthly_volume['count'],
                  mode='lines+markers',
                  name='Feedback Volume',
                  line=dict(color='purple')),
        row=2, col=2
    )
    
    # 6. Sentiment by feedback type
    if 'form_type' in sentiment_df.columns:
        type_sentiment = sentiment_df.groupby('form_type')['compound'].mean().reset_index()
        
        fig.add_trace(
            go.Bar(x=type_sentiment['form_type'], 
                  y=type_sentiment['compound'],
                  marker_color='orange',
                  name='Avg Sentiment'),
            row=2, col=3
        )
    
    # 7. Key indicators
    fig.add_trace(
        go.Indicator(
            mode="number+gauge+delta",
            value=avg_sentiment,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={"text": "Overall Sentiment Score"},
            gauge={'axis': {'range': [-1, 1]},
                   'bar': {'color': "darkblue"},
                   'steps': [{'range': [-1, -0.5], 'color': "lightgray"},
                            {'range': [-0.5, 0.5], 'color': "yellow"},
                            {'range': [0.5, 1], 'color': "lightgreen"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                                'thickness': 0.75, 'value': 0}}),
        row=3, col=3
    )
    
    # Update layout
    fig.update_layout(
        height=1200,
        title_text="Sports & Fitness Centre - Comprehensive Sentiment Analysis Dashboard",
        title_x=0.5,
        title_font_size=20,
        showlegend=False
    )
    
    # Save and show
    fig.write_html('sentiment_dashboard_summary.html')
    fig.show()
    
    # Print executive summary
    print("\n" + "="*80)
    print("📊 EXECUTIVE SENTIMENT ANALYSIS SUMMARY")
    print("="*80)
    print(f"🔢 Total Feedback Analyzed: {total_feedback:,}")
    print(f"📈 Overall Sentiment Score: {avg_sentiment:.3f} (Scale: -1 to +1)")
    print(f"😊 Positive Feedback: {positive_count} ({positive_count/total_feedback*100:.1f}%)")
    print(f"😐 Neutral Feedback: {neutral_count} ({neutral_count/total_feedback*100:.1f}%)")
    print(f"😞 Negative Feedback: {negative_count} ({negative_count/total_feedback*100:.1f}%)")
    print("\n🎯 KEY INSIGHTS:")
    
    if avg_sentiment > 0.1:
        print("✅ Overall sentiment is POSITIVE")
    elif avg_sentiment < -0.1:
        print("❌ Overall sentiment is NEGATIVE")
    else:
        print("⚖️ Overall sentiment is NEUTRAL")
    
    # Top issues
    if negative_aspects:
        print(f"\n🔴 TOP COMPLAINT AREAS:")
        for aspect, count in Counter(negative_aspects).most_common(3):
            print(f"   • {aspect}: {count} mentions")
    
    # Top strengths
    if positive_aspects:
        print(f"\n🟢 TOP STRENGTH AREAS:")
        for aspect, count in Counter(positive_aspects).most_common(3):
            print(f"   • {aspect}: {count} mentions")
    
    print("\n" + "="*80)

create_sentiment_dashboard_summary()

