import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SentimentVisualizer:
    def __init__(self, output_dir: str = "visualizations"):
        """
        Initialize the visualizer
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn')
        sns.set_palette("husl")
    
    def plot_sentiment_distribution(self, df: pd.DataFrame, title: str = "Sentiment Distribution"):
        """
        Plot the distribution of sentiments
        
        Args:
            df: DataFrame containing sentiment data
            title: Plot title
        """
        try:
            plt.figure(figsize=(10, 6))
            sentiment_counts = df['sentiment'].value_counts()
            
            # Create pie chart
            plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%')
            plt.title(title)
            
            # Save plot
            filename = os.path.join(self.output_dir, "sentiment_distribution.png")
            plt.savefig(filename)
            plt.close()
            
            logger.info(f"Saved sentiment distribution plot to {filename}")
            
        except Exception as e:
            logger.error(f"Error plotting sentiment distribution: {str(e)}")
            raise
    
    def plot_rating_distribution(self, df: pd.DataFrame, title: str = "Rating Distribution"):
        """
        Plot the distribution of ratings
        
        Args:
            df: DataFrame containing rating data
            title: Plot title
        """
        try:
            plt.figure(figsize=(10, 6))
            
            # Create histogram
            sns.histplot(data=df, x='rating', bins=5)
            plt.title(title)
            plt.xlabel("Rating")
            plt.ylabel("Count")
            
            # Save plot
            filename = os.path.join(self.output_dir, "rating_distribution.png")
            plt.savefig(filename)
            plt.close()
            
            logger.info(f"Saved rating distribution plot to {filename}")
            
        except Exception as e:
            logger.error(f"Error plotting rating distribution: {str(e)}")
            raise
    
    def plot_sentiment_trends(self, df: pd.DataFrame, title: str = "Sentiment Trends Over Time"):
        """
        Plot sentiment trends over time
        
        Args:
            df: DataFrame containing sentiment and date data
            title: Plot title
        """
        try:
            plt.figure(figsize=(12, 6))
            
            # Convert date strings to datetime
            df['date'] = pd.to_datetime(df['date'])
            
            # Group by date and sentiment
            sentiment_by_date = df.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
            
            # Create line plot
            sentiment_by_date.plot(kind='line', marker='o')
            plt.title(title)
            plt.xlabel("Date")
            plt.ylabel("Number of Reviews")
            plt.legend(title="Sentiment")
            plt.xticks(rotation=45)
            
            # Save plot
            filename = os.path.join(self.output_dir, "sentiment_trends.png")
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved sentiment trends plot to {filename}")
            
        except Exception as e:
            logger.error(f"Error plotting sentiment trends: {str(e)}")
            raise
    
    def plot_word_cloud(self, df: pd.DataFrame, sentiment: str = None, title: str = "Word Cloud"):
        """
        Generate word cloud for reviews
        
        Args:
            df: DataFrame containing review data
            sentiment: Filter by sentiment (optional)
            title: Plot title
        """
        try:
            from wordcloud import WordCloud
            
            plt.figure(figsize=(12, 8))
            
            # Filter by sentiment if specified
            if sentiment:
                text_data = ' '.join(df[df['sentiment'] == sentiment]['cleaned_text'])
                title = f"Word Cloud - {sentiment.capitalize()} Reviews"
            else:
                text_data = ' '.join(df['cleaned_text'])
            
            # Generate word cloud
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)
            
            # Plot word cloud
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.title(title)
            
            # Save plot
            filename = os.path.join(self.output_dir, f"wordcloud_{sentiment if sentiment else 'all'}.png")
            plt.savefig(filename, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved word cloud to {filename}")
            
        except Exception as e:
            logger.error(f"Error generating word cloud: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    # Load processed data
    df = pd.read_csv("data/processed_reviews.csv")
    
    # Initialize visualizer
    visualizer = SentimentVisualizer()
    
    # Generate visualizations
    visualizer.plot_sentiment_distribution(df)
    visualizer.plot_rating_distribution(df)
    visualizer.plot_sentiment_trends(df)
    visualizer.plot_word_cloud(df)
    visualizer.plot_word_cloud(df, sentiment='positive')
    visualizer.plot_word_cloud(df, sentiment='negative')
    visualizer.plot_word_cloud(df, sentiment='neutral') 