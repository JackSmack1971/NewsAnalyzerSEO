import asyncio
import csv
import json
import logging
import os
import re
from typing import Dict, List, Tuple

import aiohttp
import aiofiles
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from lxml import etree
from openai import AsyncOpenAI
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# Load environment variables
load_dotenv('/storage/emulated/0/NEWSANALYST/.env')

# Constants
MAX_RETRIES = 3
RETRY_DELAY = 20
MAX_CONCURRENT_REQUESTS = 10
NUM_CLUSTERS = 5
NUM_RESULTS = 20

# Configure logging
logging.basicConfig(
    filename='/storage/emulated/0/NEWSANALYST/news_topic_analyzer.log', 
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize AsyncOpenAI client
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class SeoKeywordResearch:
    def __init__(self, query: str, api_key: str, lang: str = 'en', country: str = 'us', domain: str = 'google.com'):
        self.query = query
        self.api_key = api_key
        self.lang = lang
        self.country = country
        self.domain = domain
        self.session = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.session.close()

    async def get_auto_complete(self) -> List[str]:
        params = {
            'api_key': self.api_key,
            'engine': 'google_autocomplete',
            'q': self.query,
            'gl': self.country,
            'hl': self.lang
        }
        async with self.session.get('https://serpapi.com/search', params=params) as response:
            results = await response.json()
            return [result.get('value') for result in results.get('suggestions', [])]

    async def get_related_searches(self) -> List[str]:
        params = {
            'api_key': self.api_key,
            'engine': 'google',
            'q': self.query,
            'google_domain': self.domain,
            'gl': self.country,
            'hl': self.lang
        }
        async with self.session.get('https://serpapi.com/search', params=params) as response:
            results = await response.json()
            return [result.get('query') for result in results.get('related_searches', [])]

    async def get_related_questions(self, depth_limit: int = 0) -> List[str]:
        params = {
            'api_key': self.api_key,
            'engine': 'google',
            'q': self.query,
            'google_domain': self.domain,
            'gl': self.country,
            'hl': self.lang
        }
        async with self.session.get('https://serpapi.com/search', params=params) as response:
            results = await response.json()
            related_questions = [result.get('question') for result in results.get('related_questions', [])]
            if depth_limit > 0:
                tasks = [self._get_depth_results(question.get('next_page_token'), depth_limit - 1)
                         for question in results.get('related_questions', [])
                         if question.get('next_page_token')]
                additional_questions = await asyncio.gather(*tasks)
                related_questions.extend([item for sublist in additional_questions for item in sublist])
            return related_questions

    async def _get_depth_results(self, token: str, depth: int) -> List[str]:
        params = {
            'api_key': self.api_key,
            'engine': 'google_related_questions',
            'next_page_token': token,
        }
        async with self.session.get('https://serpapi.com/search', params=params) as response:
            results = await response.json()
            questions = [result.get('question') for result in results.get('related_questions', [])]
            if depth > 1:
                tasks = [self._get_depth_results(question.get('next_page_token'), depth - 1)
                         for question in results.get('related_questions', [])
                         if question.get('next_page_token')]
                additional_questions = await asyncio.gather(*tasks)
                questions.extend([item for sublist in additional_questions for item in sublist])
            return questions

async def generate_google_search_query(user_input: str) -> str:
    prompt = f"Convert the following user query into an optimized Google Search query: '{user_input}'"
    try:
        completion = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a Google Search Expert. Your task is to convert unstructured user inputs to optimized Google search queries."},
                {"role": "user", "content": prompt}
            ]
        )
        if completion.choices:
            return completion.choices[0].message.content.strip()
        else:
            return "No response from GPT-4 Turbo."
    except Exception as e:
        logging.error(f"Error in generating Google search query: {e}")
        return None

async def get_organic_results(query: str, num_results: int = NUM_RESULTS, location: str = "United States") -> List[Dict]:
    params = {
        "q": query,
        "tbm": "nws",
        "location": location,
        "num": str(num_results),
        "api_key": os.getenv("SERP_API_KEY")
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get('https://serpapi.com/search', params=params) as response:
                results = await response.json()
                return results.get("news_results", [])
    except Exception as e:
        logging.error(f"Error in fetching news results: {e}")
        return []

async def scrape_website(url: str) -> Tuple[str, str]:
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    content = await response.text()
                    parser = etree.HTMLParser()
                    tree = etree.fromstring(content, parser)
                    paragraphs = tree.xpath('//p/text()')
                    formatted_data = "\n".join(paragraphs)
                    return url, formatted_data
                else:
                    logging.warning(f"Failed to retrieve the webpage: {url}")
                    return url, "Failed to retrieve the webpage"
    except Exception as e:
        logging.error(f"Error in scraping website {url}: {e}")
        return url, f"Error occurred while scraping: {str(e)}"

def cluster_news_topics(news_results: List[Dict], n_clusters: int = NUM_CLUSTERS) -> Dict[int, List[Dict]]:
    if not news_results:
        logging.warning("No news results to cluster")
        return {}
    
    titles = [article['title'] for article in news_results]
    logging.info(f"Titles to cluster: {titles}")
    
    try:
        vectorizer = TfidfVectorizer(stop_words=None, token_pattern=r'\b\w+\b', min_df=1)
        X = vectorizer.fit_transform(titles)
        kmeans = KMeans(n_clusters=min(n_clusters, len(titles)), n_init=10)
        kmeans.fit(X)
        
        clustered_news = {}
        for i, label in enumerate(kmeans.labels_):
            if label not in clustered_news:
                clustered_news[label] = []
            clustered_news[label].append(news_results[i])
        return clustered_news
    except ValueError as e:
        logging.error(f"Error in clustering: {e}")
        return {0: news_results}  # Return all results in a single cluster

async def analyze_keywords(query: str) -> Dict[str, List[str]]:
    serp_api_key = os.getenv("SERP_API_KEY")
    if not serp_api_key:
        logging.error("SERP API key not found in environment variables")
        return {"error": "SERP API key not configured"}
    
    try:
        async with SeoKeywordResearch(query, serp_api_key) as seo_research:
            auto_complete = await seo_research.get_auto_complete()
            related_searches = await seo_research.get_related_searches()
            related_questions = await seo_research.get_related_questions(depth_limit=2)
        
        logging.info(f"Keyword analysis completed for query: {query}")
        logging.info(f"Auto-complete suggestions: {len(auto_complete)}")
        logging.info(f"Related searches: {len(related_searches)}")
        logging.info(f"Related questions: {len(related_questions)}")
        
        return {
            "auto_complete": auto_complete,
            "related_searches": related_searches,
            "related_questions": related_questions
        }
    except Exception as e:
        logging.error(f"Error in keyword analysis for query '{query}': {e}")
        return {"error": f"Failed to analyze keywords: {str(e)}"}

def sanitize_filename(filename: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*\n\r]', '_', filename)
    sanitized = re.sub(r'_+', '_', sanitized)
    sanitized = sanitized.strip('_')
    return sanitized[:50]

async def save_keyword_data(data: Dict[str, List[str]], query: str):
    filename = sanitize_filename(f'/storage/emulated/0/NEWSANALYST/{query}_keyword_data')
    
    try:
        # Save as CSV
        async with aiofiles.open(f'{filename}.csv', 'w', newline='', encoding='utf-8') as csv_file:
            # Write header
            await csv_file.write(','.join(data.keys()) + '\n')
            
            # Find the maximum length of any list in the data
            max_length = max(len(lst) for lst in data.values() if isinstance(lst, list))
            
            # Write data rows
            for i in range(max_length):
                row = []
                for key in data.keys():
                    if isinstance(data[key], list) and i < len(data[key]):
                        row.append(data[key][i])
                    else:
                        row.append('')
                await csv_file.write(','.join(row) + '\n')

        # Save as JSON
        async with aiofiles.open(f'{filename}.json', 'w', encoding='utf-8') as json_file:
            await json_file.write(json.dumps(data, indent=2, ensure_ascii=False))

        # Save as TXT
        async with aiofiles.open(f'{filename}.txt', 'w', encoding='utf-8') as txt_file:
            for key, values in data.items():
                await txt_file.write(f"{key}:\n")
                if isinstance(values, list):
                    await txt_file.write('\n'.join(values) + '\n\n')
                elif isinstance(values, str):
                    await txt_file.write(values + '\n\n')
                else:
                    await txt_file.write(f"Unexpected data type for {key}: {type(values)}\n\n")

        logging.info(f"Keyword data saved successfully for query: {query}")
    except Exception as e:
        logging.error(f"Error saving keyword data for query '{query}': {e}")

async def analyze_news_query(query: str) -> str:
    logging.info(f"User query: {query}")
    google_search_query = await generate_google_search_query(query)
    logging.info(f"Generated search query: {google_search_query}")

    if google_search_query:
        news_results = await get_organic_results(google_search_query)
        clustered_news = cluster_news_topics(news_results)
        keyword_data = await analyze_keywords(google_search_query)

        await save_keyword_data(keyword_data, query)
        logging.info(f"Keyword data saved for query: {query}")

        analysis_request = "Analyze the following clustered news topics and SEO keyword data:\n\nNews Clusters:\n"
        for cluster, articles in clustered_news.items():
            analysis_request += f"Cluster {cluster}:\n"
            for article in articles[:3]:
                analysis_request += f"- {article['title']}\n"
            analysis_request += "\n"

        analysis_request += "SEO Keyword Data:\n"
        for key, values in keyword_data.items():
            analysis_request += f"{key.replace('_', ' ').title()}:\n"
            for value in values[:5]:
                analysis_request += f"- {value}\n"
            analysis_request += "\n"

        analysis_request += "\nBased on this data, please provide your analysis and recommendations following the structure outlined in your instructions."

        try:
            completion = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """You are a professional news analyst tasked with providing concise and insightful analysis of current news topics and identifying SEO opportunities. Your goal is to deliver a comprehensive yet brief analysis that will help content creators and marketers capitalize on trending news. Please provide your analysis following these steps:
1. Summarize the news topic in 2-3 sentences, highlighting the key points.
2. Analyze the significance of this news topic in the current media landscape.
3. Identify potential SEO opportunities related to this topic, considering the provided trend data.
4. Provide 2-3 actionable recommendations for content creators or marketers to capitalize on this news topic.
Present your analysis in a clear, professional manner. Use concise language and avoid unnecessary jargon. Your entire response should be between 200-300 words. Format your response using the following XML tags:
<summary> for the news topic summary
<significance> for the analysis of the topic's significance
<seo_opportunities> for identified SEO opportunities
<recommendations> for your actionable recommendations"""},
                    {"role": "user", "content": analysis_request}
                ]
            )
            logging.info("Successfully generated news topic and SEO analysis")
            return completion.choices[0].message.content
        except Exception as e:
            logging.error(f"Error in news topic and SEO analysis: {e}")
            return f"An error occurred while analyzing news topics and SEO data: {e}"
    else:
        logging.error("Failed to generate Google search query")
        return "Failed to generate a Google search query."

async def retry_with_exponential_backoff(func, *args, max_retries=MAX_RETRIES, base_delay=RETRY_DELAY):
    """
    Retry a function with exponential backoff.
    """
    for attempt in range(max_retries):
        try:
            return await func(*args)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            logging.warning(f"Attempt {attempt + 1} failed. Retrying in {delay} seconds...")
            await asyncio.sleep(delay)

async def fetch_additional_data(url: str) -> Dict:
    """
    Fetch additional metadata from a news article URL.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    soup = BeautifulSoup(await response.text(), 'html.parser')
                    return {
                        'title': soup.title.string if soup.title else '',
                        'description': soup.find('meta', attrs={'name': 'description'})['content'] if soup.find('meta', attrs={'name': 'description'}) else '',
                        'keywords': soup.find('meta', attrs={'name': 'keywords'})['content'] if soup.find('meta', attrs={'name': 'keywords'}) else '',
                    }
                else:
                    logging.warning(f"Failed to fetch additional data from {url}")
                    return {}
    except Exception as e:
        logging.error(f"Error in fetching additional data from {url}: {e}")
        return {}

async def enrich_news_results(news_results: List[Dict]) -> List[Dict]:
    """
    Enrich news results with additional metadata.
    """
    enriched_results = []
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_additional_data(article['link']) for article in news_results]
        additional_data = await asyncio.gather(*tasks)
        
        for article, data in zip(news_results, additional_data):
            enriched_article = {**article, **data}
            enriched_results.append(enriched_article)
    
    return enriched_results

def calculate_topic_relevance(article: Dict, query: str) -> float:
    """
    Calculate the relevance of an article to the query.
    """
    query_words = set(query.lower().split())
    title_words = set(article['title'].lower().split())
    description_words = set(article.get('description', '').lower().split())
    
    title_relevance = len(query_words.intersection(title_words)) / len(query_words)
    description_relevance = len(query_words.intersection(description_words)) / len(query_words)
    
    return (title_relevance * 0.7) + (description_relevance * 0.3)

def sort_news_by_relevance(news_results: List[Dict], query: str) -> List[Dict]:
    """
    Sort news results by their relevance to the query.
    """
    return sorted(news_results, key=lambda x: calculate_topic_relevance(x, query), reverse=True)

async def generate_summary(text: str) -> str:
    """
    Generate a summary of the given text using GPT-4 Turbo.
    """
    try:
        completion = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a summarization expert. Provide a concise summary of the given text."},
                {"role": "user", "content": f"Please summarize the following text in 2-3 sentences:\n\n{text}"}
            ]
        )
        if completion.choices:
            return completion.choices[0].message.content.strip()
        else:
            return "Unable to generate summary."
    except Exception as e:
        logging.error(f"Error in generating summary: {e}")
        return "Error occurred while generating summary."

async def analyze_sentiment(text: str) -> str:
    """
    Analyze the sentiment of the given text using GPT-4 Turbo.
    """
    try:
        completion = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a sentiment analysis expert. Provide a brief sentiment analysis of the given text."},
                {"role": "user", "content": f"Please analyze the sentiment of the following text in 1-2 sentences:\n\n{text}"}
            ]
        )
        if completion.choices:
            return completion.choices[0].message.content.strip()
        else:
            return "Unable to analyze sentiment."
    except Exception as e:
        logging.error(f"Error in analyzing sentiment: {e}")
        return "Error occurred while analyzing sentiment."

async def perform_advanced_analysis(query: str, news_results: List[Dict]) -> Dict:
    """
    Perform advanced analysis on the news results.
    """
    enriched_results = await enrich_news_results(news_results)
    sorted_results = sort_news_by_relevance(enriched_results, query)
    
    top_article = sorted_results[0] if sorted_results else None
    
    if top_article:
        summary = await generate_summary(top_article['description'])
        sentiment = await analyze_sentiment(top_article['description'])
    else:
        summary = "No relevant articles found."
        sentiment = "Unable to analyze sentiment due to lack of relevant articles."
    
    return {
        'top_article': top_article,
        'summary': summary,
        'sentiment': sentiment,
        'sorted_results': sorted_results[:5]  # Return top 5 most relevant articles
    }

async def comprehensive_news_analysis(query: str) -> Dict:
    """
    Perform a comprehensive news analysis for the given query.
    """
    try:
        google_search_query = await retry_with_exponential_backoff(generate_google_search_query, query)
        if not google_search_query:
            return {"error": "Failed to generate Google search query."}
        
        news_results = await retry_with_exponential_backoff(get_organic_results, google_search_query)
        if not news_results:
            return {"error": "Failed to fetch news results."}
        
        clustered_news = cluster_news_topics(news_results)
        keyword_data = await analyze_keywords(google_search_query)
        await save_keyword_data(keyword_data, query)
        
        advanced_analysis = await perform_advanced_analysis(query, news_results)
        
        gpt_analysis = await analyze_news_query(query)
        
        return {
            "google_search_query": google_search_query,
            "clustered_news": clustered_news,
            "keyword_data": keyword_data,
            "advanced_analysis": advanced_analysis,
            "gpt_analysis": gpt_analysis
        }
    except Exception as e:
        logging.error(f"Error in comprehensive news analysis: {e}")
        return {"error": f"An error occurred during the analysis: {str(e)}"}

# If you want to run this module independently for testing
if __name__ == "__main__":
    async def main():
        query = "Recent advancements in artificial intelligence"
        result = await comprehensive_news_analysis(query)
        print(json.dumps(result, indent=2))

    asyncio.run(main())