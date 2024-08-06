from kivy.app import App
from kivy.clock import Clock
import asyncio
import json

from news_analyzer_core import comprehensive_news_analysis
from news_analyzer_ui import NewsAnalyzerUI

class NewsAnalyzerApp(App):
    def build(self):
        self.ui = NewsAnalyzerUI(analyze_callback=self.start_analysis)
        return self.ui

    def start_analysis(self, query):
        Clock.schedule_once(lambda dt: asyncio.run(self.run_analysis(query)), 0)

    async def run_analysis(self, query):
        self.ui.update_results("Analyzing... Please wait.")
        results = await comprehensive_news_analysis(query)
        formatted_results = self.format_results(results)
        Clock.schedule_once(lambda dt: self.ui.update_results(formatted_results), 0)

    def format_results(self, results):
        if "error" in results:
            return f"Error: {results['error']}"

        formatted_output = "Analysis Results:\n\n"

        # Google Search Query
        formatted_output += f"Google Search Query: {results['google_search_query']}\n\n"

        # Clustered News
        formatted_output += "Clustered News Topics:\n"
        for cluster, articles in results['clustered_news'].items():
            formatted_output += f"Cluster {cluster}:\n"
            for article in articles[:3]:  # Limit to 3 articles per cluster
                formatted_output += f"- {article['title']}\n"
            formatted_output += "\n"

        # Keyword Data
        formatted_output += "Keyword Analysis:\n"
        for key, values in results['keyword_data'].items():
            formatted_output += f"{key.replace('_', ' ').title()}:\n"
            for value in values[:5]:  # Limit to 5 items per category
                formatted_output += f"- {value}\n"
            formatted_output += "\n"

        # Advanced Analysis
        advanced = results['advanced_analysis']
        formatted_output += "Advanced Analysis:\n"
        if advanced['top_article']:
            formatted_output += f"Top Article: {advanced['top_article']['title']}\n"
        formatted_output += f"Summary: {advanced['summary']}\n"
        formatted_output += f"Sentiment: {advanced['sentiment']}\n\n"

        # GPT Analysis
        formatted_output += "GPT Analysis:\n"
        formatted_output += results['gpt_analysis']

        return formatted_output

if __name__ == '__main__':
    NewsAnalyzerApp().run()