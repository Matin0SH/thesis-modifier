"""
PDF Fetch Agent - Reflexion-based Paper Retrieval
Uses Gemini 2.5 Flash with Chain-of-Thought and self-reflection to find relevant papers
"""

import os
import json
import logging
from typing import List, Dict, Tuple
import google.generativeai as genai
from dotenv import load_dotenv
import requests
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment
load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))


class PaperSearchTool:
    """Search academic papers via Semantic Scholar and arXiv APIs"""

    SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    ARXIV_URL = "http://export.arxiv.org/api/query"

    def search(self, query: str, max_results: int = 20) -> List[Dict]:
        """Search both Semantic Scholar and arXiv"""
        papers = []

        # Semantic Scholar
        try:
            papers.extend(self._search_semantic_scholar(query, max_results // 2))
        except Exception as e:
            logger.warning(f"Semantic Scholar search failed: {e}")

        # arXiv
        try:
            papers.extend(self._search_arxiv(query, max_results // 2))
        except Exception as e:
            logger.warning(f"arXiv search failed: {e}")

        return papers

    def _search_semantic_scholar(self, query: str, limit: int) -> List[Dict]:
        """Search Semantic Scholar API"""
        params = {
            'query': query,
            'limit': limit,
            'fields': 'title,authors,year,citationCount,abstract,openAccessPdf,externalIds,url'
        }

        time.sleep(2)  # Rate limiting: 2 second delay
        response = requests.get(self.SEMANTIC_SCHOLAR_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        papers = []
        for paper in data.get('data', []):
            papers.append({
                'title': paper.get('title', 'N/A'),
                'authors': [a.get('name', 'Unknown') for a in paper.get('authors', [])],
                'year': paper.get('year'),
                'citations': paper.get('citationCount', 0),
                'abstract': paper.get('abstract', 'N/A'),
                'pdf_url': paper.get('openAccessPdf', {}).get('url') if paper.get('openAccessPdf') else None,
                'doi': paper.get('externalIds', {}).get('DOI'),
                'url': paper.get('url'),
                'source': 'Semantic Scholar'
            })

        return papers

    def _search_arxiv(self, query: str, limit: int) -> List[Dict]:
        """Search arXiv API"""
        params = {
            'search_query': f'all:{query}',
            'start': 0,
            'max_results': limit,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }

        time.sleep(1)  # Rate limiting: 1 second delay
        response = requests.get(self.ARXIV_URL, params=params, timeout=10)
        response.raise_for_status()

        # Parse XML response
        import xml.etree.ElementTree as ET
        root = ET.fromstring(response.content)

        papers = []
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip()
            authors = [a.find('{http://www.w3.org/2005/Atom}name').text
                      for a in entry.findall('{http://www.w3.org/2005/Atom}author')]
            published = entry.find('{http://www.w3.org/2005/Atom}published').text[:4]
            abstract = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip()
            pdf_url = entry.find('{http://www.w3.org/2005/Atom}id').text.replace('abs', 'pdf') + '.pdf'

            papers.append({
                'title': title,
                'authors': authors,
                'year': int(published),
                'citations': 0,  # arXiv doesn't provide citation count
                'abstract': abstract,
                'pdf_url': pdf_url,
                'doi': None,
                'url': entry.find('{http://www.w3.org/2005/Atom}id').text,
                'source': 'arXiv'
            })

        return papers


class ResultValidator:
    """Validate search results quality"""

    def validate(self, query: str, papers: List[Dict], thesis_keywords: List[str]) -> Tuple[bool, str]:
        """
        Validate paper search results
        Returns: (is_valid, feedback_message)
        """
        if len(papers) == 0:
            return False, "No results found. Query too specific. Try broader keywords."

        if len(papers) > 50:
            return False, "Too many results (>50). Add more specific terms."

        # Check relevance (basic keyword matching)
        relevant_count = sum(1 for p in papers if self._is_relevant(p, thesis_keywords))
        relevance_ratio = relevant_count / len(papers)

        if relevance_ratio < 0.3:
            return False, f"Low relevance ({relevance_ratio:.0%}). Try different keywords related to thesis topic."

        # Check open access availability
        open_access_count = sum(1 for p in papers if p.get('pdf_url'))
        if open_access_count == 0:
            return False, "No open access papers found. Add 'open access' or try arXiv-specific terms."

        # Check recency
        recent_count = sum(1 for p in papers if p.get('year') and p['year'] >= 2019)
        if recent_count < len(papers) * 0.3:
            return False, "Most papers are old. Add year filter or recent terminology."

        return True, f"Good results: {len(papers)} papers, {relevance_ratio:.0%} relevant, {open_access_count} open access"

    def _is_relevant(self, paper: Dict, keywords: List[str]) -> bool:
        """Check if paper is relevant based on keywords"""
        title = paper.get('title') or ''
        abstract = paper.get('abstract') or ''
        text = (title + ' ' + abstract).lower()

        # Core technical terms that should appear
        core_terms = ['deep learning', 'neural', 'cnn', 'classification', 'medical',
                      'imaging', 'detection', 'attention', 'transformer', 'implant']

        # Check both keywords and core terms
        keyword_matches = sum(1 for kw in keywords if kw.lower() in text)
        core_matches = sum(1 for term in core_terms if term in text)

        return keyword_matches >= 1 or core_matches >= 2  # More lenient


class GeminiQueryGenerator:
    """Generate and refine search queries using Gemini with CoT"""

    def __init__(self, model_name: str = "gemini-2.0-flash-exp"):
        self.model = genai.GenerativeModel(model_name)

    def generate_initial_queries(self, thesis_content: str) -> List[str]:
        """Generate initial search queries from thesis"""
        prompt = f"""You are an expert research librarian. Analyze this thesis section and generate 8-12 search queries to find relevant academic papers.

Thesis Content:
{thesis_content[:3000]}

Think step by step:
1. Identify the main research topic and methodology
2. Extract key technical terms and concepts
3. Identify research gaps mentioned
4. Consider related work that would be valuable

Generate 8-12 concise search queries (each 3-6 words) that would find:
- Foundational papers on the main technique
- Recent advances in the specific application
- Comparative studies and benchmarks
- Related methodologies

Output ONLY the queries, one per line, no numbering or explanation."""

        response = self.model.generate_content(prompt)
        queries = [q.strip() for q in response.text.strip().split('\n') if q.strip()]
        logger.info(f"Generated {len(queries)} initial queries")
        return queries

    def refine_query(self, original_query: str, feedback: str, thesis_keywords: List[str]) -> str:
        """Refine query based on validation feedback"""
        prompt = f"""You are refining a search query that didn't produce good results.

Original Query: {original_query}
Problem: {feedback}
Thesis Keywords: {', '.join(thesis_keywords)}

Think step by step:
1. Analyze why the query failed
2. Consider alternative phrasings or terms
3. Adjust specificity level

Generate ONE improved search query (3-6 words). Output ONLY the query, nothing else."""

        response = self.model.generate_content(prompt)
        refined = response.text.strip().split('\n')[0]
        logger.info(f"Refined: '{original_query}' → '{refined}'")
        return refined


class PDFDownloader:
    """Download PDFs from URLs"""

    def __init__(self, download_dir: str = "downloaded_papers"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)

    def download(self, url: str, filename: str) -> bool:
        """
        Download PDF from URL
        Returns True if successful, False otherwise
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            response = requests.get(url, headers=headers, stream=True, timeout=30)
            response.raise_for_status()

            # Check if it's actually a PDF
            content_type = response.headers.get('content-type', '')
            if 'pdf' not in content_type.lower() and not url.endswith('.pdf'):
                logger.warning(f"URL may not be PDF: {content_type}")
                return False

            filepath = self.download_dir / filename
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"✓ Downloaded: {filename}")
            return True

        except Exception as e:
            logger.warning(f"✗ Download failed: {filename} - {e}")
            return False

    def download_papers(self, papers: List[Dict], max_downloads: int = 10) -> int:
        """
        Download PDFs for papers with available URLs
        Returns count of successful downloads
        """
        downloaded = 0

        for i, paper in enumerate(papers):
            if downloaded >= max_downloads:
                break

            pdf_url = paper.get('pdf_url')
            if not pdf_url:
                continue

            # Generate safe filename
            title = paper.get('title', f'paper_{i}')
            safe_title = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in title)
            safe_title = safe_title[:100]  # Limit length
            filename = f"{safe_title}.pdf"

            if self.download(pdf_url, filename):
                downloaded += 1
                time.sleep(1)  # Rate limiting

        return downloaded


class ReflexionPaperFetcher:
    """Main agent with reflexion loop"""

    def __init__(self):
        self.query_gen = GeminiQueryGenerator()
        self.search_tool = PaperSearchTool()
        self.validator = ResultValidator()
        self.downloader = PDFDownloader()

    def fetch_papers(self, thesis_sections: List[Dict], max_attempts: int = 3) -> Dict:
        """
        Fetch papers with reflexion-based query refinement

        Args:
            thesis_sections: List of thesis section dicts from JSONL
            max_attempts: Max refinement attempts per query

        Returns:
            Dict with queries and their paper results
        """
        # Combine thesis content and extract keywords
        full_content = '\n\n'.join([
            f"# {s['header']}\n{s['content'][:1000]}"
            for s in thesis_sections[:3]  # Use first 3 sections
        ])

        keywords = self._extract_keywords(full_content)
        logger.info(f"Extracted keywords: {', '.join(keywords[:10])}")

        # Generate initial queries
        logger.info("Generating initial queries with Gemini CoT...")
        queries = self.query_gen.generate_initial_queries(full_content)

        # Fetch papers with reflexion
        results = {}
        for i, query in enumerate(queries[:12], 1):  # Limit to 12 queries
            logger.info(f"\n[{i}/{min(len(queries), 12)}] Processing: '{query}'")

            papers = None
            current_query = query

            for attempt in range(max_attempts):
                # Search
                papers = self.search_tool.search(current_query)
                logger.info(f"  Attempt {attempt+1}: Found {len(papers)} papers")

                # Validate
                is_valid, feedback = self.validator.validate(current_query, papers, keywords)

                if is_valid:
                    logger.info(f"  ✓ {feedback}")
                    break
                else:
                    logger.warning(f"  ✗ {feedback}")
                    if attempt < max_attempts - 1:
                        # Refine and retry
                        time.sleep(1)  # Rate limiting
                        current_query = self.query_gen.refine_query(current_query, feedback, keywords)

            results[current_query] = {
                'original_query': query,
                'final_query': current_query,
                'attempts': attempt + 1,
                'papers': papers,
                'paper_count': len(papers),
                'open_access_count': sum(1 for p in papers if p.get('pdf_url'))
            }

            time.sleep(3)  # Rate limiting between queries (increased)

        return results

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract key terms from thesis (simple approach)"""
        # Common technical terms to look for
        keywords = set()
        words = text.lower().split()

        # Look for capitalized terms and common patterns
        for i, word in enumerate(words):
            if len(word) > 4 and (word.isupper() or word.istitle()):
                keywords.add(word.lower())
            # Bigrams for technical terms
            if i < len(words) - 1:
                bigram = f"{words[i]} {words[i+1]}"
                if any(term in bigram for term in ['deep learning', 'neural', 'classification',
                                                     'detection', 'attention', 'transformer']):
                    keywords.add(bigram)

        return list(keywords)[:20]

    def save_results(self, results: Dict, output_path: str):
        """Save results to JSON"""
        # Prepare summary
        summary = {
            'total_queries': len(results),
            'total_papers': sum(r['paper_count'] for r in results.values()),
            'total_open_access': sum(r['open_access_count'] for r in results.values()),
            'queries': []
        }

        for query, data in results.items():
            summary['queries'].append({
                'original_query': data['original_query'],
                'final_query': data['final_query'],
                'attempts': data['attempts'],
                'paper_count': data['paper_count'],
                'open_access_count': data['open_access_count'],
                'papers': data['papers']
            })

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"\n✓ Saved results to {output_path}")
        logger.info(f"  Total: {summary['total_papers']} papers, {summary['total_open_access']} open access")


def main():
    """Main execution"""
    logger.info("=== PDF Fetch Agent - Reflexion Mode ===\n")

    # Load thesis sections
    jsonl_path = "thesis/thesis_sections.jsonl"
    logger.info(f"Loading thesis from {jsonl_path}")

    sections = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            sections.append(json.loads(line))

    logger.info(f"Loaded {len(sections)} sections\n")

    # Fetch papers
    agent = ReflexionPaperFetcher()
    results = agent.fetch_papers(sections)

    # Save results
    output_path = "paper_search_results.json"
    agent.save_results(results, output_path)

    # Download PDFs
    logger.info("\n=== Downloading PDFs ===")
    all_papers = []
    for query_data in results.values():
        all_papers.extend(query_data['papers'])

    # Download up to 20 open access papers
    downloaded_count = agent.downloader.download_papers(all_papers, max_downloads=20)
    logger.info(f"\n✓ Downloaded {downloaded_count} PDFs to 'downloaded_papers/' folder")

    logger.info("\n=== Complete ===")


if __name__ == "__main__":
    main()
