"""
Thesis Modifier Agent - Iterative Section Improvement
"""

import os
import json
import logging
import time
from typing import Dict, List, Tuple
from dotenv import load_dotenv
import google.generativeai as genai
from prompts import MODIFIER_PROMPT, MODIFIER_PROMPT_WITH_FEEDBACK, JUDGE_PROMPT, REFINE_PROMPT

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

load_dotenv()
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))


class ThesisModifierAgent:
    """Agent that modifies thesis sections using research papers"""

    def __init__(self, model_name: str = "gemini-2.0-flash-exp"):
        self.model = genai.GenerativeModel(model_name)

    def modify_section(self, full_thesis: str, papers: str, section_header: str,
                       section_content: str, feedback: Dict = None) -> str:
        """Modify a thesis section, optionally using judge feedback"""

        if feedback:
            # Iteration 2: Use refinement with feedback
            feedback_text = MODIFIER_PROMPT_WITH_FEEDBACK.format(
                feedback=feedback.get('feedback', ''),
                suggestions='\n'.join(f"- {s}" for s in feedback.get('suggestions', []))
            )
        else:
            # Iteration 1: Initial modification
            feedback_text = ""

        prompt = MODIFIER_PROMPT.format(
            full_thesis=full_thesis[:8000],  # Truncate for context limit
            papers_context=papers[:15000],
            section_header=section_header,
            section_content=section_content,
            feedback_section=feedback_text
        )

        # Rate limiting: wait 7 seconds between requests (max 10/min)
        time.sleep(7)
        response = self.model.generate_content(prompt)
        return response.text.strip()


class JudgeAgent:
    """Agent that evaluates modified sections"""

    def __init__(self, model_name: str = "gemini-2.0-flash-exp"):
        self.model = genai.GenerativeModel(
            model_name,
            generation_config={"response_mime_type": "application/json"}
        )

    def evaluate(self, original: str, modified: str, papers: str, full_thesis: str) -> Dict:
        """Evaluate a modified section and return scores + feedback"""

        prompt = JUDGE_PROMPT.format(
            original_content=original[:3000],
            modified_content=modified[:3000],
            papers_context=papers[:10000],
            full_thesis=full_thesis[:5000]
        )

        # Rate limiting: wait 7 seconds between requests
        time.sleep(7)
        response = self.model.generate_content(prompt)
        result = json.loads(response.text)

        logger.info(f"  Judge Score: {result['overall_score']:.1f}/10")
        return result


class ThesisRefiner:
    """Orchestrator for thesis modification process"""

    def __init__(self):
        self.modifier = ThesisModifierAgent()
        self.judge = JudgeAgent()
        self.thesis_sections = []
        self.papers = []
        self.iteration_logs = []

    def load_data(self):
        """Load thesis sections and papers"""
        logger.info("Loading thesis sections...")
        with open("thesis/thesis_sections.jsonl", 'r', encoding='utf-8') as f:
            self.thesis_sections = [json.loads(line) for line in f]
        logger.info(f"  ✓ Loaded {len(self.thesis_sections)} sections")

        logger.info("Loading research papers...")
        with open("parsed_papers.jsonl", 'r', encoding='utf-8') as f:
            self.papers = [json.loads(line) for line in f]
        logger.info(f"  ✓ Loaded {len(self.papers)} papers")

    def get_full_thesis_context(self) -> str:
        """Get full thesis as context string"""
        context = []
        for sec in self.thesis_sections:
            context.append(f"# {sec['header']}\n{sec['content'][:500]}...\n")
        return '\n'.join(context)

    def get_papers_context(self) -> str:
        """Get papers as context string"""
        context = []
        for i, paper in enumerate(self.papers[:5], 1):  # Limit to 5 papers for context
            context.append(f"[Paper {i}] {paper['title']}\n{paper['content'][:2000]}...\n")
        return '\n'.join(context)

    def process_section(self, section_idx: int) -> Dict:
        """Process one section with 2 iterations"""
        section = self.thesis_sections[section_idx]
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing Section {section_idx}: {section['header']}")
        logger.info(f"{'='*70}")

        full_thesis = self.get_full_thesis_context()
        papers_context = self.get_papers_context()

        original_content = section['content']
        section_log = {
            'section_number': section_idx,
            'header': section['header'],
            'iterations': []
        }

        # Iteration 1
        logger.info("\n[Iteration 1] Modifying section...")
        modified_1 = self.modifier.modify_section(
            full_thesis, papers_context, section['header'], original_content
        )

        logger.info("[Iteration 1] Judge evaluating...")
        judge_1 = self.judge.evaluate(original_content, modified_1, papers_context, full_thesis)

        section_log['iterations'].append({
            'iteration': 1,
            'scores': judge_1['scores'],
            'overall': judge_1['overall_score'],
            'feedback': judge_1['feedback'],
            'suggestions': judge_1.get('suggestions', [])
        })

        # Iteration 2
        logger.info("\n[Iteration 2] Refining based on feedback...")
        modified_2 = self.modifier.modify_section(
            full_thesis, papers_context, section['header'], modified_1, feedback=judge_1
        )

        logger.info("[Iteration 2] Judge re-evaluating...")
        judge_2 = self.judge.evaluate(original_content, modified_2, papers_context, full_thesis)

        section_log['iterations'].append({
            'iteration': 2,
            'scores': judge_2['scores'],
            'overall': judge_2['overall_score'],
            'feedback': judge_2['feedback'],
            'suggestions': judge_2.get('suggestions', []),
            'improvement': judge_2['overall_score'] - judge_1['overall_score']
        })

        # Update section in thesis
        self.thesis_sections[section_idx]['content'] = modified_2
        self.thesis_sections[section_idx]['modified'] = True

        # Save updated thesis
        self.save_thesis()

        # Log results
        self.iteration_logs.append(section_log)
        self.save_logs()

        logger.info(f"\n✓ Section {section_idx} complete:")
        logger.info(f"  Iteration 1: {judge_1['overall_score']:.1f}/10")
        logger.info(f"  Iteration 2: {judge_2['overall_score']:.1f}/10")
        logger.info(f"  Improvement: +{judge_2['overall_score'] - judge_1['overall_score']:.1f}")

        return section_log

    def save_thesis(self):
        """Save updated thesis sections to JSONL"""
        with open("modified_thesis_sections.jsonl", 'w', encoding='utf-8') as f:
            for section in self.thesis_sections:
                f.write(json.dumps(section, ensure_ascii=False) + '\n')

    def save_logs(self):
        """Save iteration logs"""
        with open("iteration_logs.json", 'w', encoding='utf-8') as f:
            json.dump(self.iteration_logs, f, indent=2, ensure_ascii=False)

    def process_all_sections(self, start_from: int = 0):
        """Process all sections sequentially with checkpoint recovery"""
        total = len(self.thesis_sections)

        for i in range(start_from, total):
            try:
                logger.info(f"\n{'='*70}")
                logger.info(f"Progress: {i+1}/{total} sections")
                logger.info(f"{'='*70}")

                self.process_section(i)

            except Exception as e:
                logger.error(f"\n✗ Error processing section {i}: {e}")
                logger.info(f"\nCheckpoint: Resume from section {i} by running:")
                logger.info(f"  refiner.process_all_sections(start_from={i})")
                raise

        logger.info("\n" + "="*70)
        logger.info("ALL SECTIONS COMPLETE")
        logger.info("="*70)


def main():
    """Process ALL thesis sections"""
    logger.info("=== Thesis Modifier Agent - Full Run ===\n")

    refiner = ThesisRefiner()
    refiner.load_data()

    # Process ALL sections
    logger.info(f"\n=== Processing {len(refiner.thesis_sections)} Sections ===")
    refiner.process_all_sections()

    # Generate summary report
    logger.info("\n" + "="*70)
    logger.info("FINAL SUMMARY")
    logger.info("="*70)

    for log in refiner.iteration_logs:
        logger.info(f"\n{log['header']}:")
        logger.info(f"  Iter 1: {log['iterations'][0]['overall']:.1f}/10")
        logger.info(f"  Iter 2: {log['iterations'][1]['overall']:.1f}/10")
        logger.info(f"  Gain: +{log['iterations'][1]['improvement']:.1f}")

    avg_final = sum(log['iterations'][1]['overall'] for log in refiner.iteration_logs) / len(refiner.iteration_logs)
    logger.info(f"\n✓ Average Final Score: {avg_final:.1f}/10")
    logger.info("\n=== Complete ===")
    logger.info("Results: modified_thesis_sections.jsonl")
    logger.info("Logs: iteration_logs.json")


if __name__ == "__main__":
    main()
