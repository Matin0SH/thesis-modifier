"""
View Reasoning Logs - Display judge reasoning and suggestions
"""

import json
from pathlib import Path


def display_reasoning_logs():
    """Display detailed reasoning from iteration logs"""

    log_file = Path("iteration_logs.json")

    if not log_file.exists():
        print("❌ No iteration_logs.json found. Run thesis_modifier_agent.py first.")
        return

    with open(log_file, 'r', encoding='utf-8') as f:
        logs = json.load(f)

    print("\n" + "="*80)
    print("THESIS MODIFICATION REASONING LOGS")
    print("="*80)

    for section_log in logs:
        section_num = section_log['section_number']
        header = section_log['header']

        print(f"\n{'='*80}")
        print(f"SECTION {section_num}: {header}")
        print(f"{'='*80}")

        for iter_log in section_log['iterations']:
            iteration = iter_log['iteration']
            scores = iter_log['scores']
            overall = iter_log['overall']
            feedback = iter_log['feedback']
            suggestions = iter_log.get('suggestions', [])

            print(f"\n--- ITERATION {iteration} ---")
            print(f"\nSCORES:")
            print(f"  Citation Quality:    {scores['citation_quality']}/10")
            print(f"  Academic Writing:    {scores['academic_writing']}/10")
            print(f"  Factual Accuracy:    {scores['factual_accuracy']}/10")
            print(f"  Thesis Coherence:    {scores['thesis_coherence']}/10")
            print(f"  -----------------------------")
            print(f"  OVERALL SCORE:       {overall:.1f}/10")

            print(f"\nJUDGE FEEDBACK:")
            print(f"  {feedback}")

            if suggestions:
                print(f"\nSUGGESTIONS FOR IMPROVEMENT:")
                for i, suggestion in enumerate(suggestions, 1):
                    print(f"  {i}. {suggestion}")

            if 'improvement' in iter_log:
                improvement = iter_log['improvement']
                symbol = "^" if improvement > 0 else "v" if improvement < 0 else "-"
                print(f"\nIMPROVEMENT: {symbol} {improvement:+.1f} points")

        print(f"\n{'-'*80}")

    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")

    total_sections = len(logs)
    avg_iter1 = sum(log['iterations'][0]['overall'] for log in logs) / total_sections
    avg_iter2 = sum(log['iterations'][1]['overall'] for log in logs) / total_sections
    avg_improvement = avg_iter2 - avg_iter1

    print(f"\nTotal Sections Processed: {total_sections}")
    print(f"Average Score (Iteration 1): {avg_iter1:.2f}/10")
    print(f"Average Score (Iteration 2): {avg_iter2:.2f}/10")
    print(f"Average Improvement: {avg_improvement:+.2f} points")

    # Score distribution
    print(f"\nSCORE BREAKDOWN (Final Iteration):")
    score_ranges = {
        '9.0-10.0 (Excellent)': 0,
        '8.0-8.9 (Good)': 0,
        '7.0-7.9 (Satisfactory)': 0,
        '<7.0 (Needs Work)': 0
    }

    for log in logs:
        final_score = log['iterations'][1]['overall']
        if final_score >= 9.0:
            score_ranges['9.0-10.0 (Excellent)'] += 1
        elif final_score >= 8.0:
            score_ranges['8.0-8.9 (Good)'] += 1
        elif final_score >= 7.0:
            score_ranges['7.0-7.9 (Satisfactory)'] += 1
        else:
            score_ranges['<7.0 (Needs Work)'] += 1

    for range_name, count in score_ranges.items():
        percentage = (count / total_sections) * 100
        bar = '#' * int(percentage / 5)
        print(f"  {range_name:25} {count:2}/{total_sections:2} {bar} {percentage:.0f}%")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    display_reasoning_logs()
