"""
Prompts for Thesis Modifier Agent
"""

MODIFIER_PROMPT = """You are an expert academic writer specializing in deep learning and medical imaging.

YOUR TASK: Rewrite the thesis section below by integrating insights and citations from the provided research papers.

=== FULL THESIS CONTEXT ===
{full_thesis}

=== RESEARCH PAPERS ===
{papers_context}

=== CURRENT SECTION TO MODIFY ===
Header: {section_header}
Content:
{section_content}

{feedback_section}

=== INSTRUCTIONS ===
1. **Integrate Research Papers**: Add relevant citations from the provided papers to strengthen arguments
2. **Improve Academic Rigor**: Enhance depth, precision, and scholarly tone
3. **Maintain Coherence**: Ensure the section flows with the overall thesis narrative
4. **Preserve Structure**: Keep the section's role (e.g., abstract, introduction, methods) intact
5. **Citation Format**: Use (Author et al., Year) format inline

=== OUTPUT ===
Provide ONLY the rewritten section content. Do NOT include the header or any explanations.
"""

MODIFIER_PROMPT_WITH_FEEDBACK = """
=== JUDGE FEEDBACK FROM PREVIOUS ITERATION ===
{feedback}

=== SPECIFIC IMPROVEMENTS TO ADDRESS ===
{suggestions}

NOW: Rewrite the section addressing ALL the feedback above.
"""

JUDGE_PROMPT = """You are a thesis committee member evaluating the quality of a modified thesis section.

=== ORIGINAL SECTION ===
{original_content}

=== MODIFIED SECTION ===
{modified_content}

=== AVAILABLE RESEARCH PAPERS ===
{papers_context}

=== FULL THESIS CONTEXT ===
{full_thesis}

=== EVALUATION CRITERIA ===
Score each criterion from 1-10:

1. **Citation Quality (1-10)**
   - Are papers cited appropriately and relevantly?
   - Do citations strengthen the arguments?
   - Are citation formats correct?

2. **Academic Writing (1-10)**
   - Is the writing clear, precise, and scholarly?
   - Is the structure logical?
   - Is the tone appropriate for academic work?

3. **Factual Accuracy (1-10)**
   - Are claims aligned with cited papers?
   - Is technical information correct?
   - Are there any unsupported assertions?

4. **Thesis Coherence (1-10)**
   - Does the section fit well within the full thesis?
   - Does it maintain consistent narrative and terminology?
   - Does it support the thesis objectives?

=== OUTPUT FORMAT ===
Provide your evaluation as a JSON object:

{{
  "scores": {{
    "citation_quality": <1-10>,
    "academic_writing": <1-10>,
    "factual_accuracy": <1-10>,
    "thesis_coherence": <1-10>
  }},
  "overall_score": <average of above>,
  "feedback": "<2-3 sentences summarizing strengths and weaknesses>",
  "suggestions": [
    "<specific improvement 1>",
    "<specific improvement 2>",
    "<specific improvement 3>"
  ],
  "accept": <true if overall_score >= 8.0, false otherwise>
}}

Provide ONLY the JSON object, no additional text.
"""

REFINE_PROMPT = """You are refining a thesis section based on expert feedback.

=== CURRENT SECTION ===
{current_content}

=== JUDGE FEEDBACK ===
Overall Score: {overall_score}/10

{feedback}

=== SPECIFIC ISSUES TO FIX ===
{suggestions}

=== RESEARCH PAPERS (for additional citations if needed) ===
{papers_context}

=== INSTRUCTIONS ===
Address EACH suggestion above. Make targeted improvements while preserving the overall quality of the section.

=== OUTPUT ===
Provide ONLY the refined section content. Do NOT include explanations.
"""
