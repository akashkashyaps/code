import os
import re
import docx
from tqdm import tqdm
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage

class AgenticReportGrader:
    def __init__(self,
                 base_directory: str,
                 model: str = 'mistral:7b-instruct',
                 num_ctx: int = 2048,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 num_predict: int = 3000):
        self.base_directory = base_directory
        self.model = model
        self.model_name = model.replace(':', '_')  # Sanitize model name for filenames
        self.llm = ChatOllama(
            model=model,
            num_ctx=num_ctx,
            temperature=temperature,
            top_p=top_p,
            num_predict=num_predict
        )

    def llm_call(self, prompt: str) -> str:
        messages = [HumanMessage(content=prompt)]
        response = self.llm(messages)
        return response.content if hasattr(response, "content") else response

    def _extract_text_from_docx(self, file_path: str) -> str:
        doc = docx.Document(file_path)
        return "\n".join(para.text for para in doc.paragraphs if para.text.strip())

    def _parse_rubric_text(self, text: str):
        grading_criteria_pattern = re.compile(
            r'(Grading Criteria.*?)(?=\n\s*\d+\.\s.*?\(\d+%\)|$)',
            flags=re.IGNORECASE|re.DOTALL
        )
        match_gc = grading_criteria_pattern.search(text)
        grading_criteria_text = match_gc.group(1).strip() if match_gc else ""

        section_pattern = re.compile(
            r'(?P<header>(?P<section_number>\d+)\.\s*(?P<title>.*?\(\s*(?P<weight>\d+)\s*%\)))(?P<body>.*?)(?=\n\s*\d+\.\s.*?\(\d+%\)|$)',
            flags=re.DOTALL
        )

        sections = []
        for m in section_pattern.finditer(text):
            section_num = int(m.group('section_number'))
            header = m.group('header').strip()
            title_str = m.group('title').strip()
            weight_val = float(m.group('weight').strip())
            body = m.group('body').strip()

            sections.append({
                "section_number": section_num,
                "section_heading": header,
                "title": title_str,
                "weight": weight_val,
                "body_text": body
            })

        return {
            "grading_criteria": grading_criteria_text,
            "sections": sections
        }

    def _build_section_prompts(self, parsed_rubric) -> list:
        prompts = []
        for section in parsed_rubric["sections"]:
            section_prompt = f"""
Section {section['section_number']} – {section['section_heading']}

Full Rubric Subsection Text:
{section['body_text']}

Instructions:
1. Evaluate the student's work specifically on this section's criteria.
2. Provide a numeric or letter grade for this section.
3. Provide rationale in 1-2 short paragraphs.
"""
            prompts.append(section_prompt.strip())
        return prompts

    def _build_final_prompt(self, parsed_rubric, section_evaluations: list) -> str:
        sections_summary = []
        for s, ev in zip(parsed_rubric["sections"], section_evaluations):
            sections_summary.append(
                f"Section {s['section_number']} ({s['weight']}%): {s['title']}\n"
                f"Evaluation Summary:\n{ev}\n"
            )
        sections_text = "\n".join(sections_summary)

        final_prompt = f"""
You have evaluated each section individually. Now combine those results into a final score.

Grading Criteria Reference:
{parsed_rubric["grading_criteria"]}

Section Evaluations (with assigned weights):
{sections_text}

Instructions:
1. For each section, note the numeric grade you assigned (out of 100) or letter grade.
2. Convert each section’s grade into a numeric score, multiply by the weight (%).
3. Sum these weighted scores to get the final overall score.
4. Provide the final result, along with a rationale referencing how well the work met all criteria.
"""
        return final_prompt.strip()

    def evaluate_section(self, section_prompt: str, report_text: str, section_number: int) -> str:
        prompt = f"""
{section_prompt}

--- Student Report Below ---
{report_text}
"""
        return self.llm_call(prompt.strip())

    def grade_reports(self):
        for folder_name in tqdm(os.listdir(self.base_directory), desc="Processing"):
            folder_path = os.path.join(self.base_directory, folder_name)
            if not os.path.isdir(folder_path):
                continue

            report_files = [
                f for f in os.listdir(folder_path)
                if f.lower().startswith('report_') and f.endswith('.docx')
            ]
            if not report_files:
                continue
            report_path = os.path.join(folder_path, report_files[0])
            report_text = self._extract_text_from_docx(report_path)

            for prompt_num in range(1, 7):
                prompt_files = [
                    f for f in os.listdir(folder_path)
                    if f.lower() == f'prompt_{prompt_num}.docx'
                ]
                if not prompt_files:
                    continue
                prompt_path = os.path.join(folder_path, prompt_files[0])
                rubric_text = self._extract_text_from_docx(prompt_path)

                try:
                    parsed_rubric = self._parse_rubric_text(rubric_text)
                    section_prompts = self._build_section_prompts(parsed_rubric)
                    section_evaluations = []
                    for sp, s_info in zip(section_prompts, parsed_rubric["sections"]):
                        eval_text = self.evaluate_section(sp, report_text, s_info["section_number"])
                        section_evaluations.append(eval_text)

                    final_prompt = self._build_final_prompt(parsed_rubric, section_evaluations)
                    final_output = self.llm_call(final_prompt)

                    # Generate unique filename with model name
                    report_base = os.path.splitext(report_files[0])[0]
                    output_filename = f"GRADED_{report_base}_Prompt_{prompt_num}_{self.model_name}.docx"
                    output_path = os.path.join(folder_path, output_filename)
                    doc = docx.Document()
                    doc.add_paragraph(final_output)
                    doc.save(output_path)

                except Exception as e:
                    print(f"Error processing {folder_name}/Prompt_{prompt_num}: {e}")

def main():
    base_directory = '/home/akash/Downloads/grading_documents'
    models = {
        'qwen2.5:7b-instruct-q4_0': 32768,
        'deepseek-r1:7b-qwen-distill-q4_K_M': 32768,
        'deepseek-r1:8b-llama-distill-q4_K_M': 32768
    }

    for model, ctx in models.items():
        print(f"\nStarting grading with {model}")
        grader = AgenticReportGrader(
            base_directory=base_directory,
            model=model,
            num_ctx=ctx,
            temperature=0.1,
            top_p=0.9,
            num_predict=3000
        )
        grader.grade_reports()
        print(f"Completed grading with {model}")

if __name__ == '__main__':
    main()