import os
import json
import docx
from tqdm import tqdm
from langchain_ollama import ChatOllama

class AgenticReportGrader:
    def __init__(
        self,
        base_directory: str,
        model: str = 'mistral:7b-instruct',
        num_ctx: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9
    ):
        self.base_directory = base_directory
        self.llm = ChatOllama(
            model=model,
            num_ctx=num_ctx,
            temperature=temperature,
            top_p=top_p
        )

    def _extract_text_from_docx(self, file_path: str) -> str:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs if para.text])

    def generate_section_prompts(self, grading_prompt: str) -> list:
        """
        Uses the LLM to decompose the overall grading prompt into a JSON array
        of section-specific evaluation prompts.
        """
        prompt = f"""You are a Prompt Engineering Expert with extensive experience in educational assessment design.
Analyze and decompose the following grading prompt into distinct evaluation sections.
For each section, create a specific prompt that can be used to evaluate that section.
Grading Prompt:
{grading_prompt}

Format your output as a JSON array of strings, where each string is a section-specific evaluation prompt.
"""
        response = self.llm(prompt)
        try:
            section_prompts = json.loads(response)
        except Exception as e:
            raise ValueError(f"Failed to parse section prompts from LLM response: {e}\nResponse was: {response}")
        return section_prompts

    def evaluate_section(self, section_prompt: str, report_text: str, section_index: int) -> str:
        """
        Uses the LLM to evaluate a section of the student report based on the provided criteria.
        """
        prompt = f"""You are a Grading Section Specialist, a subject matter expert with meticulous attention to detail.
Evaluate the following student report based on these section criteria.

Section {section_index} Criteria:
{section_prompt}

Student Report:
{report_text}

Provide a detailed evaluation including a score and rationale in Markdown format.
"""
        return self.llm(prompt)

    def final_evaluation(self, section_evaluations: list) -> str:
        """
        Uses the LLM to synthesize the individual section evaluations into a final grade and overall feedback.
        """
        combined_evaluations = "\n\n".join(section_evaluations)
        prompt = f"""You are a Senior Grading Coordinator with extensive experience in holistic assessment.
Based on the following section evaluations, compile a final grade for the student report.
Include overall feedback, suggestions for improvement, and justify the final letter grade (A-F).

Section Evaluations:
{combined_evaluations}
"""
        return self.llm(prompt)

    def grade_reports(self):
        for folder_name in tqdm(os.listdir(self.base_directory), desc="Processing"):
            folder_path = os.path.join(self.base_directory, folder_name)
            if not os.path.isdir(folder_path):
                continue

            # Process report file (assumes a file starting with 'report_' in the folder)
            report_files = [f for f in os.listdir(folder_path) if f.lower().startswith('report_') and f.endswith('.docx')]
            if not report_files:
                continue
            report_path = os.path.join(folder_path, report_files[0])
            report_text = self._extract_text_from_docx(report_path)

            # Process prompt files for prompt_1.docx to prompt_6.docx
            for prompt_num in range(1, 7):
                prompt_files = [f for f in os.listdir(folder_path) if f.lower() == f'prompt_{prompt_num}.docx']
                if not prompt_files:
                    continue
                prompt_path = os.path.join(folder_path, prompt_files[0])
                grading_prompt_text = self._extract_text_from_docx(prompt_path)

                try:
                    # Step 1: Generate section-specific prompts from the overall grading prompt
                    section_prompts = self.generate_section_prompts(grading_prompt_text)

                    # Step 2: Evaluate each section using the corresponding prompt
                    section_evaluations = []
                    for idx, section_prompt in enumerate(section_prompts, start=1):
                        evaluation = self.evaluate_section(section_prompt, report_text, idx)
                        section_evaluations.append(evaluation)
                    
                    # Step 3: Synthesize the final evaluation from all section evaluations
                    final_output = self.final_evaluation(section_evaluations)

                    # Save final output to a DOCX file
                    output_filename = f'GRADED_{report_files[0].replace(".docx", "")}_Prompt_{prompt_num}.docx'
                    output_path = os.path.join(folder_path, output_filename)
                    
                    doc = docx.Document()
                    doc.add_paragraph(final_output)
                    doc.save(output_path)

                except Exception as e:
                    print(f"Error processing {folder_name}/Prompt_{prompt_num}: {e}")

def main():
    base_directory = '/home/akash/Downloads/grading_documents'
    models = {
        'ollama/qwen2.5:7b-instruct-q4_0': 32768
    }

    for model, ctx in models.items():
        print(f"\nStarting grading with {model}")
        grader = AgenticReportGrader(
            base_directory,
            model=model,
            num_ctx=ctx,
            temperature=0.3,
            top_p=0.9
        )
        grader.grade_reports()
        print(f"Completed grading with {model}")

if __name__ == '__main__':
    main()
