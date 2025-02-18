import os
import json
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
                 top_p: float = 0.9):
        self.base_directory = base_directory
        self.llm = ChatOllama(
            model=model,
            num_ctx=num_ctx,
            temperature=temperature,
            top_p=top_p
        )

    def llm_call(self, prompt: str) -> str:
        messages = [HumanMessage(content=prompt)]
        response = self.llm(messages)
        if hasattr(response, "content"):
            return response.content
        return response

    def clean_response(self, response: str) -> str:
        response = response.strip()
        response = re.sub(r"^```(json)?\s*", "", response)
        response = re.sub(r"\s*```$", "", response)
        return response.strip()

    def _extract_text_from_docx(self, file_path: str) -> str:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs if para.text])

    def generate_evaluation_prompts(self, grading_prompt: str) -> dict:
        """
        Decompose the grading rubric into two parts:
          1. A JSON array of section-specific evaluation prompts.
          2. A final evaluation prompt that instructs the evaluator to:
             - Use the rubric exactly (without adding new criteria),
             - Include each section's grade and justification,
        Output a JSON object with keys "section_prompts" and "final_prompt".
        """
        prompt = f"""You are a Prompt Engineering Expert with extensive experience in educational assessment design.
Analyze and decompose the following grading rubric into two parts:
1. Generate a JSON array of section-specific evaluation prompts that can be used to evaluate distinct sections of a student report. All of the section prompts should be detail;ed with every instructions on how to evaluate that section and how to grade that section according to the grading rubrics.
2. Create a final evaluation prompt for a Senior Grading Coordinator that instructs them to compile a final grading report.
   The final prompt must require:
     - Strict use of the provided rubric (do not introduce new criteria),
     - Inclusion of each section's grade along with justification,
Output the result as a valid JSON object with the following structure:
{{
  "section_prompts": [ "Section prompt 1", "Section prompt 2", ... ],
  "final_prompt": [{"Final evaluation prompt string"}]
}}

Grading Rubric:
{grading_prompt}
"""
        response = self.llm_call(prompt)
        cleaned_response = self.clean_response(response)
        try:
            evaluation_prompts = json.loads(cleaned_response)
        except Exception as e:
            raise ValueError(f"Failed to parse evaluation prompts from LLM response: {e}\nResponse was: {cleaned_response}")
        return evaluation_prompts

    def evaluate_section(self, section_prompt: str, report_text: str, section_index: int) -> str:
        prompt = f"""You are a Grading Section Specialist with meticulous attention to detail.
Evaluate the following student report based on the criteria for Section {section_index}.

Section {section_index} Criteria:
{section_prompt}

Student Report:
{report_text}

Provide a detailed evaluation for this section including a grade (numeric or letter), score, and rationale in Markdown format.
"""
        return self.llm_call(prompt)

    def final_evaluation(self, final_prompt: str, section_evaluations: list) -> str:
        """
        Combine the final prompt with the section evaluations and generate the final report.
        """
        combined_evaluations = "\n\n".join(section_evaluations)
        full_prompt = f"{final_prompt}\n\nSection Evaluations:\n{combined_evaluations}"
        return self.llm_call(full_prompt)

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
                    # Step 1: Generate both section-specific prompts and a final evaluation prompt from the rubric.
                    evaluation_prompts = self.generate_evaluation_prompts(grading_prompt_text)
                    section_prompts = evaluation_prompts.get("section_prompts", [])
                    final_prompt = evaluation_prompts.get("final_prompt", "")

                    # Step 2: Evaluate each section using the corresponding prompt.
                    section_evaluations = []
                    for idx, section_prompt in enumerate(section_prompts, start=1):
                        evaluation = self.evaluate_section(section_prompt, report_text, idx)
                        section_evaluations.append(evaluation)

                    # Step 3: Synthesize the final evaluation using the provided final prompt.
                    final_output = self.final_evaluation(final_prompt, section_evaluations)

                    # Save final output to a DOCX file.
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
        'qwen2.5:7b-instruct-q4_0': 32768
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
