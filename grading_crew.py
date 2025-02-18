import os
import json
import docx
from tqdm import tqdm
from crewai import Agent, Task, Crew
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
        return '\n'.join([para.text for para in doc.paragraphs if para.text])

    def _create_agents(self):
        """Create CrewAI agents with roles and configurations"""
        self.prompt_gen_agent = Agent(
            role='Prompt Engineering Expert',
            goal='Break down grading prompts into section-specific evaluation criteria',
            backstory='Expert in educational assessment design and prompt engineering',
            llm=self.llm,
            verbose=True
        )

        self.section_eval_agent = Agent(
            role='Grading Section Specialist',
            goal='Thoroughly evaluate student submissions against specific section criteria',
            backstory='Subject matter expert with meticulous attention to detail',
            llm=self.llm,
            verbose=True
        )

        self.final_eval_agent = Agent(
            role='Senior Grading Coordinator',
            goal='Synthesize section evaluations into final grade with comprehensive feedback',
            backstory='Experienced educator with holistic evaluation expertise',
            llm=self.llm,
            verbose=True
        )

    def _create_workflow(self, prompt_text: str, report_text: str):
        """Create dynamic workflow tasks"""
        prompt_gen_task = Task(
            description=f"""Analyze and decompose this grading prompt:
            {prompt_text}
            Identify distinct evaluation sections and create specific prompts for each section.
            Format output as a JSON array of strings.""",
            agent=self.prompt_gen_agent,
            expected_output="JSON array of section evaluation prompts"
        )

        # Create temporary crew for prompt generation
        prompt_crew = Crew(
            agents=[self.prompt_gen_agent],
            tasks=[prompt_gen_task],
            verbose=2
        )
        section_prompts = json.loads(prompt_crew.kickoff())

        # Create section evaluation tasks
        section_tasks = []
        for idx, section_prompt in enumerate(section_prompts):
            task = Task(
                description=f"""Evaluate student report against this section criteria:
                {section_prompt}
                Student Report Content:
                {report_text}
                Provide detailed evaluation with score and rationale.""",
                agent=self.section_eval_agent,
                expected_output=f"Markdown evaluation for section {idx+1}",
                output_file=f"temp_section_{idx+1}.md"
            )
            section_tasks.append(task)

        # Final evaluation task
        final_task = Task(
            description="""Compile section evaluations into final grade.
            Consider all section evaluations and provide final assessment.
            Include overall feedback and suggestions for improvement.
            Format final grade as letter grade (A-F) with justification.""",
            agent=self.final_eval_agent,
            expected_output="Final report with grade and comprehensive feedback",
            context=section_tasks,
            output_file="final_grade.docx"
        )

        return Crew(
            agents=[self.section_eval_agent, self.final_eval_agent],
            tasks=section_tasks + [final_task],
            verbose=2
        )

    def grade_reports(self):
        for folder_name in tqdm(os.listdir(self.base_directory), desc="Processing"):
            folder_path = os.path.join(self.base_directory, folder_name)
            if not os.path.isdir(folder_path):
                continue

            # Process report and prompts
            report_files = [f for f in os.listdir(folder_path) 
                          if f.lower().startswith('report_') and f.endswith('.docx')]
            if not report_files:
                continue

            report_path = os.path.join(folder_path, report_files[0])
            report_text = self._extract_text_from_docx(report_path)

            for prompt_num in range(1, 7):
                prompt_files = [f for f in os.listdir(folder_path)
                               if f.lower() == f'prompt_{prompt_num}.docx']
                if not prompt_files:
                    continue

                prompt_path = os.path.join(folder_path, prompt_files[0])
                prompt_text = self._extract_text_from_docx(prompt_path)

                try:
                    self._create_agents()
                    grading_crew = self._create_workflow(prompt_text, report_text)
                    final_output = grading_crew.kickoff()

                    # Save final output
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