import os
import docx
import ollama
from tqdm import tqdm

class ReportGrader:
    def __init__(self, base_directory: str, model: str = 'mistral:7b-instruct'):
        self.base_directory = base_directory
        self.model = model

    def _extract_text_from_docx(self, file_path: str) -> str:
        doc = docx.Document(file_path)
        
        raw_text = '\n'.join([para.text for para in doc.paragraphs if para.text])
        
        processed_text = raw_text.lower()
        for term in ['gpt', 'openai', 'chatgpt']:
            processed_text = processed_text.replace(term, '')
        
        return processed_text

    def grade_reports(self):
        for folder_name in tqdm(os.listdir(self.base_directory), desc="Processing Folders"):
            folder_path = os.path.join(self.base_directory, folder_name)
            
            if not os.path.isdir(folder_path):
                continue
            
            # Find the report file
            report_files = [f for f in os.listdir(folder_path) 
                            if f.lower().startswith('report_') and f.lower().endswith('.docx')]
            
            if not report_files:
                continue
            
            # Process the report
            report_path = os.path.join(folder_path, report_files[0])
            report_text = self._extract_text_from_docx(report_path)
            report_name = os.path.splitext(report_files[0])[0]
            
            # Process each prompt from 1 to 6
            for prompt_num in range(1, 7):
                # More flexible prompt file matching
                prompt_files = [f for f in os.listdir(folder_path) 
                                if f.lower() == f'prompt_{prompt_num}.docx']
                
                if not prompt_files:
                    continue
                
                prompt_path = os.path.join(folder_path, prompt_files[0])
                prompt_text = self._extract_text_from_docx(prompt_path)
                
                try:
                    # Verbose system message to ensure prompt adherence
                    system_message = (
                        f"CRITICAL INSTRUCTIONS FOR PROMPT {prompt_num}:\n"
                        "1. You MUST carefully read and follow the specific prompt provided below.\n"
                        "2. Do NOT simply summarize the report.\n"
                        "3. Directly address the exact requirements of the prompt.\n"
                        "4. If the prompt asks a specific question, answer THAT question.\n"
                        "5. If the prompt requests a specific type of analysis, provide THAT analysis.\n\n"
                        f"SPECIFIC PROMPT DETAILS:\n{prompt_text}\n\n"
                        "PROMPT VERIFICATION: Confirm you will follow these instructions precisely."
                    )

                    # Generate response from Ollama
                    response = ollama.chat(model=self.model, messages=[
                        {'role': 'system', 'content': system_message},
                        {'role': 'user', 'content': report_text}
                    ])
                    
                    # Prepare output filename
                    output_filename = f'{self.model}_{report_name}_Prompt_{prompt_num}.docx'
                    output_path = os.path.join(folder_path, output_filename)
                    
                    # Save response to a new Word document
                    output_doc = docx.Document()
                    
                    # Add the original prompt to the document for reference
                    output_doc.add_paragraph(f"Original Prompt (Prompt_{prompt_num}):")
                    output_doc.add_paragraph(prompt_text)
                    output_doc.add_paragraph("\n--- AI Response ---\n")
                    
                    output_doc.add_paragraph(response['message']['content'])
                    output_doc.save(output_path)
                    
                    print(f"Processed {folder_name} - Prompt {prompt_num}")
                
                except Exception as e:
                    print(f"Error processing {folder_name}/Prompt_{prompt_num}: {e}")

def main():
    base_directory = '/home/akash/Downloads/grading_documents'
    
    grader = ReportGrader(base_directory)
    grader.grade_reports()

if __name__ == '__main__':
    main()