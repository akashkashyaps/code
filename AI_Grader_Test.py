import os
import docx
import ollama
from tqdm import tqdm

class ReportGrader:
    def __init__(self, base_directory: str, model: str = 'llama3.1:8b-instruct-q4_0'):
        self.base_directory = base_directory
        self.model = model
        print(f"Base Directory: {base_directory}")
        print(f"Directory Exists: {os.path.exists(base_directory)}")
        print(f"Is Directory: {os.path.isdir(base_directory)}")
        
        # List all items in the directory
        print("Directory Contents:")
        for item in os.listdir(base_directory):
            full_path = os.path.join(base_directory, item)
            print(f"- {item}: {'Directory' if os.path.isdir(full_path) else 'File'}")

    def _extract_text_from_docx(self, file_path: str) -> str:
        print(f"Extracting text from: {file_path}")
        print(f"File Exists: {os.path.exists(file_path)}")
        
        doc = docx.Document(file_path)
        
        raw_text = '\n'.join([para.text for para in doc.paragraphs if para.text])
        
        processed_text = raw_text.lower()
        for term in ['gpt', 'openai', 'chatgpt']:
            processed_text = processed_text.replace(term, '')
        
        print(f"Extracted text length: {len(processed_text)} characters")
        return processed_text

    def grade_reports(self):
        for folder_name in tqdm(os.listdir(self.base_directory), desc="Processing Folders"):
            folder_path = os.path.join(self.base_directory, folder_name)
            
            if not os.path.isdir(folder_path):
                print(f"Skipping non-directory: {folder_path}")
                continue
            
            print(f"\nProcessing Folder: {folder_name}")
            
            report_files = [f for f in os.listdir(folder_path) if f.startswith('Report_') and f.endswith('.docx')]
            
            if not report_files:
                print(f"No report files found in: {folder_path}")
                continue
            
            report_path = os.path.join(folder_path, report_files[0])
            report_text = self._extract_text_from_docx(report_path)
            report_name = os.path.splitext(report_files[0])[0]
            
            for prompt_num in tqdm(range(1, 7), desc=f"Prompts for {folder_name}", leave=False):
                prompt_file = f'Prompt_{prompt_num}.docx'
                prompt_path = os.path.join(folder_path, prompt_file)
                
                if not os.path.exists(prompt_path):
                    print(f"Prompt file not found: {prompt_path}")
                    continue
                
                prompt_text = self._extract_text_from_docx(prompt_path)
                
                try:
                    print(f"Generating response for {folder_name}, Prompt {prompt_num}")
                    
                    response = ollama.chat(model=self.model, messages=[
                        {'role': 'system', 'content': prompt_text},
                        {'role': 'user', 'content': report_text}
                    ])
                    
                    output_filename = f'{self.model}_{report_name}_Prompt_{prompt_num}.docx'
                    output_path = os.path.join(folder_path, output_filename)
                    
                    output_doc = docx.Document()
                    output_doc.add_paragraph(response['message']['content'])
                    output_doc.save(output_path)
                    
                    print(f"Response saved to: {output_path}")
                
                except Exception as e:
                    print(f"Error processing {folder_name}/Prompt_{prompt_num}: {e}")

def main():
    base_directory = '/home/akash/Downloads/grading_documents'
    
    grader = ReportGrader(base_directory)
    grader.grade_reports()

if __name__ == '__main__':
    main()