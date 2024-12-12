import os
import docx
import ollama
from ollama import ChatResponse
from ollama import chat
from tqdm import tqdm


response = ollama.chat(model='llama3.1:8b-instruct-q4_0', messages=[
    {'role': 'user', 'content': 'Hello, can you respond?'}
])
print(response['message']['content'])


# class ReportGrader:
#     def __init__(self, base_directory: str, model: str = 'llama3.1:8b-instruct-q4_0'):
#         """
#         Initialize the Report Grader for multiple folders
        
#         :param base_directory: Path to the directory containing all report folders
#         :param model: Ollama model to use for grading
#         """
#         self.base_directory = base_directory
#         self.model = model

#     def _extract_text_from_docx(self, file_path: str) -> str:
#         """
#         Extract text from a Word document
        
#         :param file_path: Path to the Word document
#         :return: Processed full text of the document
#         """
#         doc = docx.Document(file_path)
        
#         # Extract text and convert to lowercase
#         raw_text = '\n'.join([para.text for para in doc.paragraphs if para.text])
        
#         # Replace specific terms
#         processed_text = raw_text.lower()
#         for term in ['gpt', 'openai', 'chatgpt']:
#             processed_text = processed_text.replace(term, '')
        
#         return processed_text

#     def grade_reports(self):
#         """
#         Grade reports in all subdirectories
#         """
#         # Iterate through all subdirectories
#         for folder_name in tqdm(os.listdir(self.base_directory), desc="Processing Folders"):
#             folder_path = os.path.join(self.base_directory, folder_name)
            
#             # Skip if not a directory
#             if not os.path.isdir(folder_path):
#                 continue
            
#             # Find the report file
#             report_files = [f for f in os.listdir(folder_path) if f.startswith('report_') and f.endswith('.docx')]
            
#             # Skip if no report
#             if not report_files:
#                 continue
            
#             # Process the report
#             report_path = os.path.join(folder_path, report_files[0])
#             report_text = self._extract_text_from_docx(report_path)
#             report_name = os.path.splitext(report_files[0])[0]
            
#             # Process each prompt from 1 to 6
#             for prompt_num in tqdm(range(1, 7), desc=f"Prompts for {folder_name}", leave=False):
#                 prompt_file = f'Prompt_{prompt_num}.docx'
#                 prompt_path = os.path.join(folder_path, prompt_file)
                
#                 # Skip if prompt file doesn't exist
#                 if not os.path.exists(prompt_path):
#                     continue
                
#                 prompt_text = self._extract_text_from_docx(prompt_path)
                
#                 try:
#                     # Generate response from Ollama
#                     response: ChatResponse = chat(model=self.model, messages=[
#                         {'role': 'system', 'content': prompt_text},
#                         {'role': 'user', 'content': report_text}
#                     ])
                    
#                     # Prepare output filename
#                     output_filename = f'{self.model}_{report_name}_Prompt_{prompt_num}.docx'
#                     output_path = os.path.join(folder_path, output_filename)
                    
#                     # Save response to a new Word document
#                     output_doc = docx.Document()
#                     output_doc.add_paragraph(response['message']['content'])
#                     output_doc.save(output_path)
                
#                 except Exception as e:
#                     print(f"Error processing {folder_name}/Prompt_{prompt_num}: {e}")

# def main():
#     # Specify the base directory containing all report folders
#     base_directory = '/home/akash/Downloads/grading_documents'
    
#     grader = ReportGrader(base_directory)
#     grader.grade_reports()

# if __name__ == '__main__':
#     main()