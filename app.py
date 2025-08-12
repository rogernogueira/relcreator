from textwrap import dedent
from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.models.deepseek import DeepSeek
from  dotenv import load_dotenv
from pathlib import Path
import pymupdf4llm
from agno.tools import Toolkit
from agno.utils.log import log_info
import json 
from agno.playground import Playground

load_dotenv()
directory = "data/relatorios/" 

class FileTextExtractor(Toolkit):
    """A toolkit for extracting text from PDF files in a specified directory."""
    def __init__(self, base_dir, **kwargs):
        self.base_dir = Path(base_dir)
        super().__init__(name="FileTextExtractor",tools=[self.extract_texts_from_pdfs], **kwargs)
        
    def extract_texts_from_pdfs(self):
        """Extracts text from all PDF files in the specified directory.
        Args:
            directory (str): The path to the directory containing PDF files.
        Returns:
            str: A JSON string containing the extracted texts.
        """
        path = self.base_dir
        texts = []
        for file_pdf in path.iterdir():
            if (file_pdf.suffix.lower() == ".pdf") and file_pdf.is_file():
                text = pymupdf4llm.to_markdown(file_pdf)
                texts.append({"type": "text", "text": text})
                log_info(f"Extracted text from {file_pdf.name}")
        return json.dumps(texts)

agent = Agent(
    name="RelCreator",
    model=Ollama(id='qwen2.5',options={"num_ctx":15000, "top-p":0, "temperature":0.2} ),
    #model=DeepSeek(id='deepseek-chat'),
    tools=[FileTextExtractor(base_dir = "data/relatorios")],
    instructions=dedent("""
        1. Use a ferramenta FileTextExtractor para extrair o texto dos arquivos PDF do diretório padrao .
        2. Analise o texto extraído e crie um relatório com base nas informações contidas nos relatórios.
        3. sempre responda em **português**.
                        """),
    instructions=dedent(""" Responda em português, com cordialidade """),
    debug_mode=True,
    show_tool_calls=True,
    markdown=True,

 )


playground  = Playground(agents=[agent])
app = playground.get_app()

if __name__ == "__main__":
    playground.serve("app:app", reload=True)
