from textwrap import dedent
from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.models.deepseek import DeepSeek
from  dotenv import load_dotenv
from pathlib import Path
import pymupdf4llm
import os
from agno.tools import Toolkit
from agno.utils.log import log_info
import json 
from agno.playground import Playground
import streamlit as st

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

instructions_raw = """
        1. Use a ferramenta FileTextExtractor para extrair o texto dos arquivos PDF do diretório padrao .
        2. Analise o texto extraído e crie um relatório com base nas informações contidas nos relatórios.
        3. sempre responda em **português**.
                        """
instructions = dedent(instructions_raw.strip()),
def get_agent( dir_path,instructions):
    return Agent(
        name="RelCreator",
        model=Ollama(id='qwen2.5',options={"num_ctx":4000, "top-p":0, "temperature":0.2} ),
        #model=DeepSeek(id='deepseek-chat'),
        tools=[FileTextExtractor(base_dir = dir_path)],
        instructions=instructions, 
        debug_mode=True,
        show_tool_calls=True,
        markdown=True,   
    )

if __name__ == "__main__":
    st.title("Gerador de Relatórios")
    with st.sidebar:
        st.header("Instruções do Agente")
        st.info(instructions_raw)
        diretorio_destino = directory
        if not os.path.exists(diretorio_destino):
            os.makedirs(diretorio_destino)
            st.info(f"Pasta '{diretorio_destino}' criada para salvar os arquivos.")

        uploaded_files = st.file_uploader(
            "Escolha um ou mais arquivos PDF...",
            type=["pdf"],
            accept_multiple_files=True  # Esta é a alteração principal!
        )

        if uploaded_files: # A lista de arquivos não é vazia
            st.success(f"{len(uploaded_files)} arquivo(s) selecionado(s) com sucesso!")
            
            # Iterar sobre cada arquivo na lista 'uploaded_files'
            for uploaded_file in uploaded_files:
                try:
                    # Constrói o caminho completo para o arquivo
                    caminho_arquivo = os.path.join(diretorio_destino, uploaded_file.name)

                    # Salva o arquivo no disco
                    with open(caminho_arquivo, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    st.write(f"- Arquivo '{uploaded_file.name}' salvo com sucesso!")

                except Exception as e:
                    st.error(f"Erro ao salvar o arquivo '{uploaded_file.name}': {e}")
            
        else:
            st.info("Nenhum arquivo PDF selecionado.")

    st.write("Sistema de geração de relatórios")
    prompt_input = st.text_input(label='Prompt', placeholder="Digite o prompt:")
    if not prompt_input:
        prompt_input = "Crie um relatório com base nos documentos  do diretório padrão."
    num_palavras = st.number_input("Máximo de palavras no relatório", min_value=100, max_value=2000, value=100)   
    dir_path  = st.text_input("Diretório dos PDFs", value=directory, placeholder="Digite o caminho do diretório dos PDFs")  
    #list files
    pdf_files = [f for f in Path(dir_path).glob("*.pdf") if f.is_file()]
    st.write("Arquivos PDF encontrados:")
    for pdf_file in pdf_files:
        st.write(f"- {pdf_file.name}")

    agent = get_agent(dir_path, instructions)
    st.warning("Certifique-se que o Ollama esta rodando.")

    if st.button("Iniciar Agente", type="primary", use_container_width=True) :
        with st.spinner("Executando agente..."):
            try:
                response  = agent.run(prompt_input+f", a resposta deve conter no máximo  {num_palavras} palavras")
            except Exception as e:
                st.error(f"Ocorreu um erro: {e}")
            st.markdown(response.content)