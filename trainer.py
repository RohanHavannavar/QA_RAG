import os
from typing import List,Callable,Union
from pathlib import Path
from langchain_community.embeddings import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader,PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from dataclasses import dataclass
from langchain_community.vectorstores import FAISS

@dataclass
class Trainer:

    def __init__(self,file_path_or_dir:str) -> None:
        self.file_path_or_dir = Path(file_path_or_dir)
        self.documents = self._load_pdf_and_split()
        self.embeddings = Trainer.create_embeddings()
        self.db = "faiss_db"

    def _load_pdf_and_split(self):
        """
        Load and extract text from a PDF file.

        Parameters:
        pdf_path (str): Path to the PDF file.

        Returns:
        str: Extracted text from the PDF.
        """
        if(self.file_path_or_dir.is_dir()):
            loader = DirectoryLoader(self.file_path_or_dir, glob="**/*.pdf",loader_cls = PyPDFLoader,use_multithreading=True)
        elif(self.file_path_or_dir.is_file()):
            loader = PyPDFLoader(self.file_path_or_dir)

        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        return splits
    
    @staticmethod
    def create_embeddings():
        return HuggingFaceEmbeddings()
    
    def train(self):
        if(Path(self.db).exists()):

            extension = FAISS.from_documents(self.documents, self.embeddings)
            local_db = FAISS.load_local(self.db,self.embeddings)
            local_db.merge_from(extension)
        else:
            
            local_db = FAISS.from_documents(self.documents, self.embeddings)
            local_db.save_local(folder_path=self.db)
        return local_db
# Usage example
if __name__ == "__main__":
    rag = Trainer(r"path to your pdf file")
    
    rag.train()
        
    


        
        
