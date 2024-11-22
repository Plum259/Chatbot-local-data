from langchain_aws import BedrockEmbeddings
from langchain_aws import ChatBedrock
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import DistanceStrategy
import os
import boto3
import re
import json

session = boto3.Session(
    region_name='us-west-2'
)
bedrock_client = boto3.client(service_name='bedrock-runtime', 
                              region_name='us-west-2')

# Configuración de modelos de Bedrock
llm = ChatBedrock(
       client=bedrock_client,
       model_id='anthropic.claude-3-haiku-20240307-v1:0',
       model_kwargs={
           "temperature": 0.1,
           }
           )
embed_model = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1",
    client=bedrock_client,
    region_name="us-west-2"
    )

# Instrucciones para el asistente
Instrucciones = """
Eres un asistente especializado en solucionar problemas denominados "Incidentes", tienes que ser amable y gentil, 
Sigue estas normas en cada respuesta:
1. Proporciona una descripción clara y detallada de las soluciones, basandote exclusivamente en la informacion que tengas.
2. Si no tienes suficiente información, pide más datos de manera educada y ten en cuenta que no te van a entregar de forma textual los problemas guardados en tu data.
3. Sé preciso en tus respuestas, pero evita usar lenguaje técnico que los usuarios promedio no entiendan.
4. Ten en cuenta que es bastante probable que los usuarios tengan faltas de ortografia y tipeos incorrectos como por ejemplo escribir "qwe" en lugar de "que".
5. Intenta dar solucion de forma breve y no te extiendas tanto en dar la solucion
6. Recuerda que tambien los usuarios redactar las consultas con mas espacios de lo normal, por ejemplo "No  puedo  acceder  a  la  web"
7. En los casos mencionados previemante de mala redaccion, intenta o bien pedirle de forma amable al usuario que escriba nuevamente, o si entiendes bien y encuentras la solucion, da la solucion
"""

# Clase para leer los archivos JSON
class JSONDirectoryReader:
    def __init__(self, input_dir, recursive=False):
        self.input_dir = input_dir
        self.recursive = recursive

    def load_data(self):
        documents = []
        if not os.path.exists(self.input_dir):
            print(f"El directorio {self.input_dir} no existe.")
            return documents
        for root, dirs, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith(".json"):
                    with open(os.path.join(root, file), 'r') as f:
                        data = json.load(f)
                        for issue in data.get("Incidentes", []):
                            content = (
                                f"Ticket: {issue.get('Ticket', '')}\n"
                                f"Problema: {issue.get('Problema', '')}\n"
                                f"Solucion: {issue.get('Solucion', '')}\n"
                            )
                            documents.append(Document(page_content=content))
            if not self.recursive:
                break

        if not documents:
            print("No se encontraron documentos en los archivos JSON.")

        else:
            print(f"Se encontraron {len(documents)} incidentes en la data.")   
                 
        return documents


# Función para cargar o crear el índice
def create_or_load_index():
    data_dir = "tmp/data"
    index_file = "/tmp/faiss_index.json"

    # Verifica si el índice ya existe en un archivo JSON
    if os.path.exists(index_file):
        print(f"Cargando índice desde {index_file}")
        vectorstore = FAISS.load_local(index_file, embed_model, allow_dangerous_deserialization=True, distance_strategy=DistanceStrategy.COSINE)
    else:
        # Cargar documentos y crear el índice
        reader = JSONDirectoryReader(input_dir=data_dir, recursive=True)
        documents = reader.load_data()

        if not documents:
            raise ValueError("No se encontraron documentos para procesar")
        
        vectorstore = FAISS.from_documents(documents, embed_model, distance_strategy=DistanceStrategy.COSINE)

        # Guardar el índice en un archivo binario
        vectorstore.save_local(index_file)

    return vectorstore

# Función principal de Lambda
def lambda_handler(event, context):
    print()
    print(event)
        #body = json.loads(event['body'])
        #prompt = body.get('message', '')
    nombre_usuario = json.loads(event['body'])['from']['name']
    print("Usuario: " + nombre_usuario)
    body = json.loads(event['body'])['text']
    prompt = re.search(r'&nbsp;(.+)</p>', body).group(1).replace('&nbsp', '')
    print("Consulta: " + prompt)
    prompt.lower()
    # Combinar instrucciones con el prompt del usuario
    prompt_con_instrucciones = Instrucciones + "\nUsuario: " + prompt + "\nNombre del usuario: " + nombre_usuario

    # Cargar el índice y buscar resultados
    vectorstore = create_or_load_index()
    response = vectorstore.similarity_search(prompt_con_instrucciones, k=3)
    
    context_text = "\n\n".join([doc.page_content for doc in response])

    # Crear el prompt para el LLM combinando el contexto y la consulta del usuario
    llm_prompt = f"{Instrucciones}\nContexto relevante:\n{context_text}\n\nUsuario: {prompt}\nAsistente: \n Nombre del Usuario:{nombre_usuario}"

    # Generar la respuesta utilizando el LLM
    llm_response = llm.invoke(llm_prompt)

    # Extraer el contenido del mensaje de la respuesta del LLM
        #llm_text = llm_response['text'] if isinstance(llm_response, dict) and 'text' in llm_response else str(llm_response)

    llm_text = llm_response.content

    print("Respuesta del Bot: " + llm_text)

    # Devolver la respuesta generada por el LLM
    return {
    "type": "message",
    "text": llm_text
}