import os
import re
from flask import Flask, render_template, request, jsonify
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize Pinecone client
import pinecone

os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['PINECONE_API_KEY'] = 'a232fcba-5aff-46bf-ac24-daa531663d11'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_vector_db(file_path, upload_folder):
    if not os.path.exists(file_path):
        return None

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=0)

    embeddings = SentenceTransformerEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                               model_kwargs={'device': 'cpu'})

    # Create Pinecone Vector Store
    vectorstore = PineconeVectorStore(index_name="paradoxstudy", embedding=embeddings)

    for document in documents:
        text_chunks = text_splitter.split_documents([document])
        for text_chunk in text_chunks:
            vectorstore.add_documents([text_chunk])

    # Move PDF file to upload folder
    pdf_filename = os.path.basename(file_path)
    pdf_name = os.path.splitext(pdf_filename)[0]
    pdf_folder_path = os.path.join(upload_folder, pdf_name)

    return pdf_name, pdf_folder_path

def final_result(query, selected_book):
    embeddings = SentenceTransformerEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                               model_kwargs={'device': 'cpu'})

    vectorstore = PineconeVectorStore(index_name="paradoxstudy", embedding=embeddings)

    results = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = results.get_relevant_documents(query=query)
    formatted_results = []

    for doc in docs:
        metadata = doc.metadata
        if "source" in metadata:
            source = metadata["source"]
            book_name = os.path.splitext(os.path.basename(source))[0]
            if book_name == selected_book:
                page_content = doc.page_content.replace("\\n", "\n")
                page_content = re.sub(r"(\d+)\.", r"\n\1.", page_content)
                page_content = re.sub(r"(\d+)\)", r"\1) ", page_content)
                page_content = re.sub(r"\n(\d+)\n", r"\n\n\1\n\n", page_content)
                page_content = re.sub(r"\n(\d+[\)\.])", r"\n\n\1", page_content)

                formatted_result = {
                    "metadata": metadata,
                    "page_content": page_content
                }
                formatted_results.append(formatted_result)

    return formatted_results

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get_books', methods=['GET'])
def get_books():
    book_names = [name for name in os.listdir(UPLOAD_FOLDER) if os.path.isdir(os.path.join(UPLOAD_FOLDER, name))]
    return jsonify(book_names)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'pdfFile' not in request.files:
        return "No file part"
    file = request.files['pdfFile']
    if file.filename == '':
        return "No selected file"
    if file and allowed_file(file.filename):
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        pdf_name, db_folder_path = create_vector_db(file_path, UPLOAD_FOLDER)
        os.remove(file_path)
        return jsonify({"pdf_name": pdf_name, "db_folder_path": db_folder_path})
    else:
        return "Invalid file type"

@app.route('/ask', methods=['POST'])
def ask_question():
    query = request.form['question']
    selected_book = request.form['selected_book']
    result = final_result(query, selected_book)

    return jsonify(result)

if __name__ == '__main__':
        app.run()

