# app.py
import os
# Smart device selection BEFORE loading model: Use GPU if available, CPU for macOS ARM64 to avoid crash
import platform

# CRITICAL: Disable MPS on macOS ARM64 BEFORE importing torch to prevent segmentation fault
if platform.system() == 'Darwin' and platform.machine() == 'arm64':
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # Disable MPS completely

import base64
import csv
import ssl
import uvicorn
from io import StringIO
import faiss #GPU
import httpx
import importlib
import json
import logging
import numpy as np
import pytz
import re
import sys
import torch
import unicodedata
import uuid
from colorama import Fore, Style
from datetime import datetime, timedelta
from banking_product_router import BankingProductRouter
from semantic_memory import compress_conversation_history, truncate_to_last_duplas
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
from fastapi import FastAPI, Request
from fastapi import HTTPException
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from transformers import AutoTokenizer, AutoModel
import re as _re

# =========================
# INICIALIZACIÓN
# =========================
load_dotenv(override=True)

on_color = os.getenv("ON_COLOR", "true")
if on_color and on_color.lower() == "true":
    c_rst_ = Style.RESET_ALL
    c_blue = Fore.LIGHTBLUE_EX
    c_cyan = Fore.CYAN
    c_mage = Fore.LIGHTMAGENTA_EX
    c_yell = Fore.LIGHTYELLOW_EX
    c_gree = Fore.LIGHTGREEN_EX
    c_red_ = Fore.LIGHTRED_EX
    c_gray = Fore.LIGHTBLACK_EX
else:
    c_rst_ = ""
    c_blue = ""
    c_cyan = ""
    c_mage = ""
    c_yell = ""
    c_gree = ""
    c_red_ = ""
    c_gray = ""

app = FastAPI()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
)

app_name = os.getenv("APP_NAME", "???")

logging.info(f"{c_gree}APP NAME:                {c_red_}{app_name}{c_rst_}")
logging.info(f"{c_cyan}* Python:                {c_yell}{sys.version}{c_rst_}")
logging.info(f"{c_blue}* dotenv version:        {c_cyan}{importlib.metadata.version('python-dotenv')}{c_rst_}")
logging.info(f"{c_blue}* elasticsearch version: {c_cyan}{importlib.metadata.version('elasticsearch')}{c_rst_}")
logging.info(f"{c_blue}* faiss version:         {c_cyan}{faiss.__version__}{c_rst_}")
logging.info(f"{c_blue}* httpx version:         {c_cyan}{importlib.metadata.version('httpx')}{c_rst_}")
logging.info(f"{c_blue}* ibm_watsonx_ai version:{c_cyan}{importlib.metadata.version('ibm_watsonx_ai')}{c_rst_}")
logging.info(f"{c_blue}* numpy version:         {c_cyan}{importlib.metadata.version('numpy')}{c_rst_}")
logging.info(f"{c_blue}* pytz version:          {c_cyan}{importlib.metadata.version('pytz')}{c_rst_}")
logging.info(f"{c_blue}* torch version:         {c_cyan}{importlib.metadata.version('torch')}{c_rst_}")
logging.info(f"{c_blue}* transformers version:  {c_cyan}{importlib.metadata.version('transformers')}{c_rst_}")
logging.info(f"{c_blue}* uvicorn version:       {c_cyan}{importlib.metadata.version('uvicorn')}{c_rst_}")


# Variables de entorno ElasticSearch
E_ADMIN_USER = os.getenv("E_ADMIN_USER")
E_ADMIN_PASSWORD = os.getenv("E_ADMIN_PASSWORD")
E_HOST = os.getenv("E_HOST")
E_PORT = os.getenv("E_PORT")
E_CERT_PATH = os.getenv("E_CERT_PATH")
E_CERT_FILE_NAME = os.getenv("E_CERT_FILE_NAME")

# Variables de entorno IBM watsonx_ai
IBM_CLOUD_URL = os.getenv("IBM_CLOUD_URL")
API_KEY = os.getenv("API_KEY")
PROJECT_ID = os.getenv("PROJECT_ID")

# Variables reducción y compresión de historia conversacional
# history_max_dupla_msgs = 2 # Si el número de mensajes del ASISTENTE-CLIENTE es mayor o igual a este valor, 
# se aplicará reducción y compresión al historial de conversación. Si se pone en un número muy alto, no se aplicará reducción ni compresión al historial.
history_max_dupla_msgs = int(os.getenv("HISTORY_MAX_DUPLA_MSGS"))
#history_max_asistente_chars = 24 # Máximo de caracteres permitidos para cada mensaje del ASISTENTE VIRTUAL antes de aplicar compresión. 
# Si un mensaje del asistente supera este límite, se comprime (resumiendo o eliminando partes) para reducir tokens. 
# Si se pone en un número muy alto, no se aplicará compresión a los mensajes del asistente.
history_max_asistente_chars = int(os.getenv("HISTORY_MAX_ASISTENTE_CHARS"))
#history_preserve_last_n_turns= 1 # En 1 mantiene el último mensaje del asistente sin comprimir, 
# los mensajes anteriores del asistente si se comprimen si superan el límite de caracteres. Si se pone en 0, no se preserva ningún mensaje del asistente sin comprimir.
history_preserve_last_n_turns = int(os.getenv("HISTORY_PRESERVE_LAST_N_TURNS"))

# Prompts base
try:
    PROMPT_BASE = os.getenv("V_PROMPT_BASE", "")
    file_prompt = os.getenv("FILE_PROMPT", "")
    if len(PROMPT_BASE) < 20:
        logging.warning(f"{c_red_}** V_PROMPT_BASE vacío o muy corto, leyendo archivo '{file_prompt}'...{c_rst_}")
        with open(file_prompt, "r", encoding="utf-8") as f:
            PROMPT_BASE = f.read()
            logging.warning(f"{c_red_}PROMPT BASE LO LEE DE ARCHIVO, CAMBIAR!!!{c_rst_}")
        if len(PROMPT_BASE) < 20:
            raise ValueError("El archivo prompt también está vacío o es demasiado corto.")
except Exception as e:
    logging.error(f"{c_red_}** No se cargó el prompt base:\n{e}{c_rst_}")
    exit()

PRODUCTOS_MAP = {
    "cuentas": {"es_index":os.getenv("E_INDEX_NAME_CTASV2"), "blocklist": os.getenv("V_BLOCKLIST_CTASV2")},
    "prestamos": {"es_index":os.getenv("E_INDEX_NAME_PRESTAMOS"), "blocklist": os.getenv("V_BLOCKLIST_PRESTAMOS")},
    "seguros": {"es_index":os.getenv("E_INDEX_NAME_SEGUROS"), "blocklist": os.getenv("V_BLOCKLIST_SEGUROS")},
    "tarjetas": {"es_index":os.getenv("E_INDEX_NAME_TARJETAS"), "blocklist": os.getenv("V_BLOCKLIST_TARJETAS")}
}

critical_speech = os.getenv("CRITICAL_SPEECH", None)
guardrail_speech = os.getenv("GUARDRAIL_SPEECH", None)
out_of_data_speech = os.getenv("OUT_OF_DATA_SPEECH", None)
no_banking_speech = os.getenv("NO_BANKING_SPEECH", None)
saludo_speech = os.getenv("SALUDO_SPEECH", None)
emoji_number_speech = os.getenv("EMOJI_NUMBER_SPEECH", None)

def load_data_from_elasticsearch(index_name, embedding_field, chunk_field, section_field, sec_seq_field):
    embeddings = []
    ids = []
    chunk_data = []
    sections = []
    sec_seqs = []

    peru_tz = pytz.timezone('America/Lima')
    current_date = datetime.now(peru_tz).isoformat()
    # Define the query with filters for end_date and data_classification
    e_query = {
        "query": {
            "bool": {
                "must": [
                    {"range": {"start_date": {"lte": current_date}}},  # Filter for start_date less than or equal to current date #Dennys 2024-12-01
                    {"range": {"end_date": {"gt": current_date}}},  # Filter for end_date greater than curre
                    {"match": {"data_classification": "Public"}}   # Filter for data_classification equal to "Public"
                ]
            }
        },
        "_source": [embedding_field, chunk_field, section_field, sec_seq_field]
    }

    # Use Elasticsearch's scan helper to get all documents matching the query
    for hit in scan(elastic_client, index=index_name, query=e_query):
        embeddings.append(hit['_source'][embedding_field])
        chunk_data.append(hit['_source'][chunk_field])
        sections.append(hit['_source'][section_field])
        sec_seqs.append(hit['_source'][sec_seq_field])
        ids.append(hit['_id'])  # Store the document IDs for later retrieval

    return np.array(embeddings).astype('float32'), chunk_data, sections, sec_seqs, ids

def load_lookup_from_csv_url(url: str) -> dict:
    lookup = {}
    with httpx.Client(timeout=10) as client:
        response = client.get(url)
        response.raise_for_status()
    csv_content = response.content.decode("utf-8-sig")
    reader = csv.DictReader(StringIO(csv_content), delimiter=";")
    for row in reader:
        lookup[row["id_item"].strip()] = row["valor"].strip().strip("'")
    return lookup

# Cargar matriz de códigos de negocio a texto para reemplazo posterior en la respuesta semántica
try:
    cos_bucket_doc_attached_items = os.getenv("COS_BUCKET_DOC_ATTACHED_ITEMS")
    LOOKUP_CODES = load_lookup_from_csv_url(
        cos_bucket_doc_attached_items
    )
    logging.info(f"{c_cyan}** Matriz códigos cargada: {c_yell}{cos_bucket_doc_attached_items}{c_rst_}")
    logging.info(f"{c_cyan}** Infografías:            {c_yell}{os.getenv('COS_BUCKET_INFOGRAPHIES')}{c_rst_}")
except Exception as e:
    logging.error(f"{c_red_}** No se ha cargado la matriz de códigos(URL|TRX|INF|EML|TEL|DIR)\no no se cargó la ruta de infografías\n{e}{c_rst_}")
    exit()

def generate_total_prompt(kb_data, conversation_history, blocklist):
    PROMPT_TOTAL = PROMPT_BASE + "\n" + f"""

<HISTORIAL DE CONVERSACIÓN>
{conversation_history}
</HISTORIAL DE CONVERSACIÓN>

<BASE DE CONOCIMIENTO>
{kb_data}
</BASE DE CONOCIMIENTO>

<BLOCKLIST>
{blocklist} 
</BLOCKLIST>

"""
    return PROMPT_TOTAL

def semantic_search(query_embedding, dominio_producto, top_k=6):
    # PHASE 1 OPTIMIZATION: Use pre-built FAISS index instead of rebuilding
    index_faiss = FAISS_INDEXES[dominio_producto]
    distances, indices = index_faiss.search(query_embedding, top_k)
    return indices[0], distances[0]  # Return the best K results and distances

def extraer_valor(cadena_json, nombre_variable):
    # Patrón para capturar cadenas, booleanos y números
    try:
        patron = rf'"{nombre_variable}":\s*(?:"([^"]*)"|(\w+))'
        resultado = re.search(patron, cadena_json)
        if resultado:
            valor = resultado.group(1) or resultado.group(2)
            return valor
        else:
            return None
    except:
        return None
    
def normalize_text(text: str) -> str:
    if not text:
        return text

    # 1. Normalizar unicode
    text = unicodedata.normalize("NFKC", text)

    # 2. Eliminar caracteres de control (CRÍTICO)
    text = re.sub(r'[\x00-\x1F\x7F]', ' ', text)

    # 3. Reemplazar bullets y símbolos raros
    text = re.sub(r"[•·●▪◦■►]", " ", text)

    # 4. Limpiar espacios múltiples
    text = re.sub(r"\s+", " ", text).strip()

    return text

def replace_business_codes(text: str, lookup: dict) -> str:
    if not text:
        return text
    pattern = r"(?:'|\")?\b(URL|TRX|INF|EML|TEL|DIR)\d{4}\b(?:'|\")?"

    def replacer(match):
        full_match = match.group(0)
        code = re.search(r'(URL|TRX|INF|EML|TEL|DIR)\d{4}', full_match).group(0)
        value = lookup.get(code)
        if not value:
            return code  # fallback
        return f" {value} "

    text = re.sub(pattern, replacer, text)
    text = re.sub(r"\s+", " ", text).strip()

    return text

def log_request_time():
    # Hora del servidor (con timezone incluido si el sistema lo tiene)
    server_time = datetime.now().astimezone().isoformat()

    # Hora local Lima/Perú
    lima_tz = pytz.timezone("America/Lima")
    local_time = datetime.now(lima_tz).isoformat()

    print(
        f"\n{c_gree}====== {c_rst_}API Request:: {c_gree}SERVER TIME : {server_time} {c_rst_}| {c_yell}LOCAL TIME Lima/Perú: {local_time} {c_gree}======{c_rst_}\n"
    )

def sinteticeSemanticAnswer(rawResponseSemanticAnswer: str, chunks_found: int, dominio_producto: str) -> str:

    client_response = extraer_valor(rawResponseSemanticAnswer, "client_response")
    out_of_data     = extraer_valor(rawResponseSemanticAnswer, "out_of_data")
    no_banking_conversation     = extraer_valor(rawResponseSemanticAnswer, "no_banking_conversation")
    transfer_human  = extraer_valor(rawResponseSemanticAnswer, "transfer_human")
    finish          = extraer_valor(rawResponseSemanticAnswer, "finish")
    guardrail       = extraer_valor(rawResponseSemanticAnswer, "guardrail")
    critical        = extraer_valor(rawResponseSemanticAnswer, "critical")
    go_menu         = extraer_valor(rawResponseSemanticAnswer, "go_menu")
    attach_trx_flow = extraer_valor(rawResponseSemanticAnswer, "attach_trx_flow")
    first_contact_resolution = extraer_valor(rawResponseSemanticAnswer, "first_contact_resolution")
    keep_conversation = extraer_valor(rawResponseSemanticAnswer, "keep_conversation")

    # Forzar out_of_data solo cuando no se identificó ningún dominio de producto.
    # Si el dominio fue identificado (cuentas/tarjetas/etc.) pero no se hallaron chunks,
    # el LLM tiene contexto conversacional previo y su decisión se respeta.
    if chunks_found < 1 and dominio_producto not in PRODUCTOS_MAP:
        out_of_data = "TRUE"
        first_contact_resolution = "FALSE"

    # Normalización
    def is_true(value):
        return str(value).lower() == "true"

    responseSemanticAnswer_processed = ""

    # Reglas de prioridad
    # PRIORIDAD 1
    if is_true(critical):
        responseSemanticAnswer_processed = critical_speech
        first_contact_resolution = "FALSE"
        dominio_producto = "POST_GEN_AI_CRITICAL"
    # PRIORIDAD 2
    elif attach_trx_flow and attach_trx_flow.upper() != "FALSE":
        responseSemanticAnswer_processed = replace_business_codes(attach_trx_flow, LOOKUP_CODES)
        first_contact_resolution = "FALSE"
        dominio_producto = "POST_GEN_AI_ATTACH_TRX_FLOW"
    # PRIORIDAD 3
    elif is_true(finish):
        responseSemanticAnswer_processed = "Te derivaré a NPS"
        first_contact_resolution = "TRUE"
        dominio_producto = "POST_GEN_AI_FINISH"
    # PRIORIDAD 4
    elif is_true(guardrail):
        responseSemanticAnswer_processed = guardrail_speech
        first_contact_resolution = "FALSE"
        dominio_producto = "POST_GEN_AI_GUARDRAIL"
    # PRIORIDAD 5
    elif is_true(transfer_human):
        responseSemanticAnswer_processed = "Para continuar con la atención, te transferiré con un asesor."
        first_contact_resolution = "FALSE"
        dominio_producto = "POST_GEN_AI_TRANSFER_HUMAN"
    # PRIORIDAD 6
    elif is_true(go_menu):
        responseSemanticAnswer_processed = "deriva_menu"
        first_contact_resolution = "FALSE"
        dominio_producto = "POST_GEN_AI_DERIVA_MENU"
    # PRIORIDAD 7
    elif is_true(no_banking_conversation):
        responseSemanticAnswer_processed = no_banking_speech
        first_contact_resolution = "FALSE"
        dominio_producto = "POST_GEN_AI_NO_BANKING_CONVERSATION"
    # PRIORIDAD 8
    elif is_true(out_of_data) and not is_true(keep_conversation):
        responseSemanticAnswer_processed = out_of_data_speech
        first_contact_resolution = "FALSE"
        dominio_producto = "POST_GEN_AI_OUT_OF_DATA"
    # DEFAULT
    else:
        responseSemanticAnswer_processed = replace_business_codes(client_response, LOOKUP_CODES)

    if client_response:
        client_response = client_response.replace("\\n", "\n")

    complete_response_updated = {
        "client_response": client_response,
        "critical": critical,               #Prioridad 1
        "attach_trx_flow": attach_trx_flow, #Prioridad 2
        "finish": finish,                   #Prioridad 3
        "guardrail": guardrail,             #Prioridad 4
        "transfer_human": transfer_human,   #Prioridad 5
        "go_menu": go_menu,                 #Prioridad 6
        "no_banking_conversation": no_banking_conversation, #Prioridad 7
        "out_of_data": out_of_data,         #Prioridad 8
        "first_contact_resolution": first_contact_resolution
        }
    
    complete_response_updated = "```json\n" + json.dumps(complete_response_updated, indent=2, ensure_ascii=False) + "\n```"

    flags =  f"| 1° {c_yell}critical:{c_rst_}{critical} "
    flags += f"| 2° {c_yell}attach_trx_flow:{c_rst_}{c_gree}{attach_trx_flow}{c_rst_} "
    flags += f"| 3° {c_yell}finish:{c_rst_}{finish} "
    flags += f"| 4° {c_yell}guardrail:{c_rst_}{guardrail} "
    flags += f"| 5° {c_yell}transfer_human:{c_rst_}{transfer_human} "
    flags += f"| 6° {c_yell}go_menu:{c_rst_}{go_menu} "
    flags += f"| 7° {c_yell}no_banking_conversation:{c_rst_}{no_banking_conversation} "
    flags += f"| 8° {c_yell}out_of_data:{c_rst_}{out_of_data} and {c_yell}keep_conversation:{c_rst_}{keep_conversation} "
    flags = flags.replace("TRUE", f"{c_gree}TRUE{c_rst_}").replace("FALSE", f"{c_red_}FALSE{c_rst_}").replace("|", f"{c_mage}|{c_rst_}")
    print("\n")
    logging.info(f"🔵 {c_cyan}FLAGS priority:{c_rst_} {flags}")

    outputvars =  f"{c_yell}client_response:{c_rst_}{client_response} "
    outputvars += f"| {c_yell}first_contact_resolution:{c_rst_}{first_contact_resolution} "
    outputvars += f"| {c_yell}responseSemanticAnswer_processed:{c_rst_}{responseSemanticAnswer_processed} "
    outputvars = outputvars.replace("TRUE", f"{c_gree}TRUE{c_rst_}").replace("FALSE", f"{c_red_}FALSE{c_rst_}").replace("|", f"{c_mage}|{c_rst_}")
    print("\n")
    logging.info(f"🟢 {c_gree}Output vars:{c_rst_}    {outputvars}")
    print("\n")
    logging.info(f"{c_gree}🟢 dominio_producto procesado: {c_yell}{dominio_producto}{c_rst_}")

    return responseSemanticAnswer_processed, first_contact_resolution, complete_response_updated, dominio_producto

def generate_answer_from_chunks(conversation_history, dominio_producto, model_llm, n_chunks, min_new_tokens, max_new_tokens, model_id, conversation_id):
    
    total_prompt = ""
    semantic_search_time = timedelta(0)  # timedelta(0) por defecto
    logging.info(f"{c_mage}🟪 dominio_producto detectado: {c_yell}{dominio_producto}{c_rst_}")

    if dominio_producto == "__multi__":
        # ── RAG multi-dominio: buscar en todos los índices detectados y combinar ──
        logging.info(f"{c_mage}🟪 [RAG] Ejecutando búsqueda multi-dominio...{c_rst_}")
        top_ids = []
        sorted_chunks = []
        combined_kb = ""
        chunks_per_domain = max(1, n_chunks // len(PRODUCTOS_MAP))  # distribuir chunks

        # Generar embedding una sola vez (igual que en el flujo normal)
        query_for_embedding = get_query_for_embedding(conversation_history, model_llm)
        if TEST_MODE:
            cache_query_embedding = np.random.randn(1, 384).astype(np.float32)
        else:
            with torch.inference_mode():
                if ENABLE_MODEL_CACHE and EMBEDDING_MODEL is not None:
                    _inp = EMBEDDING_TOKENIZER(
                        query_for_embedding, return_tensors="pt",
                        truncation=True, padding=True, max_length=512
                    )
                    _inp = {k: v.to(EMBEDDING_DEVICE).contiguous() for k, v in _inp.items()}
                    _out = EMBEDDING_MODEL(**_inp)
                    cache_query_embedding = (
                        _out.last_hidden_state.contiguous().mean(dim=1).cpu().numpy()
                    )
                else:
                    _tok = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
                    _mod = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
                    _inp = _tok(query_for_embedding, return_tensors="pt",
                                truncation=True, padding=True, max_length=512)
                    _out = _mod(**_inp)
                    cache_query_embedding = _out.last_hidden_state.mean(dim=1).numpy()

        _ss_start = datetime.now()
        for prod in PRODUCTOS_MAP:
            if prod not in FAISS_INDEXES:
                continue
            try:
                _top_indices, _ = semantic_search(cache_query_embedding, prod, top_k=chunks_per_domain)
                _top_ids_prod   = [doc_ids[prod][i]    for i in _top_indices]
                _top_chunks_prod = [chunk_data[prod][i] for i in _top_indices]
                top_ids.extend(_top_ids_prod)
                sorted_chunks.extend(_top_chunks_prod)
                if _top_chunks_prod:
                    combined_kb += f"\n\n[Dominio: {prod.upper()}]\n" + "\n".join(_top_chunks_prod)
                logging.info(
                    f"{c_mage}🟪 [RAG multi] {prod}: {c_yell}{len(_top_ids_prod)} chunks{c_rst_}"
                )
            except Exception as e:
                logging.warning(f"{c_red_}[RAG multi] Error en dominio {prod}: {e}{c_rst_}")

        semantic_search_time = datetime.now() - _ss_start
        total_prompt = generate_total_prompt(combined_kb, conversation_history, "")
        dominio_producto = "cuentas"  # dominio de retorno por defecto para multi
    elif dominio_producto not in PRODUCTOS_MAP:
        logging.info(
            f"{c_mage}🟪 [RAG] dominio no mapeado en PRODUCTOS_MAP. Se genera prompt SIN KB específica.{c_rst_}"
        )
        # Total prompt sin conocimiento específico, el LLM deberá responder basándose solo en su conocimiento general y el historial de conversación. 
        # Se le indicará que no se ha encontrado información relevante en la base de conocimientos para el producto identificado.
        top_ids = []
        sorted_chunks = []
        total_prompt = generate_total_prompt("", conversation_history, "") 
    else:
        logging.info(f"{c_mage}🟪 [RAG] dominio mapeado. Se genera prompt CON KB para: {c_yell}{dominio_producto}{c_rst_}")
        blocklist = PRODUCTOS_MAP[dominio_producto]["blocklist"]

        # ── NUEVO: obtener query óptima para embedding ──────────────
        query_for_embedding = get_query_for_embedding(conversation_history, model_llm)
        # ────────────────────────────────────────────────────────────
        
        # TEST_MODE: Use random embedding for Mac testing (bypasses PyTorch crash)
        if TEST_MODE:
            print(f"{c_yell}🧪 TEST_MODE: Using random embedding (for Mac testing only){c_rst_}")
            # Generate random embedding matching the model's dimension (384 for multilingual-e5-small)
            cache_query_embedding = np.random.randn(1, 384).astype(np.float32)
        else:
            # PHASE 1 OPTIMIZATION: Use cached or per-request embedding model
            # Use inference_mode for better performance and stability
            with torch.inference_mode():
                if ENABLE_MODEL_CACHE and EMBEDDING_MODEL is not None:
                    # Use cached model (faster, but may crash on macOS ARM64)
                    cache_query_input = EMBEDDING_TOKENIZER(
                        query_for_embedding, #conversation_history,
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                        max_length=512
                    )
                    
                    # Move input tensors to the same device as the model
                    cache_query_input = {
                        k: v.to(EMBEDDING_DEVICE).contiguous()
                        for k, v in cache_query_input.items()
                    }
                    
                    # Run model inference
                    cache_query_output = EMBEDDING_MODEL(**cache_query_input)
                    
                    # Ensure output is contiguous before operations
                    cache_query_embedding = (
                        cache_query_output.last_hidden_state
                        .contiguous()
                        .mean(dim=1)
                        .cpu()
                        .numpy()
                    )
                else:
                    # Per-request loading (slower but more stable on macOS ARM64)
                    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
                    model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
                    
                    cache_query_input = tokenizer(
                        query_for_embedding, # conversation_history,
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                        max_length=512
                    )
                    
                    # Run on CPU for stability
                    cache_query_output = model(**cache_query_input)
                    cache_query_embedding = (
                        cache_query_output.last_hidden_state
                        .mean(dim=1)
                        .numpy()
                    )

        _ss_start = datetime.now()
        top_indices, top_distances = semantic_search(cache_query_embedding, dominio_producto, top_k=n_chunks)
        semantic_search_time = datetime.now() - _ss_start

        # ── NUEVO: filtro por distancia L2 por dominio ──────────────────
        _default_threshold = float(os.getenv("MAX_L2_DISTANCE", "5.5"))
        _key = f"MAX_L2_DISTANCE_{dominio_producto.upper()}"
        MAX_L2_DISTANCE = float(os.getenv(_key, _default_threshold))

        mask = top_distances <= MAX_L2_DISTANCE
        top_indices   = [i   for i, keep in zip(top_indices,   mask) if keep]
        top_distances = [d   for d, keep in zip(top_distances, mask) if keep]

        logging.info(
            f"{c_mage}🟪 Chunks tras filtro L2 (dist<={MAX_L2_DISTANCE}): "
            f"{c_yell}{len(top_indices)}/{n_chunks}{c_rst_}"
        )

        # Retrieve the corresponding data for the top results
        top_ids = [doc_ids[dominio_producto][i] for i in top_indices]
        top_chunk_data = [chunk_data[dominio_producto][i] for i in top_indices]
        top_sections = [sections[dominio_producto][i] for i in top_indices]
        top_sec_seqs = [sec_seqs[dominio_producto][i] for i in top_indices]

        # Sort chunks while preserving their FAISS distance for logging
        chunk_records = list(zip(top_chunk_data, top_sections, top_sec_seqs, top_distances))
        sorted_chunk_records = sorted(chunk_records, key=lambda x: (x[1], x[2]))
        sorted_chunks = [item[0] for item in sorted_chunk_records]

        # KB data
        kb_data = ""

        if LOG_CHUNKS:
            logging.info(f"{c_mage}* KB chunks recuperados ({dominio_producto}): {c_yell}{len(sorted_chunks)}{c_rst_}")

        # Print out the sorted chunks
        for i, (chunk, _, _, distance) in enumerate(sorted_chunk_records):
            chunk_text = str(chunk).strip()
            if LOG_CHUNKS:
                chunk_preview = chunk_text.replace("\n", " ").replace("\r", " ")
                if len(chunk_preview) > 200:
                    chunk_preview = chunk_preview[:200] + "..."

                logging.info(
                    f"{c_mage}* KB chunk {i + 1:02d}/{len(sorted_chunks)} [dist={float(distance):.2f}]:{c_cyan} {chunk_preview}{c_rst_}"
                )

            kb_data += chunk_text + ' | '

        total_prompt = generate_total_prompt(kb_data, conversation_history, blocklist)

    datetime_start_time = datetime.now()
    complete_response = ""
    llm_answer = model_llm.generate_text(total_prompt).strip()
    complete_response += llm_answer

    datetime_end_time = datetime.now()
    llm_total_time = datetime_end_time - datetime_start_time

    chunks_found = len(top_ids)

    responseSemanticAnswer, first_contact_resolution, complete_response_updated, dominio_producto = sinteticeSemanticAnswer(complete_response, chunks_found, dominio_producto)
    responseSemanticAnswer = responseSemanticAnswer.replace("\\n", "\n")

    request_id = str(uuid.uuid4())
    request_sample = {
        "llm_request_id": request_id,
        "llm_complete_response": complete_response_updated,
        "llm_timestamp_start_time": datetime_start_time.isoformat(),
        "llm_timestamp_end_time": datetime_end_time.isoformat(),
        "llm_instance_url": IBM_CLOUD_URL,
        "llm_project_id": PROJECT_ID,
        "min_new_tokens": min_new_tokens,
        "max_new_tokens": max_new_tokens,
        "model_id": model_id,
        "conversation_id": conversation_id
    }
    return  top_ids, sorted_chunks, responseSemanticAnswer, datetime_start_time, llm_total_time, semantic_search_time, total_prompt, request_sample, first_contact_resolution, dominio_producto


def pre_review_before_generative_ai(ultimo_mensaje: str) -> str:
    """
    Pre-analiza el último mensaje del CLIENTE antes de enviarlo a la IA Generativa.
    Permite detectar casos simples sin necesidad de invocar el LLM:
      - 'critical'           si el mensaje contiene lisuras o insultos en español,
                             incluyendo variantes con mayúsculas, tildes o letras repetidas (Caso A).
      - 'deriva_menu'        si el mensaje contiene una variante de la palabra 'menú' (Caso B).
      - 'saludos'            si el mensaje es únicamente un saludo sin intención bancaria (Caso C).
      - 'emoji_number' si el mensaje es únicamente emojis (Caso D),
                             es únicamente dígitos (Caso E),
                             o es una combinación de emojis y números sin texto alfabético (Caso F).
      - 'go_generative_ai'   en cualquier otro caso (se envía al LLM con normalidad).
    """

    # Normalización reutilizable para Casos A, B y C:
    # - minúsculas, sin tildes, sin caracteres especiales, sin letras repetidas
    msg_normalizado = unicodedata.normalize("NFD", ultimo_mensaje.lower())
    msg_sin_tildes  = "".join(c for c in msg_normalizado if unicodedata.category(c) != "Mn")
    msg_limpio      = re.sub(r'[^a-z0-9\s]', '', msg_sin_tildes)
    msg_limpio      = re.sub(r'([aeiou])\1+', r'\1', msg_limpio)        # vocales: colapsa todas las repetidas        "mieeerda"  → "mierda"
    msg_limpio      = re.sub(r'([rl])\1{2,}', r'\1\1', msg_limpio)      # r y l:   colapsa si hay 3 o más repetidas   "zorrrrra"  → "zorra"
    msg_limpio      = re.sub(r'([^aeiourl\s])\1+', r'\1', msg_limpio)   # resto:   colapsa cualquier repetición        "mierdddda" → "mierda"

    # Caso A: lisuras e insultos en español
    DROGAS = [
        "anfetamina", "anfetaminas", "bazuco", "bazuca", "cannabis", "cocaina",
        "droga", "drogas", "drogadicto", "drogadicta", "drogadiccion", "drogado", "drogada", "drogarse",
        "extasis", "fentanilo", "heroina", "inhalar", "jeringa", "jerigas", "ketamina",
        "marihuana", "mariguana", "metanfetamina", "metanfetaminas", "morfina",
        "narcotraficante", "narcotraficantes", "narcotrafico",
        "opioide", "opioides", "opio",
        "pasta base", "psychedelico", "psicodelico", "psicodelicos",
        "rohypnol", "sobredosis", "terokal", "tussi", "weed", "xanax",
    ]

    # Palabras dirigidas a insultar o degradar a una persona
    INSULTOS = [
        "ananau", "animal", "atorrante", "awevo", "awevon", "ayayero",
        "bastardo", "bastarda", "bestia", "bisexual",
        "cachudo", "cachuda", "cabron", "cabrona", "choro", "chorra", "cojudo", "cojuda",
        "desgraciado", "desgraciada",
        "floro", "florona",
        "gay", "golpear",
        "homofobico", "homosexual", "huevon", "huevona",
        "idiota", "idiotas", "idiotez", "imbecil", "imbeciles", "imbecilidad", "inutil", "inutiles",
        "lesbiana", "loca",
        "maldito", "maldita", "mamerto", "mamerta", "mamertos", "mamertas", "marica", "maricon", "mariquita", "mariquitas",
        "pandejo", "pandeja", "pandejos", "pandejas", "pandejada", "pastrulo", "pendejo", "pendeja", "pendejos", "pendejas", "pendejada", "perra",
        "retrasado", "retrasada", "retrasados", "retrasadas",
        "sonso", "sonsa", "sonsos", "sonsas", "sonsaso", "sonsasa", "sonsasos", "sonsasas", "subnormal",
        "tarado", "tarada", "tarados", "taradas", "tonta", "tonto", "tontos", "tontas", "tonteria", "tonterias", "travesti", "trans",
        "chuki", "chuky", "chuchumeca",
    ]

    # Palabras obscenas, sexuales, escatológicas o de contenido prohibido
    MALAS_PALABRAS = [
        "caca", "cachar", "cachada", "cachando", "cagada", "cagar", "cago", "caracho", "carajo", "chucha", "chuchatumadre", "coger", "cogida", "concha", "conchatumadre", "cono", "copular",
        "crimen", "criminal", "criminales", "ctm",
        "defecar", "defecacion", "drogas",
        "eyacular", "eyaculacion", "excremento",
        "follar", "follada", "follando", "fornicacion", "fornicar", "fornicando",
        "hdp", "hijodeputa", "hijoeputa",
        "masoquismo", "masoquista", "masturbar", "masturbarse", "masturbacion", "matar", "matarse", "mecago", "mecagon", "meorin", "meorinar", "meretriz", "meretrices",
        "mierda", "mierd", "mojon", "muerte",
        "narcotrafico", "narcotraficante",
        "orina", "orinar", "orgasmo", "orgia",
        "pajero", "pajera", "pajeros", "pajeras", "pajerearse", "pajerear", "pene", "pedo", "pedorro", "pedorra", "porno", "pornografia", "pornografico",
        "peludo", "peluda", "prostituta", "prostituto", "prostitucion", "puta", "putas", "puto", "putos", "put",
        "rctm", "ramera", "rameras", "reconcha", "reconchatu", "relacion sexual",
        "sadismo", "sadista", "semen", "sexo", "sexual", "sexualidad", "sicario", "sodoma", "sodomia", "suicidio", "suicida", "suicidarse",
        "tirar", "tirada", "tirarse",
        "vagina", "vomitar", "vomito", "vomiton", "violacion", "violar", "violador", "violencia", "violento",
        "zoofilia", "zorra", "zorras",
    ]
    LISURAS = DROGAS + INSULTOS + MALAS_PALABRAS
    caso_a = any(re.search(rf'\b{re.escape(l)}\b', msg_limpio) for l in LISURAS)
    if caso_a:
        return "critical"

    # Caso B: variantes de "menú"
    caso_b = bool(re.search(r'\bmenuu*\b', msg_sin_tildes, re.IGNORECASE))
    if caso_b:
        return "deriva_menu"

    # Caso C: saludos sin intención bancaria
    # Detecta si el mensaje comienza con un saludo y no contiene palabras de intención bancaria
    SALUDOS = [
        "buen dia", "buen tarde", "buen noche",          # variantes normalizadas de buenos/buenas
        "buenas", "buenos",
        "hola", "ola", "holi", "holaa", "holap",
        "hey", "hi", "hello", "olis", "holis", "holiss", "holisss",
        "saludos", "que tal", "como estas", "como esta",
        "buenoas", "buenoa",                              # errores tipográficos comunes
    ]
    INTENCION_BANCARIA = [
        "cuenta", "tarjeta", "prestamo", "seguro", "credito", "debito",
        "transferencia", "pago", "cobro", "retiro", "deposito", "saldo",
        "producto", "servicio", "informacion", "consulta", "quiero", "necesito",
        "ayuda", "quisiera", "solicitar", "aperturar", "abrir", "cancelar",
        "adquiero", "adquirir", "contrato", "contratar", "afilio", "afiliar",
        "obtengo", "obtener", "pido", "pedir","solicito"
    ]
    tiene_saludo    = any(re.search(rf'\b{re.escape(s)}\b', msg_limpio) for s in SALUDOS)
    tiene_intencion = any(re.search(rf'\b{re.escape(i)}\b', msg_limpio) for i in INTENCION_BANCARIA)
    caso_c = tiene_saludo and not tiene_intencion
    if caso_c:
        return "saludos"

    # Caso D: solo emojis
    emoji_pattern = re.compile(
        r'^(?:[\U0000203C\U00002049\U00002122\U00002139\U00002194-\U00002199'
        r'\U000021A9-\U000021AA\U0000231A-\U0000231B\U00002328\U000023CF'
        r'\U000023E9-\U000023F3\U000023F8-\U000023FA\U000024C2'
        r'\U000025AA-\U000025AB\U000025B6\U000025C0\U000025FB-\U000025FE'
        r'\U00002600-\U00002604\U0000260E\U00002611\U00002614-\U00002615'
        r'\U00002618\U0000261D\U00002620\U00002622-\U00002623\U00002626'
        r'\U0000262A\U0000262E-\U0000262F\U00002638-\U0000263A\U00002640'
        r'\U00002642\U00002648-\U00002653\U0000265F\U00002660\U00002663'
        r'\U00002665-\U00002666\U00002668\U0000267B\U0000267E'
        r'\U00002692-\U00002697\U00002699\U0000269B-\U0000269C'
        r'\U000026A0-\U000026A1\U000026AA-\U000026AB\U000026B0-\U000026B1'
        r'\U000026BD-\U000026BE\U000026C4-\U000026C5\U000026C8'
        r'\U000026CE-\U000026CF\U000026D1\U000026D3-\U000026D4'
        r'\U000026E9-\U000026EA\U000026F0-\U000026F5\U000026F7-\U000026FA'
        r'\U000026FD\U00002702\U00002705\U00002708-\U0000270D\U0000270F'
        r'\U00002712\U00002714\U00002716\U0000271D\U00002721\U00002728'
        r'\U00002733-\U00002734\U00002744\U00002747\U0000274C\U0000274E'
        r'\U00002753-\U00002755\U00002757\U00002763-\U00002764'
        r'\U00002795-\U00002797\U000027A1\U000027B0\U000027BF'
        r'\U00002934-\U00002935\U00002B05-\U00002B07\U00002B1B-\U00002B1C'
        r'\U00002B50\U00002B55\U00003030\U0000303D\U00003297\U00003299'
        r'\U0001F000-\U0001FAFF])+$',
        re.UNICODE
    )
    caso_d = bool(emoji_pattern.match(ultimo_mensaje))

    # Caso E: solo dígitos
    caso_e = bool(re.fullmatch(r'\d+', ultimo_mensaje))

    # Caso F: combinación de emojis y/o números sin texto alfabético real
    caso_f = bool(re.fullmatch(r'[\d\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0000FE00-\U0000FE0F\s]+', ultimo_mensaje))

    if caso_d or caso_e or caso_f:
        return "emoji_number"

    return "go_generative_ai"

def ultimo_mensaje_cliente(conversation_history: str) -> str:
    matches = re.findall(r"CLIENTE:\s*'([^']*)'", conversation_history)
    if not matches:
        logging.warning(f"{c_yell}* get_query_for_embedding: No se encontró mensaje del CLIENTE. Se usa historial completo.{c_rst_}")
        return conversation_history,[]
    
    umc = matches[-1].strip()
    return umc, matches

def ultimo_mensaje_asistente(conversation_history: str) -> str:
    matches = re.findall(r"ASISTENTE VIRTUAL:\s*'([^']*)'", conversation_history)
    if not matches:
        logging.warning(f"{c_yell}* No se encontró mensaje del ASISTENTE VIRTUAL. Se devuelve ''.{c_rst_}")
        return "",[]
    
    umc = matches[-1].strip()
    return umc, matches

def contar_mensajes_cliente(conversation_history: str) -> int:
    """
    Cuenta cuántos mensajes ha enviado el CLIENTE en el conversation_history.
    """
    if not conversation_history:
        return 0

    # Busca todas las ocurrencias de CLIENTE:
    matches = re.findall(r"\bCLIENTE\s*:", conversation_history)
    
    return len(matches)

# Mensajes del cliente sin contenido bancario propio: el cliente profundiza
# en el tema que el asistente ya mencionó. Para el embedding se necesita
# combinar el contexto del asistente con la pregunta del cliente.
_EMBEDDING_CONTEXT_NEEDED = {
    'que tipos tienes', 'qué tipos tienes', 'que tipos hay', 'qué tipos hay',
    'cuales son', 'cuáles son', 'que opciones hay', 'qué opciones hay',
    'cuales son los tipos', 'cuáles son los tipos', 'que tipos existen', 'qué tipos existen',
    'cuentame mas', 'cuéntame más', 'y cuáles son', 'y cuales son',
    'dime más', 'dime mas', 'cuál es la diferencia', 'cual es la diferencia',
    'que más hay', 'qué más hay', 'cuantos tipos hay', 'cuántos tipos hay',
    'que significa', 'qué significa', 'y cuáles son los tipos', 'y cuales son los tipos',
    'qué diferencia hay', 'que diferencia hay', 'cuál es mejor', 'cual es mejor',
    'quiero activar', 'quiero activarlo', 'quiero activarla',
    'quiero desafiliarme', 'quiero cancelarlo', 'quiero cancelarla',
    'quiero desafiliarlo', 'quiero desafiliarla',
    'activar', 'desafiliar', 'cancelar',
    # Solicitudes de información sin producto explícito
    'dame los requisitos', 'dame los requisitos para tenerlo', 'dame los requisitos para tenerla',
    'quiero los requisitos', 'quiero saber los requisitos', 'quiero conocer los requisitos',
    'los requisitos', 'los requisitos para tenerlo', 'los requisitos para tenerla',
    'cuales son los requisitos', 'cuáles son los requisitos',
    'que requisitos necesito', 'qué requisitos necesito',
    'que necesito para tenerlo', 'qué necesito para tenerlo',
    'que necesito para tenerla', 'qué necesito para tenerla',
    'cuanto cuesta', 'cuánto cuesta', 'que costo tiene', 'qué costo tiene',
    'que cubre', 'qué cubre', 'que incluye', 'qué incluye',
    'cuales son las coberturas', 'cuáles son las coberturas',
    'cuales son las exclusiones', 'cuáles son las exclusiones',
    'dame mas informacion', 'dame más información',
    'mas informacion', 'más información', 'mas detalles', 'más detalles',
    # Expresiones de canal sin producto explícito
    'por llamada', 'quiero hacerlo por llamada', 'lo quiero por llamada',
    'por teléfono', 'por telefono', 'llamando', 'a través de llamada',
    'por banca por teléfono', 'por banca por telefono',
    'por la web', 'por internet', 'por la página', 'en la web',
    'por viabcp', 'en viabcp', 'a través de viabcp', 'por via bcp', 'en via bcp',
    'por la app', 'por la aplicación', 'desde la app', 'por banca móvil',
    'en agencia', 'en la agencia', 'presencialmente', 'en ventanilla',
    'quiero hacerlo por la web', 'quiero hacerlo por la app',
    'quiero hacerlo en agencia', 'lo hago por la web', 'lo hago por la app',
    'prefiero la web', 'prefiero la app', 'prefiero la agencia', 'prefiero llamar',
    # Apócopes de seguros sin contexto suficiente
    'tarjetas plus', 'proteccion de tarjetas', 'protección de tarjetas',
    'el vehicular', 'el soat', 'soat', 'el de viajes', 'de viajes',
    'el de salud', 'de salud', 'el de vida', 'de vida',
    'proteccion financiera', 'protección financiera', 'desgravamen',
    'sos salud', 's.o.s salud', 'onco respaldo', 'respaldo vida',
    'vida devolucion', 'vida devolución', 'seguro multiple', 'seguro múltiple',
}

# Frases con las que el cliente confirma que desea solicitar/adquirir un producto
# que el ASISTENTE acabó de describir o ofrecer.
# En estos casos usar el último mensaje del ASISTENTE como contexto del embedding
# para recuperar el proceso de solicitud del producto mencionado por el asistente.
_EMBEDDING_SOLICITUD_CONFIRMADA = {
    'si estoy interesada', 'sí estoy interesada', 'si estoy interesado', 'sí estoy interesado',
    'si eso quiero', 'sí eso quiero', 'si quiero', 'sí quiero',
    'si por favor', 'sí por favor', 'si porfa', 'sí porfa',
    'si me interesa', 'sí me interesa', 'si claro', 'sí claro',
    'quiero solicitarla', 'quiero solicitarlo', 'me interesa solicitarla',
    'quiero esa', 'quiero ese', 'la quiero', 'lo quiero',
    'dale', 'adelante', 'procéde', 'procede',
    'quiero obtenerlo', 'quiero obtenerla', 'quiero adquirirlo', 'quiero adquirirla',
    'quiero contratarlo', 'quiero contratarla', 'deseo adquirirlo', 'deseo adquirirla',
    'quiero comprarlo', 'quiero comprarla', 'sí lo quiero', 'si lo quiero',
    'sí la quiero', 'si la quiero', 'quiero aplicar', 'quiero acceder',
}

# Expresiones de CANAL sin producto explícito.
# Cuando el cliente dice «quiero hacerlo por llamada» sin nombrar el producto,
# se extrae el producto del último mensaje del ASISTENTE y se construye
# una query sintética: 'adquirir [producto] por teléfono'.
_EMBEDDING_CANAL = {
    # Teléfono / llamada
    'por llamada': 'por teléfono',
    'quiero hacerlo por llamada': 'por teléfono',
    'lo quiero por llamada': 'por teléfono',
    'por teléfono': 'por teléfono',
    'por telefono': 'por teléfono',
    'llamando': 'por teléfono',
    'a través de llamada': 'por teléfono',
    'por banca por teléfono': 'por teléfono',
    # Web / internet
    'por la web': 'por la web',
    'por internet': 'por internet',
    'por la página': 'por la web',
    'en la web': 'por la web',
    'por la página web': 'por la web',
    'por viabcp': 'por la web viabcp',
    'en viabcp': 'por la web viabcp',
    'a través de viabcp': 'por la web viabcp',
    'por via bcp': 'por la web viabcp',
    'en via bcp': 'por la web viabcp',
    # App / banca móvil
    'por la app': 'por banca móvil',
    'por la aplicación': 'por banca móvil',
    'por banca móvil': 'por banca móvil',
    'desde la app': 'por banca móvil',
    # Agencia / presencial
    'en agencia': 'en agencia bcp',
    'en la agencia': 'en agencia bcp',
    'presencialmente': 'en agencia bcp',
    'yendo a la agencia': 'en agencia bcp',
    'en ventanilla': 'en agencia bcp',
}

def get_query_for_embedding(conversation_history: str, model_llm) -> str:
    """
    Construye la query óptima para el embedding según el tipo de mensaje del cliente.

    Casos:
    - Mensaje con producto bancario explícito → usa solo el último mensaje del cliente.
    - Mensaje conversacional ("sí", "ok", "no") → usa el penúltimo mensaje del cliente.
    - Mensaje de desambiguación sin producto ("QUE TIPOS TIENES?", "¿cuáles son?") →
      combina el último mensaje del ASISTENTE + el último mensaje del CLIENTE para
      que el embedding capture el contexto completo y recupere chunks específicos
      (ej: tipos de Cuenta Mancomunada, no solo los más frecuentes).
    """
    

    # 1. Extraer último mensaje del CLIENTE
    ultimo_mensaje, matches = ultimo_mensaje_cliente(conversation_history)
    if ultimo_mensaje == conversation_history:
        return conversation_history

    # 2. Detectar si es una pregunta de desambiguación sin producto explícito
    ultimo_norm = _re.sub(r'[¿?¡!,.]', '', ultimo_mensaje.lower()).strip()
    if ultimo_norm in _EMBEDDING_CONTEXT_NEEDED:
        # Combinar contexto del asistente con la pregunta del cliente
        ultimo_asistente, _ = ultimo_mensaje_asistente(conversation_history)
        if ultimo_asistente:
            resultado = f"ASISTENTE: '{ultimo_asistente}' | CLIENTE: '{ultimo_mensaje}'"
            logging.info(f"{c_mage}🟪 Desambiguación → combinando contexto asistente+cliente: {c_yell}{resultado}{c_rst_}")
        else:
            # Sin mensaje del asistente, usar penúltimo del cliente
            mensajes_anteriores = [m.strip() for m in matches[:-1] if m.strip()]
            resultado = f"| CLIENTE: '{mensajes_anteriores[-1]}'" if mensajes_anteriores else conversation_history
            logging.info(f"{c_mage}🟪 Desambiguación sin asistente → usando penúltimo cliente: {c_yell}{resultado}{c_rst_}")
        return resultado

    # 2c. Cliente expresa un CANAL de acción sin producto explícito
    # Ej: 'Quiero hacerlo por llamada', 'por la web', 'por la app'
    # Extraer el producto del último mensaje del ASISTENTE y construir
    # query sintética: 'adquirir [producto] [canal normalizado]'
    canal_normalizado = _EMBEDDING_CANAL.get(ultimo_norm)
    if canal_normalizado:
        ultimo_asistente, _ = ultimo_mensaje_asistente(conversation_history)
        if ultimo_asistente:
            import re as _re2
            match_prod = _re2.match(
                r'(?:La |El |Los |Las )?((?:Tarjeta|Crédito|Cuenta|Seguro|Préstamo)[^.]+?)'
                r'(?:\s+(?:del BCP|BCP|es|te ofrece|permite|ofrece|incluye|tiene|está)|,)',
                ultimo_asistente
            )
            if match_prod:
                nombre_producto = match_prod.group(1).strip()
                resultado = f'adquirir {nombre_producto} {canal_normalizado}'
            else:
                resultado = f'{ultimo_asistente[:60]} {canal_normalizado}'
            logging.info(f"{c_mage}🟪 Canal sin producto → query sintética: {c_yell}{resultado}{c_rst_}")
            return resultado

    # 2b. Cliente confirma que desea solicitar el producto que el asistente ofreció
    # Ej: 'Si, estoy interesada', 'si porfa', 'si eso quiero'
    # El asistente describió BENEFICIOS del producto, no el proceso de solicitud.
    # Construir una query sintética 'solicitar [producto]' extrayendo el nombre
    # del producto del primer mensaje del asistente que lo menciona.
    if ultimo_norm in _EMBEDDING_SOLICITUD_CONFIRMADA:
        ultimo_asistente, _ = ultimo_mensaje_asistente(conversation_history)
        if ultimo_asistente:
            # Extraer el nombre del producto: está al inicio del mensaje del asistente
            # Ej: 'La Tarjeta de Crédito Visa Infinite Sapphire LATAM Pass te ofrece...'
            # Tomar solo las primeras palabras hasta el primer verbo/descripción
            import re as _re2
            # Buscar nombre de producto entre 'La ' y el primer verbo de descripción
            match_prod = _re2.match(
                r'(?:La |El |Los |Las )?(Tarjeta[^,]+?|Crédito[^,]+?|Cuenta[^,]+?|Seguro[^,]+?|Préstamo[^,]+?)'
                r'(?:\s+(?:es|te ofrece|permite|ofrece|incluye|tiene|está|es una|es un)|,)',
                ultimo_asistente
            )
            if match_prod:
                nombre_producto = match_prod.group(1).strip()
                # Las tarjetas de crédito comparten el mismo proceso de solicitud (URL0053).
                # Una query genérica recupera ese chunk mejor que una específica por modelo,
                # que atrae las fichas informativas de la tarjeta (URL del producto) en lugar
                # del flujo de solicitud.
                if 'Tarjeta de Crédito' in nombre_producto:
                    resultado = 'solicitar tarjeta de crédito BCP'
                else:
                    resultado = f'solicitar {nombre_producto}'
            else:
                # Fallback: usar las primeras 80 chars del asistente como contexto
                resultado = ultimo_asistente[:80]
            logging.info(f"{c_mage}🟪 Confirmación solicitud → query sintética: {c_yell}{resultado}{c_rst_}")
            return resultado

    # 3. Prompt para clasificar si menciona producto bancario
    prompt_clasificador = f"""Eres un clasificador para un banco.

Determina si el siguiente mensaje de un cliente está relacionado con temas bancarios o financieros.

Considera relacionado con temas bancarios CUALQUIER mensaje que involucre:
- Productos: cuentas, préstamos, tarjetas, seguros
- Operaciones: compras, retiros, transferencias, pagos, cobros, cargos, abonos
- Problemas: cargos no reconocidos, movimientos no autorizados, fraudes, devoluciones, reclamos
- Consultas: saldos, estados de cuenta, requisitos, beneficios

Considera NO relacionado SOLO si el mensaje es completamente ajeno a temas bancarios
(ej: "¿cómo está el clima?", "cuéntame un chiste", "hola", "sí", "ok", "gracias").

Mensaje del cliente: "{ultimo_mensaje}"

Responde únicamente con una de estas dos opciones, sin explicaciones:
MENCIONA_PRODUCTO
NO_MENCIONA_PRODUCTO"""

    clasificacion = model_llm.generate_text(prompt_clasificador).strip().upper()
    logging.info(f"{c_mage}🟪 Clasificación del mensaje: {c_yell}{clasificacion}{c_rst_}")

    # 4. Retornar según clasificación
    if "MENCIONA_PRODUCTO" in clasificacion:
        # Si el mensaje es corto (≤ 4 palabras) el cliente probablemente usa un nombre
        # informal o parcial del producto (ej: "la sapphire", "la signature", "el iridium").
        # En ese caso el embedding del mensaje solo es demasiado informal para matchear
        # los chunks formales de la KB → todos filtrados por L2 → alucinación.
        # Solución: combinar con el último mensaje del ASISTENTE para que el embedding
        # capture el nombre formal del producto, igual que hace _EMBEDDING_CONTEXT_NEEDED.
        if len(ultimo_mensaje.split()) <= 4:
            ultimo_asistente, _ = ultimo_mensaje_asistente(conversation_history)
            if ultimo_asistente:
                resultado = f"ASISTENTE: '{ultimo_asistente[:200]}' | CLIENTE: '{ultimo_mensaje}'"
                logging.info(f"{c_mage}🟪 Producto detectado (msg corto) → combinando contexto asistente+cliente: {c_yell}{resultado}{c_rst_}")
            else:
                resultado = ultimo_mensaje
                logging.info(f"{c_mage}🟪 Producto detectado (msg corto, sin asistente) → usando solo último mensaje: {c_yell}{resultado}{c_rst_}")
        else:
            resultado = f"| CLIENTE: '{ultimo_mensaje}'"
            logging.info(f"{c_mage}🟪 Producto detectado → usando solo último mensaje: {c_yell}{resultado}{c_rst_}")
    else:
        # Respuesta conversacional ("sí", "no", "ok"...).
        # El historial completo produce un embedding diluido con L2 alto → chunks descartados.
        # Usar el penúltimo mensaje del CLIENTE (el que sí mencionó el producto) da
        # un embedding enfocado y distancias L2 bajas → chunks recuperados correctamente.
        mensajes_anteriores = [m.strip() for m in matches[:-1] if m.strip()]
        if mensajes_anteriores:
            resultado = f"| CLIENTE: '{mensajes_anteriores[-1]}'"
            logging.info(f"{c_mage}🟪 Conversacional → usando penúltimo mensaje del CLIENTE: {c_yell}{resultado}{c_rst_}")
        else:
            resultado = conversation_history
            logging.info(f"{c_gree}* Conversacional sin historial previo → usando historial completo{c_rst_}")

    return resultado

# =========================
# PHASE 1 OPTIMIZATION: Cache Embedding Model at Startup
# =========================
# Feature flags
ENABLE_MODEL_CACHE = os.getenv('ENABLE_MODEL_CACHE', 'true').lower() == 'true'
TEST_MODE = os.getenv('TEST_MODE', 'false').lower() == 'true'  # Skip embedding generation for Mac testing
LOG_CHUNKS = os.getenv('LOG_CHUNKS', 'false').lower() == 'true'  # Log preview of retrieved KB chunks

if torch.cuda.is_available():
    EMBEDDING_DEVICE = 'cuda'
    print(f"{c_gree}🟩 Using CUDA GPU for embeddings{c_rst_}")
elif platform.system() == 'Darwin' and platform.machine() == 'arm64':
    # macOS ARM64: Disable model caching by default due to PyTorch stability issues
    if ENABLE_MODEL_CACHE:
        print(f"{c_yell}⚠️  macOS ARM64 detected with model caching enabled - may cause crashes!{c_rst_}")
        print(f"{c_yell}⚠️  Set ENABLE_MODEL_CACHE=false to use per-request loading{c_rst_}")
    EMBEDDING_DEVICE = 'cpu'
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
elif torch.backends.mps.is_available():
    EMBEDDING_DEVICE = 'mps'
    print(f"{c_gree}🟩 Using Apple Metal (MPS) for embeddings{c_rst_}")
else:
    EMBEDDING_DEVICE = 'cpu'
    print(f"{c_cyan}Using CPU for embeddings{c_rst_}")

# Initialize model cache variables
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
EMBEDDING_TOKENIZER = None
EMBEDDING_MODEL = None

if ENABLE_MODEL_CACHE:
    print(f"{c_cyan}Loading embedding model at startup on {EMBEDDING_DEVICE.upper()}...{c_rst_}")
    EMBEDDING_TOKENIZER = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    EMBEDDING_MODEL = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)
    EMBEDDING_MODEL = EMBEDDING_MODEL.to(EMBEDDING_DEVICE)
    EMBEDDING_MODEL.eval()  # Set to evaluation mode
    print(f"{c_gree}🟩 Embedding model cached successfully on {EMBEDDING_DEVICE.upper()}{c_rst_}")
else:
    print(f"{c_yell}⚠️  Model caching DISABLED - using per-request loading (slower but more stable){c_rst_}")

# Conexión a ElasticSearch
cert_content = base64.b64decode(os.environ["E_CERT"])
embeddings, chunk_data, sections, sec_seqs, doc_ids = {}, {}, {}, {}, {}

# Build SSL context in-memory to avoid filesystem permission issues on OpenShift
_ssl_ctx = ssl.create_default_context(cadata=cert_content.decode("utf-8"))
try:
    elastic_client = Elasticsearch([f'https://{E_HOST}:{E_PORT}'],
                                   basic_auth=(E_ADMIN_USER, E_ADMIN_PASSWORD),
                                   request_timeout=120,
                                   max_retries=3,
                                   ssl_context=_ssl_ctx)
    # Field names
    embedding_field = 'embedding'
    chunk_field = 'chunk'
    section_field = 'section'
    sec_seq_field = 'sec_seq'

    for producto, config in PRODUCTOS_MAP.items():
        if not elastic_client.indices.exists(index=config['es_index']):
            logging.info(f"{c_red_}Índice {config['es_index']} no disponible{c_rst_}")
            exit()
        else:
            (embeddings[producto],
            chunk_data[producto],
            sections[producto],
            sec_seqs[producto],
            doc_ids[producto]) = load_data_from_elasticsearch(
                config['es_index'], embedding_field, chunk_field, section_field, sec_seq_field
            )
            logging.info(f"{c_cyan}Índice para '{producto}': {c_yell}{config['es_index']} - Documentos cargados: {len(chunk_data[producto])}{c_rst_}")

    # =========================
    # PHASE 1 OPTIMIZATION: Pre-build FAISS indexes at startup
    # =========================
    print(f"{c_cyan}Pre-building FAISS indexes...{c_rst_}")
    FAISS_INDEXES = {}
    for producto, emb in embeddings.items():
        dimension = emb.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(emb)
        FAISS_INDEXES[producto] = index
        print(f"{c_gree}🟩 FAISS index for '{producto}' built ({len(emb)} vectors){c_rst_}")

except Exception as e:
    logging.info(f"{c_red_}Error de conexión ElasticSearch:\n{e}{c_rst_}")
    exit()

# Inicialización watsonx_ai con APIClient
WX_TIMEOUT = 120  # segundos
WX_MAX_CONNECTIONS = 10

if not IBM_CLOUD_URL or not API_KEY:
    raise RuntimeError("Faltan variables IBM_CLOUD_URL o API_KEY para inicializar watsonx_ai.")

wx_credentials = {"url": IBM_CLOUD_URL, "apikey": API_KEY}
wx_httpx_client = httpx.Client(
    timeout=WX_TIMEOUT,
    limits=httpx.Limits(max_connections=WX_MAX_CONNECTIONS)
)

try:
    # Firma para versiones recientes del SDK.
    client = APIClient(
        credentials=wx_credentials,
        project_id=PROJECT_ID or None,
        httpx_client=wx_httpx_client
    )
except TypeError:
    # Compatibilidad con versiones donde APIClient solo recibe credentials.
    client = APIClient(wx_credentials)
    if PROJECT_ID:
        client.set.default_project(PROJECT_ID)

API_KEY_SM = os.environ["API_KEY_SM"]
banking_router = BankingProductRouter(api_client=client, es_client=elastic_client)

@app.post("/get_rag_response")
async def get_rag_response(request: Request):
    api_key = request.headers.get("X-API-KEY")
    if api_key != API_KEY_SM:
        raise HTTPException(status_code=401, detail="Unauthorized")

    log_request_time()
    raw_body = await request.body()
    raw_text = raw_body.decode("utf-8", errors="ignore")
    clean_text = normalize_text(raw_text)
    try:
        data = json.loads(clean_text)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"JSON inválido: {e}")

    conversation_history = data.get("conversation_history", None)

    conversation_id = data.get("conversation_id", "")
    model_id = data.get("model_id", None)
    decoding_method = data.get("decoding_method", None)
    min_new_tokens = data.get("min_new_tokens", None)
    max_new_tokens = data.get("max_new_tokens", None)
    temperature = data.get("temperature", None)
    n_chunks = data.get("n_chunks", None)
    top_k = data.get("top_k", None)

    logging.info(f"{c_mage}* conversation_id: {c_yell}{conversation_id}{c_rst_}")
    logging.info(f"{c_mage}🟪 conversation_history: {c_yell}{conversation_history}{c_rst_}")
    nro_msg_cliente = contar_mensajes_cliente(conversation_history)
    logging.info(f"{c_cyan}🔢 Número de mensajes del CLIENTE: {c_red_}{nro_msg_cliente}{c_rst_}")

    #history_max_dupla_msgs = 2 # Si el número de mensajes del ASISTENTE-CLIENTE es mayor o igual a este valor, se aplicará reducción y compresión al historial de conversación. Si se pone en un número muy alto, no se aplicará reducción ni compresión al historial.
    #history_max_asistente_chars = 24 # Máximo de caracteres permitidos para cada mensaje del ASISTENTE VIRTUAL antes de aplicar compresión. Si un mensaje del asistente supera este límite, se comprime (resumiendo o eliminando partes) para reducir tokens. Si se pone en un número muy alto, no se aplicará compresión a los mensajes del asistente.
    #history_preserve_last_n_turns= 1 # En 1 mantiene el último mensaje del asistente sin comprimir, los mensajes anteriores del asistente si se comprimen si superan el límite de caracteres. Si se pone en 0, no se preserva ningún mensaje del asistente sin comprimir.

     # Si el historial de conversación tiene más mensajes del CLIENTE que el límite definido, se aplica reducción y compresión al historial de conversación antes de enviarlo a la IA Generativa.
    if nro_msg_cliente > history_max_dupla_msgs:
        print("\n")
        print(f"   {c_blue}📕 Mensajes del cliente en historial: {c_red_}>={nro_msg_cliente}{c_blue}. Duplas ASISTENTE-CLIENTE permitidas: {c_gree}{history_max_dupla_msgs}{c_blue}.{c_rst_}")
        print(f"   {c_gray}0. Historial {c_red_}ANTES{c_gray} de reducción ({c_red_}{len(conversation_history)}{c_gray} caracteres):\n   {c_rst_}{conversation_history}{c_rst_}")
        print("-------")
        print(f"   {c_blue}📒 Aplicando REDUCCIÓN de historial a {c_gree}{history_max_dupla_msgs}{c_blue} duplas ASISTENTE-CLIENTE...{c_rst_}")
        conversation_history = truncate_to_last_duplas(conversation_history, max_dupla_msgs = history_max_dupla_msgs)
        print(f"   {c_gray}1. Historial {c_yell}DESPUÉS{c_gray} de REDUCCIÓN ({c_yell}{len(conversation_history)}{c_gray} caracteres):\n   {c_rst_}{conversation_history}{c_rst_}")
        print("-------")
        print(f"   {c_blue}📗 Aplicando COMPRESIÓN de mensajes del ASISTENTE a {c_gree}{history_max_asistente_chars}{c_blue} caracteres, manteniendo los últimos {c_gree}{history_preserve_last_n_turns}{c_blue} mensajes intactos...{c_rst_}")
        conversation_history = compress_conversation_history(conversation_history, 
                                                             min_turns_to_compress = history_max_dupla_msgs, 
                                                             max_asistente_chars = history_max_asistente_chars, 
                                                             preserve_last_n_turns=history_preserve_last_n_turns)
        print(f"   {c_gray}2. Historial {c_gree}DESPUÉS{c_gray} de COMPRESIÓN ({c_gree}{len(conversation_history)}{c_gray} caracteres):\n   {c_rst_}{conversation_history}{c_rst_}")
        print("-------")
        print("\n")

    # Pre-analiza el último mensaje del CLIENTE antes de enviarlo a la IA Generativa.
    ultimo_msg_asistente, matches_a = ultimo_mensaje_asistente(conversation_history)
    logging.info(f"{c_cyan}🤖 Previo mensaje ASISTENTE: {c_yell}'{ultimo_msg_asistente}'{c_rst_}")

    ultimo_msg_cliente, matches_c = ultimo_mensaje_cliente(conversation_history)
    logging.info(f"{c_cyan}👩🏻‍🚀 Último mensaje CLIENTE: {c_yell}'{ultimo_msg_cliente}'{c_rst_}")

    intention = pre_review_before_generative_ai(ultimo_msg_cliente)
    logging.info(f"{c_cyan}🟦 PRE-Intención detectada: {c_yell}{intention}{c_rst_}")
    #------------------------------------------------------------------------

    # Si en el mensaje del cliente se detecta lisuras,solicitud de menú, saludos o mensaje emoji/número
    if intention in ['critical', 'deriva_menu', 'saludos', 'emoji_number']:
        first_contact_resolution = ""
        if intention == "critical":
            responseSemanticAnswer = critical_speech # Despues de insulto, pregunta intención
            first_contact_resolution = "FALSE" 
        elif intention == "deriva_menu":
            responseSemanticAnswer = intention # DERIVA A MENÚ
            first_contact_resolution = "FALSE"
        elif intention == "saludos":
            responseSemanticAnswer = saludo_speech # Despues de saludo, pregunta intención
            first_contact_resolution = "FALSE" 
        else: #intention == "emoji_number":
            responseSemanticAnswer = emoji_number_speech # Despues de emoji_number, pregunta intención
            first_contact_resolution = "FALSE" 
        
        dominio_producto_procesado = "pre_gen_ai_" + intention
        logging.info(f"{c_cyan}🟡 Requiere procesar GenAI: {c_yell}NO{c_rst_}\n")
        logging.info(f"{c_gree}🟡 dominio_producto proces: {c_yell}{dominio_producto_procesado}{c_rst_}")
        logging.info(f"{c_mage}🟣 responseSemanticAnswer:  {c_yell}{responseSemanticAnswer}{c_rst_}")
        logging.info(f"{c_mage}🟣 first_contact_resolution:{c_yell}{first_contact_resolution}{c_rst_}")
        
        return_response = {
            "dominio_producto": dominio_producto_procesado,
            #"dominio_producto" posible values: 
                # 'pre_gen_ai_critical',
                # 'pre_gen_ai_deriva_menu',
                # 'pre_gen_ai_saludos',
                # 'pre_gen_ai_emoji_number',

            "first_contact_resolution": first_contact_resolution,
            'responseSemanticAnswer': responseSemanticAnswer,
            "llm_complete_response": "",
            'llm_first_stream_time': "",
            'llm_total_time': 0,
            'semantic_search_time': 0,
            "llm_request_id": "",
            "llm_timestamp_start_time": "",
            "llm_timestamp_end_time": "",
            'top_ids': [],
            
        }
    
    else: # Si en el mensaje del cliente NO se detecta lisuras,solicitud de menú, saludos o mensaje emoji/número -> Deriva a Generativa
        #intention== "go_generative_ai":
        logging.info(f"{c_cyan}🟢 Requiere procesar GenAI: {c_gree}SI{c_rst_}\n")        
        dominio_producto = banking_router.identify_banking_product(conversation_history, model_id)

        # ── MULTI-DOMAIN FALLBACK ────────────────────────────────────────────
        # Si el clasificador devuelve 'none' y el mensaje del cliente contiene
        # keywords de múltiples dominios, ejecutar RAG en ambos índices y
        # combinar los chunks para que el LLM pueda responder con contexto completo.
        if dominio_producto not in PRODUCTOS_MAP:
            _DOMAIN_KEYWORDS = {
                "cuentas":   ["cuenta", "cuentas", "cts", "plazo fijo", "fondo mutuo",
                              "fondos mutuos", "tyba", "mancomunada", "sueldo", "ahorros",
                              "corriente", "digital", "ilimitada", "premio"],
                "tarjetas":  ["tarjeta", "tarjetas", "visa", "amex", "american express",
                              "débito", "debito", "crédito", "credito", "qore", "latam pass",
                              "iridium", "sapphire", "platinum", "signature", "infinite"],
                "prestamos": ["préstamo", "prestamo", "crédito personal", "hipotecario",
                              "vehicular", "mivivienda", "pyme", "compra de deuda"],
                "seguros":   ["seguro", "seguros", "cobertura", "siniestro", "póliza"],
            }
            _msg_lower = ultimo_msg_cliente.lower()
            _detected_domains = [
                d for d, kws in _DOMAIN_KEYWORDS.items()
                if any(kw in _msg_lower for kw in kws)
            ]
            if len(_detected_domains) >= 2:
                dominio_producto = "__multi__"
                logging.info(
                    f"{c_mage}🟪 [RAG] Multi-dominio detectado: {c_yell}{_detected_domains}{c_rst_}"
                )
        # ────────────────────────────────────────────────────────────────────

        decoding_method_input = (decoding_method or "greedy").lower()
        if decoding_method_input == "greedy":
            decoding_method_final = DecodingMethods.GREEDY
        else:
            decoding_method_final = DecodingMethods.SAMPLING

        model_params = {
            "decoding_method": decoding_method_final,
            "min_new_tokens": min_new_tokens,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k
        }

        model_llm = ModelInference(
            model_id=model_id,
            params=model_params,
            api_client=client,
            validate=False,
            persistent_connection=True
        )
        top_ids, sorted_chunks, responseSemanticAnswer, llm_first_stream_time, llm_total_time, semantic_search_time, \
        prompt, request_sample, first_contact_resolution, dominio_producto = generate_answer_from_chunks(conversation_history, dominio_producto, model_llm, n_chunks, \
                                                                                        min_new_tokens, max_new_tokens, model_id, conversation_id)
        # Limpiar respuesta del LLM
        raw_llm = request_sample.get("llm_complete_response", "")

        return_response = {
            "dominio_producto": dominio_producto,
                #"dominio_producto" posible values: 
                # 'cuentas',
                # 'tarjetas',
                # 'prestamos',
                # 'seguros',
                ##### 'none' (si no se detecta un producto específico) VERIFICAR SI SE USA O FUE REEMPLAZADO POR LOS POST_GEN_AI_XXX
                # "POST_GEN_AI_OUT_OF_DATA"
                # "POST_GEN_AI_NO_BANKING_CONVERSATION"
                # "POST_GEN_AI_DERIVA_MENU" -> Muy probable no se active porque la detección de "deriva_menu" se hace en el pre-review, pero se deja la opción por si se quiere usar también después del GenAI.
                # "POST_GEN_AI_TRANSFER_HUMAN"
                # "POST_GEN_AI_GUARDRAIL"
                # "POST_GEN_AI_FINISH"
                # "POST_GEN_AI_ATTACH_TRX_FLOW"
                # "POST_GEN_AI_CRITICAL" -> Muy probable no se active porque la detección de "critical" se hace en el pre-review, pero se deja la opción por si se quiere usar también después del GenAI.

            "first_contact_resolution": first_contact_resolution,
            'responseSemanticAnswer': responseSemanticAnswer,
            "llm_complete_response": raw_llm,
            'llm_first_stream_time': llm_first_stream_time,
            'llm_total_time': llm_total_time,
            'semantic_search_time': semantic_search_time,
            "llm_request_id": request_sample.get("llm_request_id"),
            "llm_timestamp_start_time": request_sample.get("llm_timestamp_start_time"),
            "llm_timestamp_end_time": request_sample.get("llm_timestamp_end_time"),
            'top_ids': top_ids
        }

        logging.info(f"{c_mage}🟣 responseSemanticAnswer: {c_cyan}{responseSemanticAnswer}{c_rst_}")
        print("\n")
    
    return return_response

# --------------------------- ASGI WRAPPER ---------------------------
# FastAPI is ASGI-native; expose it directly for Uvicorn/Gunicorn.
asgi_app = app

if __name__ == '__main__':
    #import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=5010, log_level='info')