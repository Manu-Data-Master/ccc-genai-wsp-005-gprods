"""
Módulo: banking_product_router
Autor: Manuel David Alcántara (IBM Corp.)
Correo: manuel.alcantara@pe.ibm.com

Descripción:
Librería para clasificar conversaciones bancarias en uno de los siguientes
productos: cuentas, préstamos, seguros, tarjetas o none.

La clasificación se basa en:
- Análisis del historial completo de conversación
- Mayor peso al último mensaje del cliente
- Contexto recuperado desde ElasticSearch (RAG)
- Inferencia usando modelo fundacional de IBM watsonx
"""

import os
import re
import base64
from typing import List, Dict

from dotenv import load_dotenv
from elasticsearch import Elasticsearch

from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods


class BankingProductRouter:
    """
    Clase principal para clasificar conversaciones bancarias.

    Attributes:
        es (Elasticsearch): Cliente de ElasticSearch
        indices (Dict[str, str]): Mapeo de dominios a índices
        doc_limit (int): Número máximo de documentos por índice
    """

    def __init__(self, api_client=None, es_client: Elasticsearch = None, doc_limit: int = 5):
        """
        Inicializa el router cargando configuraciones y conexiones.

        Args:
            api_client:             APIClient ya autenticado (de main.py). Si se omite,
                                    el router crea sus propias credenciales (modo standalone).
            es_client (Elasticsearch): Cliente ES ya inicializado (de main.py). Si se omite,
                                    el router crea su propia conexión con timeout reducido.
            doc_limit (int):        Límite de documentos a recuperar por índice
        """

        load_dotenv(override=True)

        self.doc_limit = doc_limit
        self._api_client = api_client
        self._model_cache: Dict[str, ModelInference] = {}

        # -----------------------------
        # CONFIG ELASTICSEARCH
        # -----------------------------
        # Si main.py pasa su propio cliente, reutilizarlo (un solo pool de conexiones).
        # Si no, crear uno propio con timeout corto adecuado para el router.
        if es_client is not None:
            self.es = es_client
        else:
            self.es = self._init_elasticsearch()

        self.indices = {
            "cuentas": os.getenv("E_INDEX_NAME_CTASV2"),
            "prestamos": os.getenv("E_INDEX_NAME_PRESTAMOS"),
            "seguros": os.getenv("E_INDEX_NAME_SEGUROS"),
            "tarjetas": os.getenv("E_INDEX_NAME_TARJETAS"),
        }

    # ======================================================
    # INICIALIZADORES
    # ======================================================

    def _init_elasticsearch(self) -> Elasticsearch:
        """
        Inicializa una conexión propia a ElasticSearch.
        Solo se usa en modo standalone (cuando main.py no pasa es_client).
        Usa timeout reducido porque el router no puede bloquear el request principal.

        Returns:
            Elasticsearch: Cliente configurado
        """
        try:
            cert_content = base64.b64decode(os.environ["E_CERT"])
            cert_path = "files/elasticsearch_cert.pem"

            os.makedirs("files", exist_ok=True)

            with open(cert_path, "wb") as f:
                f.write(cert_content)

            es = Elasticsearch(
                [f'https://{os.getenv("E_HOST")}:{os.getenv("E_PORT")}'],
                basic_auth=(os.getenv("E_ADMIN_USER"), os.getenv("E_ADMIN_PASSWORD")),
                request_timeout=10,   # ← timeout corto: el router no puede bloquear el request principal
                max_retries=1,        # ← un solo reintento; fallar rápido es preferible a colgar
                ca_certs=cert_path,
            )

            return es

        except Exception as e:
            raise ConnectionError(f"Error conectando a ElasticSearch: {e}")

    def _get_model(self, model_id: str) -> ModelInference:
        """
        Retorna el modelo de inferencia de watsonx, cacheado por model_id.

        Args:
            model_id (str): Identificador del modelo a utilizar

        Returns:
            ModelInference: Modelo configurado
        """

        if model_id in self._model_cache:
            return self._model_cache[model_id]

        params = {
            "decoding_method": DecodingMethods.GREEDY,
            "max_new_tokens": 10,
            "temperature": 0,
        }

        if self._api_client is not None:
            model = ModelInference(
                model_id=model_id,
                params=params,
                api_client=self._api_client,
                validate=False,
                persistent_connection=True,
            )
        else:
            model = ModelInference(
                model_id=model_id,
                params=params,
                credentials={"url": os.getenv("IBM_CLOUD_URL"), "apikey": os.getenv("API_KEY")},
                project_id=os.getenv("PROJECT_ID"),
            )

        self._model_cache[model_id] = model
        return model

    # ======================================================
    # MÉTODOS INTERNOS
    # ======================================================

    # Respuestas conversacionales que no identifican un dominio de producto
    _CONVERSATIONAL = {
        'sí', 'si', 'no', 'ok', 'dale', 'bueno', 'claro', 'listo',
        'gracias', 'perfecto', 'entendido', 'de acuerdo', 'esta bien',
        'está bien', 'ya', 'aja', 'ajá', 'por favor', 'adelante',
    }

    # Frases con las que el cliente confirma que desea solicitar el producto que el asistente
    # acaba de describir, sin mencionar el producto explícitamente.
    # En estos casos hay que recuperar el dominio del mensaje anterior significativo.
    _SOLICITUD_CONFIRMADA = {
        'si estoy interesada', 'sí estoy interesada', 'si estoy interesado', 'sí estoy interesado',
        'si eso quiero', 'sí eso quiero', 'si quiero', 'sí quiero',
        'si por favor', 'sí por favor', 'si porfa', 'sí porfa',
        'si me interesa', 'sí me interesa', 'si claro', 'sí claro',
        'quiero solicitarla', 'quiero solicitarlo', 'me interesa solicitarla',
        'quiero esa', 'quiero ese', 'la quiero', 'lo quiero',
        'adelante', 'procéde', 'procede',
    }

    # Mensajes de desambiguación sin contenido bancario: el cliente pide más información
    # sobre lo que el asistente ya mencionó, sin repetir el producto/dominio.
    # En estos casos hay que recuperar el dominio del mensaje anterior significativo.
    _AMBIGUOUS_NO_DOMAIN = {
        'que tipos tienes', 'qué tipos tienes', 'que tipos hay', 'qué tipos hay',
        'cuales son', 'cuáles son', 'que opciones hay', 'qué opciones hay',
        'cuales son los tipos', 'cuáles son los tipos', 'que tipos existen', 'qué tipos existen',
        'cuentame mas', 'cuéntame más', 'y cuáles son', 'y cuales son',
        'me puedes decir', 'dime más', 'dime mas', 'cuál es la diferencia', 'cual es la diferencia',
        'que más hay', 'qué más hay', 'cuantos tipos hay', 'cuántos tipos hay',
        'que significa', 'qué significa', 'y cuáles son los tipos', 'y cuales son los tipos',
        'qué diferencia hay', 'que diferencia hay', 'cuál es mejor', 'cual es mejor',
        'cuéntame más sobre eso', 'cuentame mas sobre eso', 'y esos cuáles son',
        # Frases "sobre [X]" sin acción: el cliente pide información del subproducto mencionado
        'sobre eso', 'sobre ese', 'sobre esa', 'sobre ellos', 'sobre ellas',
        'sobre el primero', 'sobre el segundo', 'sobre el tercero',
        'sobre la primera', 'sobre la segunda', 'sobre la tercera',
        'información sobre eso', 'informacion sobre eso',
        'más sobre eso', 'mas sobre eso', 'más sobre ese', 'mas sobre ese',
        # Frases de solicitud de información sin producto explícito
        'dame los requisitos', 'dame los requisitos para tenerlo', 'dame los requisitos para tenerla',
        'quiero los requisitos', 'quiero saber los requisitos', 'quiero conocer los requisitos',
        "quiero saber cuántas",
        'los requisitos', 'los requisitos para tenerlo', 'los requisitos para tenerla',
        'cuales son los requisitos', 'cuáles son los requisitos',
        'que requisitos necesito', 'qué requisitos necesito',
        'que necesito para tenerlo', 'qué necesito para tenerlo',
        'que necesito para tenerla', 'qué necesito para tenerla',
        'cuanto cuesta', 'cuánto cuesta', 'que costo tiene', 'qué costo tiene',
        'cual es su costo', 'cuál es su costo', 'cual es el costo', 'cuál es el costo',
        'cual es su precio', 'cuál es su precio', 'cual es el precio', 'cuál es el precio',
        'cuanto sale', 'cuánto sale', 'cuanto vale', 'cuánto vale',
        'cuanto es', 'cuánto es', 'cuanto me cuesta', 'cuánto me cuesta',
        'a cuanto sale', 'a cuánto sale', 'cuanto cobra', 'cuánto cobra',
        'su costo', 'el costo', 'el precio', 'su precio',
        'que cubre', 'qué cubre', 'que incluye', 'qué incluye',
        'cuales son las coberturas', 'cuáles son las coberturas', 'cuales son las tarifas', 'cuáles son las tarifas',
        # Frases de cobertura sin producto explícito
        'dime la cobertura del seguro', 'dime la cobertura', 'la cobertura del seguro',
        'cobertura del seguro', 'cuál es la cobertura', 'cual es la cobertura',
        'qué cobertura tiene', 'que cobertura tiene', 'qué cubre el seguro', 'que cubre el seguro',
        'qué coberturas tiene', 'que coberturas tiene', 'dime las coberturas',
        'dame la cobertura', 'dame las coberturas', 'cuáles son sus coberturas', 'cuales son sus coberturas',
        'cuales son las exclusiones', 'cuáles son las exclusiones',
        'dame mas informacion', 'dame más información',
        'mas informacion', 'más información', 'mas detalles', 'más detalles',
        # Acciones sin producto explícito
        'quiero activar', 'quiero activarlo', 'quiero activarla',
        'activar', 'desafiliar', 'cancelar',
        'quiero desafiliarme', 'quiero cancelarlo', 'quiero cancelarla',
        'quiero desafiliarlo', 'quiero desafiliarla',
        # Frases de solicitud/adquisición sin producto explícito
        'como lo solicito', 'cómo lo solicito', 'como la solicito', 'cómo la solicito',
        'como lo pido', 'cómo lo pido', 'como la pido', 'cómo la pido',
        'como lo adquiero', 'cómo lo adquiero', 'como la adquiero', 'cómo la adquiero',
        'como lo contrato', 'cómo lo contrato', 'como la contrato', 'cómo la contrato',
        'como solicito', 'cómo solicito', 'como adquiero', 'cómo adquiero',
        'como contrato', 'cómo contrato', 'como me afilio', 'cómo me afilio',
        'quiero solicitarlo', 'quiero solicitarla',
        'quiero adquirirlo', 'quiero adquirirla',
        'quiero contratarlo', 'quiero contratarla',
        'quiero obtenerlo', 'quiero obtenerla',
        'como lo obtengo', 'cómo lo obtengo', 'como la obtengo', 'cómo la obtengo',
        'pasos para solicitarlo', 'pasos para solicitarla',
        'pasos para adquirirlo', 'pasos para adquirirla',
        # Frases "sacar + pronombre" con pregunta de info (sin producto explícito)
        'quiero sacarlo que requisitos piden', 'quiero sacarla que requisitos piden',
        'quiero sacarlo cuales son los requisitos', 'quiero sacarla cuales son los requisitos',
        'quiero sacarlo qué requisitos piden', 'quiero sacarla qué requisitos piden',
        'quiero sacarlo que necesito', 'quiero sacarla que necesito',
        'quiero sacarlo qué necesito', 'quiero sacarla qué necesito',
        # Frases con pronombre + pregunta de requisitos/info sin producto
        'que requisitos piden', 'qué requisitos piden',
        'que requisitos me piden', 'qué requisitos me piden',
        'que me piden', 'qué me piden',
        'que piden para tenerlo', 'qué piden para tenerlo',
        'que piden para tenerla', 'qué piden para tenerla',
        'que documentos piden', 'qué documentos piden',
        'que necesito para sacarlo', 'qué necesito para sacarlo',
        'que necesito para sacarla', 'qué necesito para sacarla',
        'cuanto tengo que ganar', 'cuánto tengo que ganar',
        'cuanto debo ganar', 'cuánto debo ganar',
        # Expresiones de canal sin producto explícito
        'por llamada', 'quiero hacerlo por llamada', 'lo quiero por llamada',
        'por teléfono', 'por telefono', 'llamando', 'a través de llamada',
        'por banca por teléfono', 'por banca por telefono',
        'por la web', 'por internet', 'por la página', 'en la web', 'por la página web',
        'por viabcp', 'en viabcp', 'a través de viabcp', 'por via bcp', 'en via bcp',
        'por la app', 'por la aplicación', 'desde la app', 'por banca móvil',
        'en agencia', 'en la agencia', 'presencialmente', 'en ventanilla',
        'yendo a la agencia', 'en una agencia',
        'quiero hacerlo por la web', 'quiero hacerlo por la app',
        'quiero hacerlo en agencia', 'quiero hacerlo presencialmente',
        'lo hago por la web', 'lo hago por la app', 'lo hago en agencia',
        'prefiero la web', 'prefiero la app', 'prefiero la agencia', 'prefiero llamar',
    }

    def _get_last_client_message(self, conversation_history: str) -> str:
        """
        Extrae el último mensaje del cliente desde el historial para clasificar el dominio.

        Estrategia:
          1. Si el último mensaje es conversacional (ej. "sí", "ok") → fallback al anterior.
          2. Si el último mensaje es una desambiguación sin dominio (ej. "¿cuáles son?",
             "quiero activar") → combina el mensaje anterior significativo + el último,
             para capturar el contexto acumulado (ej. "Sobre mi seguro de tarjeta" +
             "tarjetas plus" → query más rica para el clasificador).
          3. Si el último mensaje es un subproducto de ≤ 2 palabras (ej. "tarjetas plus",
             "vida ahorro") → combina con el mensaje anterior significativo en lugar de
             reemplazar, para no perder el término del producto.
          4. Si ninguna condición aplica → devuelve el último mensaje tal cual.
        """
        matches = re.findall(r"CLIENTE:\s*'([^']+)'", conversation_history)
        if not matches:
            return ""

        ultimo = matches[-1].strip()
        # Normalizar: minúsculas y sin signos de puntuación para comparar
        ultimo_norm = re.sub(r'[¿?¡!,.]', '', ultimo.lower()).strip()

        es_conversacional = ultimo.lower() in self._CONVERSATIONAL
        es_ambiguo = (
            ultimo_norm in self._AMBIGUOUS_NO_DOMAIN
            or ultimo_norm in self._SOLICITUD_CONFIRMADA
            # Verificación por subcadena: detecta frases como "si, por banca móvil"
            # que CONTIENEN una entrada de _AMBIGUOUS_NO_DOMAIN ("por banca móvil")
            # pero no son exact match por llevar un prefijo conversacional ("si, ")
            or any(phrase in ultimo_norm for phrase in self._AMBIGUOUS_NO_DOMAIN if len(phrase) > 4)
        )
        # Mensaje que empieza con "sobre" + nombre de producto (≤4 palabras): es selección de subproducto
        es_sobre_producto = (
            ultimo_norm.startswith('sobre ')
            and len(ultimo.split()) <= 5
            and not es_conversacional
        )

        # Detecta frases del patrón "[verbo][lo/la] + [pregunta de info]" sin producto explícito.
        # Ej: "quiero sacarlo que requisitos piden", "quiero obtenerla cuánto cuesta"
        # El pronombre átono (lo/la/los/las) indica que el producto fue mencionado antes.
        _PRONOMBRE_ATONICO = re.compile(
            r'\b(sacar|obtener|pedir|solicitar|adquirir|contratar|tener|hacer|conseguir)'
            r'(lo|la|los|las)\b'
        )
        _PREGUNTA_INFO = re.compile(
            r'\b(requisito|document|necesito|piden|cuesta|cobran|incluye|cubre|'
            r'plazo|tasa|interes|interés|monto|cuota|costo|proceso|paso|tramite|trámite)\b'
        )
        es_verbo_pronombre_info = (
            bool(_PRONOMBRE_ATONICO.search(ultimo_norm))
            and bool(_PREGUNTA_INFO.search(ultimo_norm))
            and not es_conversacional
        )

        es_subproducto_corto = (
            (len(ultimo.split()) <= 2 or es_sobre_producto)
            and not es_conversacional
            and not es_ambiguo
        )

        # Caso 1: mensaje conversacional → fallback simple al anterior significativo
        if es_conversacional:
            for msg in reversed(matches[:-1]):
                msg_clean = msg.strip()
                msg_norm = re.sub(r'[¿?¡!,.]', '', msg_clean.lower()).strip()
                is_conv = msg_clean.lower() in self._CONVERSATIONAL
                is_amb = (
                    msg_norm in self._AMBIGUOUS_NO_DOMAIN
                    or any(phrase in msg_norm for phrase in self._AMBIGUOUS_NO_DOMAIN if len(phrase) > 4)
                )
                if not is_conv and not is_amb:
                    return msg_clean
            return ultimo

        # Caso 2, 3 y 4: ambiguo, subproducto corto, o verbo+pronombre+info → combinar con contexto anterior
        if es_ambiguo or es_subproducto_corto or es_verbo_pronombre_info:
            for msg in reversed(matches[:-1]):
                msg_clean = msg.strip()
                msg_norm = re.sub(r'[¿?¡!,.]', '', msg_clean.lower()).strip()
                if (
                    msg_clean.lower() not in self._CONVERSATIONAL
                    and msg_norm not in self._AMBIGUOUS_NO_DOMAIN
                    and msg_norm not in self._SOLICITUD_CONFIRMADA
                ):
                    # Combinar: contexto anterior + último mensaje para query más rica
                    return f"{msg_clean} {ultimo}"
            # Si no hay anterior significativo, devolver el último tal cual
            return ultimo

        return ultimo

    def _get_docs_from_index(self, index_name: str, query_text: str,
                            min_score: float = 1.0) -> List[str]:
        docs = []
        query = {
            "size": self.doc_limit,
            "min_score": min_score,          # ← descarta chunks con score bajo
            "query": {
                "match": {
                    "chunk": {
                        "query": query_text,
                        "operator": "or",    # ← más flexible que "and"
                    }
                }
            }
        }
        try:
            results = self.es.search(index=index_name, body=query)
            for hit in results["hits"]["hits"]:
                content = hit.get("_source", {}).get("chunk", "")
                if content:
                    docs.append(content)
        except Exception as e:
            # Un timeout/error en ES no debe interrumpir la clasificación.
            # El router continúa con contexto parcial o vacío.
            print(f"[WARNING] BankingProductRouter: error consultando índice '{index_name}': {type(e).__name__}: {e}")
        return docs

    def _build_context(self, query_text: str) -> str:
        context = ""

        for domain, index in self.indices.items():
            docs = self._get_docs_from_index(index, query_text)

            if not docs:
                continue

            context += f"\n### {domain.upper()} ###\n"
            for d in docs:
                context += f"- {d}\n"

        return context

    # ======================================================
    # MÉTODO PÚBLICO
    # ======================================================

    def identify_banking_product(self, conversation_history: str, model_id: str) -> str:
        """
        Clasifica una conversación en un producto bancario.

        Args:
            conversation_history (str): Historial completo de la conversación
            model_id (str): Identificador del modelo watsonx a utilizar

        Returns:
            str: Categoría detectada (cuentas, prestamos, seguros, tarjetas, none)
        """
        model = self._get_model(model_id)
        last_message = self._get_last_client_message(conversation_history)
        context = self._build_context(last_message)
        prompt = f"""
Eres un clasificador de productos bancarios.

OBJETIVO:
Clasificar la conversación en UNA sola categoría:
- cuentas
- prestamos
- seguros
- tarjetas
- none

REGLAS ESTRICTAS:
- SOLO usa el contexto si es relevante
- Si el contexto no coincide con el mensaje → ignóralo
- NO inventes productos
- NO mezcles categorías
- Responde SOLO con una palabra

EJEMPLOS:
# ── CUENTAS ──────────────────────────────────────────────
"abrir cuenta corriente" → cuentas
"abrir cuenta de ahorros con banca móvil" → cuentas
"adelanto de sueldo BCP" → cuentas
"anular cuenta de ahorros" → cuentas
"cancelar fondo mutuo" → cuentas
"cerrar cuenta a plazo fijo" → cuentas
"compra no reconocida en cuenta" → cuentas
"constancia de cuenta bancaria" → cuentas
"cuenta bloqueada" → cuentas
"cuenta de ahorros" → cuentas
"cuenta depósito a plazo fijo" → cuentas
"cuenta digital" → cuentas
"cuenta ilimitada" → cuentas
"cuenta mancomunada conjunta" → cuentas
"cuenta mancomunada indistinta" → cuentas
"cuenta mancomunada menor" → cuentas
"cuenta premio" → cuentas
"cuenta sueldo" → cuentas
"disposición de fondos de cuenta CTS" → cuentas
"estado de cuenta" → cuentas
"fondos mutuos" → cuentas
"invertir en TYBA" → cuentas
"marcaje de cuenta sueldo" → cuentas
"meses de depósito de CTS" → cuentas
"movimientos de cuenta" → cuentas
"número de CCI" → cuentas
"número de cuenta BCP" → cuentas
"saldo de cuenta" → cuentas
"vincular RUC a cuenta sueldo" → cuentas

# ── PRÉSTAMOS ─────────────────────────────────────────────
"crédito" → prestamos
"créditos" → prestamos
"amortizar crédito" → prestamos
"compra de deuda" → prestamos
"crédito hipotecario" → prestamos
"crédito mivivienda" → prestamos
"crédito personal" → prestamos
"crédito pyme" → prestamos
"crédito vehicular auto usado" → prestamos
"crédito vehicular inteligente" → prestamos
"crédito vehicular tradicional" → prestamos
"cuotas de mi crédito" → prestamos
"deuda de mi préstamo" → prestamos
"evaluación crediticia BCP" → prestamos
"fecha de pago de mi crédito" → prestamos
"pagar crédito por banca móvil" → prestamos
"cronograma actualizado" → prestamos
"TEA de un crédito" → prestamos
"renta de cuarta categoría para crédito" → prestamos
"renta de quinta categoría para crédito" → prestamos
"reprogramar deuda" → prestamos
"requisitos para préstamo hipotecario" → prestamos
"solicitar préstamo personal" → prestamos

# ── SEGUROS ──────────────────────────────────────────────
"activar cobertura del seguro SOS" → seguros
"activar cobertura seguro protección financiera" → seguros
"activar seguro de desgravamen" → seguros
"activar seguro de viaje" → seguros
"activar seguro de vida devolución" → seguros
"activar seguro vehicular" → seguros
"activar SOAT" → seguros
"activar siniestro onco respaldo" → seguros
"activar siniestro seguro múltiple" → seguros
"activar siniestro seguro respaldo vida" → seguros
"beneficiarios del seguro" → seguros
"cancelar seguro" → seguros
"cobertura del seguro" → seguros
"contratar seguro BCP" → seguros
"desafiliarme de seguro no solicitado" → seguros
"desafiliarme del seguro onco respaldo" → seguros
"edad máxima para contratar seguro BCP" → seguros
"estado de reclamo de seguro" → seguros
"estado de solicitud de activación de seguro" → seguros
"exclusiones del seguro" → seguros
"invalidez total y permanente" → seguros
"medios para contratar seguro en agencia BCP" → seguros
"modificar beneficiarios del seguro" → seguros
"prima del seguro" → seguros
"reembolso por seguro no solicitado" → seguros
"requisitos para contratar seguro BCP" → seguros
"seguro de desgravamen" → seguros
"seguro de protección financiera" → seguros
"seguro de viajes" → seguros
"seguro múltiple" → seguros
"seguro no reconocido en cuenta" → seguros
"seguro onco respaldo digital" → seguros
"seguro protección de tarjetas plus" → seguros
"seguro respaldo vida" → seguros
"seguro s.o.s salud" → seguros
"seguro vehicular" → seguros
"seguro vida ahorro" → seguros
"seguro vida devolución plus" → seguros
"siniestro" → seguros
"solicitar seguro por ventanilla BCP" → seguros
"SOAT para moto lineal" → seguros
"tarjetas plus" → seguros

# ── TARJETAS ─────────────────────────────────────────────
"acumular millas latam pass" → tarjetas
"activar tarjeta de crédito" → tarjetas
"anular tarjeta de crédito" → tarjetas
"bloquear tarjeta de crédito" → tarjetas
"ciclo de facturación" → tarjetas
"compras sin intereses con tarjeta" → tarjetas
"disposición de efectivo" → tarjetas
"efectivo preferente" → tarjetas
"fecha de pago de tarjeta" → tarjetas
"línea de crédito de tarjeta" → tarjetas
"membresía de la tarjeta" → tarjetas
"préstamo tarjetero" → tarjetas
"Qore BCP" → tarjetas
"reembolso por cobro duplicado" → tarjetas
"reponer tarjeta de débito" → tarjetas
"solicitar tarjeta adicional" → tarjetas
"solicitar tarjeta de crédito" → tarjetas
"tarjeta AMEX latam pass" → tarjetas
"tarjeta de crédito amex black latam pass" → tarjetas
"tarjeta de crédito amex clásica latam pass" → tarjetas
"tarjeta de crédito amex oro latam pass" → tarjetas
"tarjeta de crédito amex platinum latam pass" → tarjetas
"tarjeta de crédito visa clásica" → tarjetas
"tarjeta de crédito visa clásica latam pass" → tarjetas
"tarjeta de crédito visa clásica qore" → tarjetas
"tarjeta de crédito visa infinite iridium latam pass" → tarjetas
"tarjeta de crédito visa infinite qore" → tarjetas
"tarjeta de crédito visa infinite sapphire latam pass" → tarjetas
"tarjeta de crédito visa light" → tarjetas
"tarjeta de crédito visa oro latam pass" → tarjetas
"tarjeta de crédito visa oro qore" → tarjetas
"tarjeta de crédito visa platinum latam pass" → tarjetas
"tarjeta de crédito visa platinum qore" → tarjetas
"tarjeta de crédito visa signature latam pass" → tarjetas
"tarjeta de crédito visa signature qore" → tarjetas
"tarjeta de débito visa clásica" → tarjetas
"tarjeta de débito visa clásica latam pass" → tarjetas
"tarjeta de débito visa con diseño" → tarjetas
"tarjeta de débito visa oro" → tarjetas
"tarjeta de ahorros" → tarjetas
"costo de reposicion de la tarjeta de ahorros" → tarjetas
"reposicion de tarjeta de ahorros" → tarjetas
"reponer tarjeta de ahorros" → tarjetas
"tarjeta de la cuenta de ahorros" → tarjetas
"mi tarjeta de ahorros" → tarjetas

# ── NONE ─────────────────────────────────────────────────
"activar compras en el exterior" → none
"activar compras por internet" → none
"activar notificaciones del aplicativo" → none
"aumentar límites de transferencias" → none
"cambio de número de celular" → none
"canales para hacer giros" → none
"cancelar un giro" → none
"cambio a dólares" → none
"cambio a soles" → none
"chequera" → none
"clave de 6" → none
"clave para cobrar un giro" → none
"clave Token" → none
"cobro por envío de estado de cuenta físico" → none
"código de remesa" → none
"código de verificación de compra" → none
"código SWIFT del BCP" → none
"transferencia diferida" → none
"transferencia inmediata" → none
"comisión por transferencia diferida" → none
"comisión por transferencia inmediata" → none
"costo de depósitos interplaza" → none
"eliminar suscripción" → none
"empresas socias para remesas" → none
"giros" → none
"Para cambiar la clave" → none
"límite de retiro en cajeros automáticos" → none
"monto máximo de depósito mediante giro" → none
"monto máximo de envío por remesa" → none
"opciones para invertir dinero" → none
"remesa" → none
"tiempo para recoger un giro" → none
"interbank" → none
"bbva" → none
"santander" → none
"scotiabank" → none
"citibank" → none

---

ÚLTIMO MENSAJE:
{last_message}

---

CONTEXTO (puede estar incompleto):
{context}

---

RESPUESTA:
"""
        response = model.generate(prompt=prompt)
        return response["results"][0]["generated_text"].strip().lower()