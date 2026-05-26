# Imagen base con soporte para CUDA y cuDNN
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Establecemos el directorio de trabajo
WORKDIR /app

# Instalar dependencias básicas
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Establece el alias para python3 como python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Actualizar pip
RUN pip install --upgrade pip

# PYTHON
RUN python --version

RUN mkdir cache
RUN chmod -R 777 /app/cache
RUN mkdir files
RUN chmod -R 777 /app/files

RUN pip install "numpy<2"

# Instalar faiss con soporte para GPU
RUN pip install --no-cache-dir faiss-gpu

# Instalar otras librerías
#RUN pip install fastapi uvicorn
RUN pip install fastapi uvicorn==0.40.0
RUN pip install python-dotenv
RUN pip install elasticsearch==8.17.2
#RUN pip install ibm-watsonx-ai
RUN pip install ibm-watsonx-ai==1.3.42
#RUN pip install flask==3.1.2 -> No se usa
RUN pip install pyarrow
RUN pip install colorama
#RUN pip install chardet
RUN pip install chardet==5.2.0
#RUN pip install flasgger -> No se usa
#RUN pip install flask-swagger-ui -> No se usa
RUN pip install prometheus-client==0.20.0
#RUN pip install gunicorn
RUN pip install gunicorn==23.0.0

# Instalar PyTorch con soporte para CUDA
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Instalar Hugging Face Transformers y Sentence Transformers
RUN pip install transformers==4.50.0 sentence-transformers

# Copiar el resto de los archivos de la aplicación al contenedor
COPY . .

# Exponer el puerto en el que la aplicación se ejecutará
EXPOSE 5010

# Asegura logs en tiempo real y visibilidad en contenedores
ENV PYTHONUNBUFFERED=1

# Comando para ejecutar FastAPI (ASGI nativo) con Gunicorn + Uvicorn workers
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:5010", "--workers", "4", "--timeout", "0", "--log-level", "info", "--access-logfile", "-", "--error-logfile", "-"]
