# syntax=docker/dockerfile:1
#
# Image de `ollama.cpp` : le service Python et le binaire `llama-server`.
#
# @spec docs/BACKLOG.md OC-090 « Conteneurisation dev / staging / prod »
# @spec docs/ollama.cpp-architecture.md §5.2 « Choix technologiques »
# @spec docs/DAT.md §2 « Services et processus », §8 « Stratégie de déploiement »
#
# `llama.cpp` est compilé depuis ses sources amont à une révision ÉPINGLÉE, jamais vendorée dans
# ce dépôt : `ollama.cpp` reste une couche au-dessus d'un upstream intact (mission §33). Changer
# de version du backend se fait en changeant `LLAMA_CPP_REF`, sans toucher au code du middleware.

# --- Étape 1 : compilation de llama-server ---------------------------------------------------------
FROM debian:bookworm-slim AS llama-build

ARG LLAMA_CPP_REPO=https://github.com/ggml-org/llama.cpp
# Révision auditée lors de la conception de ce middleware (cf. docs/ollama.cpp-architecture.md §0.3).
ARG LLAMA_CPP_REF=master
# Compilation CPU par défaut : l'image reste utilisable partout. Pour du GPU, reconstruire avec
# les options CUDA/ROCm de llama.cpp — les drapeaux passent par `CMAKE_EXTRA_ARGS`.
ARG CMAKE_EXTRA_ARGS=""

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
RUN git clone --depth 1 --branch "${LLAMA_CPP_REF}" "${LLAMA_CPP_REPO}" llama.cpp

# Compilation hors source : l'arbre amont n'est jamais modifié.
RUN cmake -S llama.cpp -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_TOOLS=ON \
        ${CMAKE_EXTRA_ARGS} \
    && cmake --build build --target llama-server -j"$(nproc)" \
    && find build -name 'llama-server' -type f -exec cp {} /usr/local/bin/llama-server \; \
    && find build -name '*.so' -exec cp {} /usr/local/lib/ \; || true

# --- Étape 2 : image de service --------------------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

# `libgomp1` est requis par les binaires OpenMP de ggml ; `libcurl` par le téléchargement intégré.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libcurl4 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=llama-build /usr/local/bin/llama-server /usr/local/bin/llama-server
COPY --from=llama-build /usr/local/lib/*.so /usr/local/lib/
RUN ldconfig

WORKDIR /app

# Les dépendances d'abord : la couche est réutilisée tant que `requirements.txt` ne change pas.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ollamacpp ./ollamacpp
COPY scripts ./scripts

# Le service n'a besoin d'aucun privilège : il n'écoute que sur son port et n'écrit que dans son
# répertoire de modèles.
RUN useradd --create-home --uid 10001 ollamacpp \
    && mkdir -p /models \
    && chown -R ollamacpp:ollamacpp /models /app
USER ollamacpp

ENV OLLAMACPP_MODELS=/models \
    OLLAMACPP_HOST=0.0.0.0 \
    OLLAMACPP_PORT=11434 \
    OLLAMACPP_LLAMA_SERVER_BIN=/usr/local/bin/llama-server \
    PYTHONUNBUFFERED=1

EXPOSE 11434
VOLUME ["/models"]

# `/api/version` est public et ne charge aucun modèle : c'est la sonde de vie la moins coûteuse.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:11434/api/version', timeout=4).status==200 else 1)"

CMD ["python", "-m", "ollamacpp"]
