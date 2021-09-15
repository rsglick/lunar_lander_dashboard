FROM python:3.7
RUN apt-get update && apt-get install -y --no-install-recommends \
       apt-utils \
       build-essential \
       curl \
       xvfb \
       ffmpeg \
       xorg-dev \
       libsdl2-dev \
       swig \
       cmake \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 8501
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
COPY app.py models ./
CMD streamlit run --server.port $PORT app.py