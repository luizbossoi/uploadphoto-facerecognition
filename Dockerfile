FROM python:3.8-slim


WORKDIR /app

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt
RUN mkdir uploads
RUN mkdir output

COPY . .

CMD [ "python3", "face_matching.py" ]