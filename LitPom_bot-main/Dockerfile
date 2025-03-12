## app/Dockerfile

FROM python:3.10

WORKDIR /LitPom_bot-main

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python3", "LitPom_bot.py"]
