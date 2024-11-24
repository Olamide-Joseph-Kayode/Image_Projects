FROM python:3.9.20
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD streamlit run dog_cat_app.py
