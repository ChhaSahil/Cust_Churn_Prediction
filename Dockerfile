FROM python
RUN /usr/local/bin/python -m pip install --upgrade pip
WORKDIR /churn
COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
EXPOSE 8501
COPY . .
ENTRYPOINT ["streamlit", "run"]
CMD ["stream_app.py"]