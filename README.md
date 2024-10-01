# mekari_aaie_challenge

> video demo link: https://drive.google.com/file/d/1FzhZEBVjIlnEP0d8aFCqGePJ0__ZBMQX/view?usp=sharing

> The dataset can be accessed from: https://drive.google.com/file/d/1_xaRB6d2K_9-1dUmdU0GjtaqPO7uQnTM/view

> The program uses Google Gemini AI, LangChain, ChromaDB, and Streamlit

> The Mekari Associate AI Engineer Challenge pdf is a file that contains System High-Level Architecture and some screenshots of demo

> requirements.txt is a file containing libraries that needed to be install for the program to run

> The Jupiter Notebook contains the code in a notebook environment and include some explanation of the code (More easy to comprehend because of the sections)

> spotify_backend.py contains the backend part of the program that consist of:
-   Upload raw data using pandas library.
-   Feature engineering because of the sheer volume of data. Because of the consideration of relevancy and limited resources, I decided to cleanse the data
-   based on some of the columns, namely the 'review_timestamp' and 'review_likes' to limit the data and use the relatively insightful data only.
-   Embeddings, and vectorization of the document is performed here and the vector database is saved locally so it can be called for the streamlit app.
-   The process of backend and frontend is separated to reduce the time to initialize the streamlit.

> spotify_insight.py contains the frontend and intialize chatbot that consist of:
-   Calling the vector database that previously saved.
-   Retrieve the vector and built the prompt.
-   Calling of the streamlit to initialize UI by running locally.
-   Q&A Chain and Quality scoring for the response.

> The dataset path, API Key, and Google ADC may need to be rewritten based on your own credentials.
