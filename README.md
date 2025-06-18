#  MindGuard: AI-Based Mental Health Monitoring through Text

MindGuard is an AI-powered system designed to detect early signs of mental health distress through user-generated text. Leveraging the power of NLP and deep learning, it classifies input into mental health risk levels (Low, Medium, High) and raises alerts for proactive support and intervention.



##  Features

-  Classifies mental health risk levels from text input using BERT
-  High accuracy with explainable AI (LIME)
-  Sends real-time alerts for Medium/High-risk texts
-  RESTful API for easy integration into other platforms
-  Scalable deployment using Docker and cloud services



##  Tech Stack

- **Language**: Python 3.10+
- **Framework**: FastAPI or Flask
- **ML Models**: BERT (via Hugging Face Transformers)
- **Libraries**: Scikit-learn, NLTK/spaCy, LIME, Pandas
- **Deployment**: Docker, AWS/GCP/Heroku
- **Database**: PostgreSQL or MongoDB (optional for logs)



##  Dataset (Suggested)

- [Reddit Mental Health Dataset (CLPsych)](https://zenodo.org/record/3609700)
- [DAIC-WoZ](https://dcapswoz.ict.usc.edu/)


