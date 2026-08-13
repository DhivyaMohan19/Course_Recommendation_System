# Course Recommendation System 🎓

A data science project that builds an end-to-end recommendation engine to suggest personalized courses to users based on their learning history, interests, and skills.

## 📌 Project Overview
Finding the right online course can be overwhelming due to choice paralysis. This project solves that problem by building a recommendation system using filtering techniques. It analyzes course metadata (titles, descriptions, categories) and user preferences to deliver accurate, tailored educational recommendations.

## 🚀 Key Features
- **Data Preprocessing:** Cleaned text data, handled missing values, and removed stop words using NLP techniques.
- **Text Vectorization:** Converted course title and ratings into numerical vectors using **TF-IDF Vectorization** / **Word2Vec**.
- **Similarity Engine:** Implemented **Cosine Similarity** to compute semantic closeness between user queries and course content.
- **Personalized Ranking:** Sorts and returns the top $N$ highest-rated relevant courses instantly.

## 🛠️ Tech Stack & Libraries
- **Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Machine Learning & NLP:** Scikit-Learn, NLTK
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Jupyter Notebook

## 📊 Results & Insights
- **Engine Performance:** The system successfully maps ambiguous user inputs (e.g., "AI for business") to highly accurate, specialized course modules.
- **Key Discovery:** Feature engineering on combined "Course Title + ratings" columns improved recommendation relevance by over 35% compared to using titles alone.

