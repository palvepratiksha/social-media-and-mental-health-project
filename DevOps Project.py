#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install emoji')



# In[4]:


get_ipython().system('pip install TextBlob')


# In[5]:


import numpy as np
import pandas as pd
import re
import emoji
import string
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob


# In[29]:


df=pd.read_csv("Downloads/smmh.csv")


# In[30]:


df.columns


# In[31]:


new_column_names = {
    'Timestamp': 'timestamp',
    '1. What is your age?': 'age',
    '2. Gender': 'gender',
    '3. Relationship Status': 'relationship_status',
    '4. Occupation Status': 'occupation_status',
    '5. What type of organizations are you affiliated with?': 'affiliated_organizations',
    '6. Do you use social media?': 'use_social_media',
    '7. What social media platforms do you commonly use?': 'social_media_platforms',
    '8. What is the average time you spend on social media every day?': 'daily_social_media_time',
    '9. How often do you find yourself using Social media without a specific purpose?': 'frequency_social_media_no_purpose',
    '10. How often do you get distracted by Social media when you are busy doing something?': 'frequency_social_media_distracted',
    "11. Do you feel restless if you haven't used Social media in a while?": 'restless_without_social_media',
    '12. On a scale of 1 to 5, how easily distracted are you?': 'distractibility_scale',
    '13. On a scale of 1 to 5, how much are you bothered by worries?': 'worry_level_scale',
    '14. Do you find it difficult to concentrate on things?': 'difficulty_concentrating',
    '15. On a scale of 1-5, how often do you compare yourself to other successful people through the use of social media?': 'compare_to_successful_people_scale',
    '16. Following the previous question, how do you feel about these comparisons, generally speaking?': 'feelings_about_comparisons',
    '17. How often do you look to seek validation from features of social media?': 'frequency_seeking_validation',
    '18. How often do you feel depressed or down?': 'frequency_feeling_depressed',
    '19. On a scale of 1 to 5, how frequently does your interest in daily activities fluctuate?': 'interest_fluctuation_scale',
    '20. On a scale of 1 to 5, how often do you face issues regarding sleep?': 'sleep_issues_scale',
}

df=df.rename(columns=new_column_names)


# In[32]:


df


# In[33]:


# Download required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')


# In[34]:


# Define global stopwords and punctuation
stop_words = set(stopwords.words("english"))
exclude_punct = string.punctuation


# In[35]:


def remove_html(text):
    """Remove HTML tags from text."""
    return re.sub(r"<.*?>", "", text)





# In[36]:


def remove_urls(text):
    """Remove URLs from text."""
    return re.sub(r"(?:http|https|ftp)://\S+", "", text)


# In[37]:


def convert_emojis(text):
    """Convert emojis to text descriptions."""
    return emoji.demojize(text)


# In[38]:


def remove_punctuation(text):
    """Remove punctuation from text."""
    return text.translate(str.maketrans("", "", exclude_punct))


# In[39]:


def remove_digits(text):
    """Remove digits from text and replace with spaces."""
    return ''.join([char if char.isalpha() or char.isspace() else ' ' for char in text])


# In[40]:


def remove_stopwords(text):
    """Remove English stopwords."""
    words = nltk.word_tokenize(text)
    return ' '.join([word for word in words if word.lower() not in stop_words])


# In[41]:


def preprocess_text(text):
    """Apply all preprocessing steps to the input text."""
    text = str(text).lower()
    text = remove_html(text)
    text = remove_urls(text)
    text = convert_emojis(text)
    text = remove_punctuation(text)
    text = remove_digits(text)
    text = remove_stopwords(text)
    return text


# In[42]:


def get_sentiment(text):
    """Return sentiment category using TextBlob polarity score."""
    try:
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0:
            return "positive"
        elif polarity < 0:
            return "negative"
        else:
            return "neutral"
    except Exception as e:
        print(f"Error processing text: {text}")
        return "error"


# In[45]:


def main():
    # Load mental health and social media dataset
    df = pd.read_csv("Downloads/smmh.csv")

    # Handle missing values in text column
    text_col = '16. Following the previous question, how do you feel about these comparisons, generally speaking?'
    df[text_col] = df[text_col].fillna('')

    # Sample 100 entries
    df = df.sample(100).reset_index(drop=True)

    # Preprocess and analyze sentiment on textual responses
    df['cleaned_text'] = df[text_col].apply(preprocess_text)
    df['sentiment'] = df['cleaned_text'].apply(get_sentiment)

    # Print sample output
    print(df[[text_col, 'cleaned_text', 'sentiment']].head())


# In[46]:


if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




