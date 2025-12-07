"""
文本預處理工具
"""

import re
import string
import pandas as pd
from typing import List, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# 下載必要的 NLTK 資料（如果尚未下載）
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


def clean_text(text: str, remove_stopwords: bool = False, language: str = 'english') -> str:
    """
    清理文本
    
    Args:
        text: 原始文本
        remove_stopwords: 是否移除停用詞
        language: 語言（'english' 或 'chinese'）
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # 移除多餘空白
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # 移除停用詞（如果需要）
    if remove_stopwords:
        if language == 'english':
            stop_words = set(stopwords.words('english'))
            tokens = word_tokenize(text)
            text = ' '.join([token for token in tokens if token.lower() not in stop_words])
    
    return text


def tokenize_text(text: str, language: str = 'english') -> List[str]:
    """分詞"""
    if pd.isna(text):
        return []
    
    text = str(text)
    if language == 'english':
        return word_tokenize(text)
    else:
        # 中文分詞（簡單版本，可替換為 jieba）
        return list(text)


def split_sentences(text: str, language: str = 'english') -> List[str]:
    """句子分割"""
    if pd.isna(text):
        return []
    
    text = str(text)
    if language == 'english':
        return sent_tokenize(text)
    else:
        # 中文句子分割（簡單版本）
        return re.split(r'[。！？\n]', text)


def preprocess_dataframe(
    df: pd.DataFrame,
    text_column: str = "text",
    remove_stopwords: bool = False
) -> pd.DataFrame:
    """批量預處理 DataFrame"""
    df = df.copy()
    df[text_column] = df[text_column].apply(
        lambda x: clean_text(x, remove_stopwords=remove_stopwords)
    )
    return df

