import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from scipy.spatial.distance import jensenshannon

def compute_text_similarity(texts_i, texts_j, top_n=10):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=top_n)
    
    # Vectorize the texts
    X_i = vectorizer.fit_transform(texts_i).toarray()
    X_j = vectorizer.transform(texts_j).toarray()
    
    # Compute the mean vectors
    m_i = np.mean(X_i, axis=0)
    m_j = np.mean(X_j, axis=0)
    
    # Apply LDA on the mean vectors
    lda = LatentDirichletAllocation(n_components=1, random_state=42)
    lda_m_i = lda.fit_transform([m_i])
    lda_m_j = lda.transform([m_j])
    
    # Compute the Jensen-Shannon distance
    sim = 1 - jensenshannon(lda_m_i[0], lda_m_j[0])
    return sim

def f_textual(textual_data, tau=0.5):
    Tr = set()
    Tu = set()

    Sr = textual_data['user_id']

    for v_i in Sr:
        texts_i = textual_data[textual_data['user_id'] == v_i]['post_text'].tolist()
        
        for v_j in Sr:
            if v_i != v_j:
                texts_j = textual_data[textual_data['user_id'] == v_j]['post_text'].tolist()
                
                if compute_text_similarity(texts_i, texts_j) >= tau:
                    Tr.add((v_i, v_j))
                else:
                    Tu.add((v_i, v_j))
    
    # Create the affinity matrix
    user_ids = list(Sr)
    n = len(user_ids)
    affinity_matrix = np.zeros((n, n))
    user_index = {user_id: index for index, user_id in enumerate(user_ids)}
    
    for (v_i, v_j) in Tr:
        i, j = user_index[v_i], user_index[v_j]
        affinity_matrix[i, j] = 1
        affinity_matrix[j, i] = 1

    return Tr, Tu, affinity_matrix


textual_data = pd.read_csv('results/Sr_data_with_post_text.csv', low_memory=False)

print(textual_data)

# Gọi hàm f_textual
Tr, Tu, affinity_matrix = f_textual(textual_data)

# Hiển thị kết quả
print("Các cặp tương đồng văn bản Tr: ", Tr)
print("\n\nCác cập không tương đồng văn bản Tu: ", Tu)
print("\nMa trận tương đồng (affinity matrix):")
print(affinity_matrix.shape)

# Lưu kết quả vào file CSV
np.savetxt('results/affinity_matrix.csv', affinity_matrix, delimiter=',')




