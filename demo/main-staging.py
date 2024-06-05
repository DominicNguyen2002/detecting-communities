import pandas as pd
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from scipy.spatial.distance import jensenshannon

# Hàm tính độ tương hỗ
def compute_reciprocity_prob(row1, row2, epsilon=1e-5):
    followers1, user_tweet1 = row1['followers'], row1['user_tweet']
    followers2, user_tweet2 = row2['followers'], row2['user_tweet']

    # Tính toán xác suất tương tác qua lại
    J_followers = lambda x, y: abs(x * y) / abs(x + y + epsilon)
    J_user_tweet = lambda x, y: abs(x * y) / abs(x + y + epsilon)
    phi = -np.log(epsilon + J_followers(followers1, followers2) * J_user_tweet(user_tweet1, user_tweet2)) * (epsilon + J_followers(followers1, followers2) * J_user_tweet(user_tweet1, user_tweet2))
    prob = 1 / (1 + np.exp(phi))
    return prob

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

# Hàm áp dụng f_structural để tính toán các cặp người dùng có cấu trúc tương quan và không tương quan
def f_structural(data, t=0.5):
    Sr = set()
    Su = set()

    user_data = data[['user_id', 'followers', 'user_tweet']].drop_duplicates(subset=['user_id'])
    user_ids = user_data['user_id'].unique()

    for i in range(len(user_ids)):
        for j in range(i + 1, len(user_ids)):
            user1 = user_ids[i]
            user2 = user_ids[j]
            
            row1 = user_data[user_data['user_id'] == user1].iloc[0]
            row2 = user_data[user_data['user_id'] == user2].iloc[0]
            
            p_R = compute_reciprocity_prob(row1, row2)
            
            if p_R >= t:
                Sr.add((user1, user2))
            else:
                Su.add((user1, user2))

    # Lọc các user_id có trong Sr
    user_ids_in_Sr = set([user_id for user_pair in Sr for user_id in user_pair])
    texts_Sr = data[data['user_id'].isin(user_ids_in_Sr)][['user_id', 'text']].drop_duplicates()
    unique_posts_Sr = texts_Sr.drop_duplicates(subset=['user_id'])

    # Tạo ma trận kề dựa trên user_ids_in_Sr
    filtered_user_ids = list(user_ids_in_Sr)
    n = len(filtered_user_ids)
    adj_matrix = np.zeros((n, n))
    user_index = {user_id: index for index, user_id in enumerate(filtered_user_ids)}

    for (user1, user2) in Sr:
        if user1 in user_index and user2 in user_index:
            i, j = user_index[user1], user_index[user2]
            adj_matrix[i, j] = 1
            adj_matrix[j, i] = 1

    return unique_posts_Sr, Sr, Su, adj_matrix

# Hàm áp dụng f_textual để tính toán các cặp văn bản có liên quan và không liên quan
def f_textual(f_textual_data, t=0.5):
    Tr = set()
    Tu = set()

    Sr = f_textual_data['user_id']

    for v_i in Sr:
        texts_i = f_textual_data[f_textual_data['user_id'] == v_i]['text'].tolist()
        
        for v_j in Sr:
            if v_i != v_j:
                texts_j = f_textual_data[f_textual_data['user_id'] == v_j]['text'].tolist()
                
                # if compute_text_similarity_jsd(texts_i, texts_j) >= t:
                if compute_text_similarity(texts_i, texts_j) >= t:
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

def initialize_clusters(Sr, Tr, max_clusters):
    selected_users_Sr = random.sample(Sr, max_clusters)
    selected_users_Tr = random.sample(Tr, max_clusters)
    clusters = [{user} for user in selected_users_Sr + selected_users_Tr]
    return clusters

def compute_pairwise_similarities(users, affinity_matrix, user_index):
    similarities = []
    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            user1, user2 = users[i], users[j]
            idx1, idx2 = user_index[user1], user_index[user2]
            similarity = affinity_matrix[idx1, idx2]
            similarities.append((user1, user2, similarity))
    return similarities

def update_cluster_means(clusters, affinity_matrix, user_index):
    new_means = []
    for cluster in clusters:
        cluster_indices = [user_index[user] for user in cluster]
        cluster_affinities = affinity_matrix[cluster_indices][:, cluster_indices]
        mean_affinity = np.mean(cluster_affinities, axis=0)
        new_means.append(mean_affinity)
    return new_means

def detecting_communities(f_structural_data, t=0.5, max_clusters=10):

    unique_posts_Sr, Sr, Su, adj_matrix = f_structural(f_structural_data)

    Sr = unique_posts_Sr['user_id'].tolist()
    
    Tr, Tu, affinity_matrix = f_textual(unique_posts_Sr, t)
    
    # Bước 4: Khởi tạo các cụm
    clusters = []
    user_index = {user_id: index for index, user_id in enumerate(Sr)}  # Sửa đổi ở đây

    # Chọn ngẫu nhiên bốn nút từ Sr
    initial_nodes = random.sample(Sr, max_clusters)
    for i in range(max_clusters):
        clusters.append(set([initial_nodes[i]]))

    # Bước 5: Tính toán sự tương đồng cặp và tạo cụm
    for i in range(len(initial_nodes)):
        for j in range(i + 1, len(initial_nodes)):
            v_i = initial_nodes[i]
            v_j = initial_nodes[j]
            if affinity_matrix[user_index[v_i], user_index[v_j]] >= t:
                clusters.append(set([v_i, v_j]))
            else:
                clusters.append(set([v_i]))
                clusters.append(set([v_j]))

    
    # Bước 6: Lặp lại cho đến khi số lượng cụm vượt quá M
    while len(clusters) < max_clusters:
        new_clusters = []
        for cluster in clusters:
            cluster_nodes = list(cluster)
            for i in range(len(cluster_nodes)):
                for j in range(i + 1, len(cluster_nodes)):
                    v_i = cluster_nodes[i]
                    v_j = cluster_nodes[j]
                    if affinity_matrix[user_index[v_i], user_index[v_j]] >= t:
                        new_clusters.append(set([v_i, v_j]))
                    else:
                        new_clusters.append(set([v_i]))
                        new_clusters.append(set([v_j]))
        clusters = new_clusters

    # Bước 7: Gán các nút vào các cụm
    final_clusters = []
    for cluster in clusters:
        cluster_nodes = list(cluster)
        cluster_texts = [unique_posts_Sr[unique_posts_Sr['user_id'] == node]['text'].tolist() for node in cluster_nodes]
        
        final_clusters.append(cluster)
    
    return final_clusters

# Đọc dữ liệu từ file CSV
f_structural_data = pd.read_csv('data/tweets.csv', low_memory=False)

# Gọi hàm detecting_communities
local_communities = detecting_communities(f_structural_data)


# Hiển thị kết quả
print(local_communities)

# Vẽ đồ thị kết quả local_communities
def draw_communities_graph(communities):
    # Tạo đồ thị
    G = nx.Graph()

    # Màu cho các cộng đồng
    community_colors = [
    'skyblue', 'lightgreen', 'salmon', 'red', 'gold', 'purple', 'orange', 'pink', 'brown', 'cyan',
    'blue', 'green', 'yellow', 'magenta', 'lime', 'olive', 'navy', 'teal', 'aqua', 'maroon',
    'beige', 'chocolate', 'coral', 'crimson', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkkhaki',
    'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkturquoise',
    'darkviolet', 'deeppink', 'deepskyblue', 'dodgerblue', 'firebrick', 'forestgreen', 'fuchsia', 'goldenrod', 'greenyellow', 'hotpink'
]

    # Thêm các nút và cạnh
    for community in communities:
        nodes = list(community)
        G.add_nodes_from(nodes)
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                G.add_edge(nodes[i], nodes[j])

    pos = nx.spring_layout(G)  # Bố cục để vẽ

    # Vẽ các node theo cộng đồng
    plt.figure(figsize=(12, 12))
    for i, community in enumerate(communities):
        color = community_colors[i % len(community_colors)]  # Chọn màu cho cộng đồng
        nx.draw_networkx_nodes(G, pos, nodelist=community, node_color=color, node_size=500)
    
    # Vẽ các cạnh
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    
    # Vẽ các nhãn
    nx.draw_networkx_labels(G, pos, font_size=8, font_color='black', font_weight='bold')

    plt.title('Community Detection')
    plt.axis('off')
    plt.legend()
    plt.show()

# Vẽ đồ thị các cộng đồng
draw_communities_graph(local_communities)
