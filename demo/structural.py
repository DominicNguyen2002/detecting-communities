import pandas as pd
import numpy as np

def compute_reciprocity_prob(row1, row2, epsilon=1e-5):
    followers1, friends1 = row1['followers'], row1['friends']
    followers2, friends2 = row2['followers'], row2['friends']

    J_followers = lambda x, y: abs(x * y) / abs(x + y + epsilon)
    J_friends = lambda x, y: abs(x * y) / abs(x + y + epsilon)
    phi = -np.log(epsilon + J_followers(followers1, followers2) * J_friends(friends1, friends2)) * (epsilon + J_followers(followers1, followers2) * J_friends(friends1, friends2))
    prob = 1 / (1 + np.exp(phi))
    return prob

def f_structural(data, tau=0.5):
    Sr = set()
    Su = set()

    user_data = data[['user_id', 'followers', 'friends']].drop_duplicates(subset=['user_id'])
    user_ids = user_data['user_id'].unique()

    for i in range(len(user_ids)):
        for j in range(i + 1, len(user_ids)):
            user1 = user_ids[i]
            user2 = user_ids[j]
            
            row1 = user_data[user_data['user_id'] == user1].iloc[0]
            row2 = user_data[user_data['user_id'] == user2].iloc[0]
            
            p_R = compute_reciprocity_prob(row1, row2)
            
            if p_R >= tau:
                Sr.add((user1, user2))
            else:
                Su.add((user1, user2))

    # Lọc các user_id có trong Sr
    user_ids_in_Sr = set([user_id for user_pair in Sr for user_id in user_pair])
    post_texts_Sr = data[data['user_id'].isin(user_ids_in_Sr)][['user_id', 'post_text']].drop_duplicates()
    unique_posts_Sr = post_texts_Sr.drop_duplicates(subset=['user_id'])

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

# Đọc dữ liệu từ file CSV
data = pd.read_csv('data/Mental-Health-Twitter.csv', low_memory=False)

# Gọi hàm f_structural
unique_posts_Sr, Sr, Su, adj_matrix = f_structural(data)

# Hiển thị kết quả
print('unique post',unique_posts_Sr)
print('\nCác cặp tương đồng cấu trúc Sr',Sr)
print('\n\nCác cặp không tương đồng cấu trúc Su',Su)
print("\nMa trận kề (adjacency matrix):")
print(adj_matrix.shape)

# Lưu kết quả vào file CSV
unique_posts_Sr.to_csv('results/Sr_data_with_post_text.csv', index=False)
# Lưu kết quả vào file CSV
np.savetxt('results/adj_matrix.csv', adj_matrix, delimiter=',')
