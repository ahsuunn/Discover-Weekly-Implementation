import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

def load_matrix(filepath):
    """Muat dan siapkan matriks dari file CSV"""
    try:
        matrix = pd.read_csv(filepath)
        matrix = matrix.iloc[:, 1:].values
        num_users, num_items = matrix.shape
        
        print(f"Memuat matriks dengan {num_users} pengguna dan {num_items} item")
        return matrix, num_users, num_items
    
    except FileNotFoundError:
        print(f"Error: File {filepath} tidak ditemukan")
        raise
    except Exception as e:
        print(f"Error saat memuat matriks: {str(e)}")
        raise

def matrix_factorization(matrix, num_users, num_items, latent_features=2, 
                        learning_rate=0.001, regularization=0.05, num_iterations=100):
    """Faktorisasi matriks"""
    # Inisialisasi matriks pengguna dan item
    np.random.seed(5)
    user_matrix = np.random.rand(num_users, latent_features)
    item_matrix = np.random.rand(num_items, latent_features)
    
    print("Start faktorisasi matriks")
    
    for iteration in range(num_iterations):
        for i in range(num_users):
            for j in range(num_items):
                if matrix[i, j] > 0:  # Perbarui entri yang tidak nol
                    prediction = np.dot(user_matrix[i, :], item_matrix[j, :].T)
                    error = matrix[i, j] - prediction
                    
                    for k in range(latent_features):
                        user_matrix[i, k] += learning_rate * (2 * error * item_matrix[j, k] - 
                                                            regularization * user_matrix[i, k])
                        item_matrix[j, k] += learning_rate * (2 * error * user_matrix[i, k] - 
                                                            regularization * item_matrix[j, k])
        
        # Batasi nilai untuk menghindari overflow
        user_matrix = np.clip(user_matrix, -10, 10)
        item_matrix = np.clip(item_matrix, -10, 10)
        
        # Cetak progres
        if iteration % 10 == 0:
            total_error = np.sum((matrix - np.dot(user_matrix, item_matrix.T))**2)
            print(f"Iterasi {iteration}, Error: {total_error:.4f}")
    
    return user_matrix, item_matrix

class MusicTasteAnalyzer:
    def __init__(self, matrix, user_matrix, item_matrix):
        self.matrix = matrix
        self.user_matrix = user_matrix
        self.item_matrix = item_matrix
        self.cos_sim = cosine_similarity(user_matrix)
    
    def analyze_user_taste(self, user_id, top_n_similar=5):
        """Analisis selera musik pengguna dan beri rekomendasi"""
        user_history = self.matrix[user_id]
        listened_songs = np.where(user_history > 0)[0]
    
        similar_users = np.argsort(self.cos_sim[user_id])[::-1][1:top_n_similar+1]
    
        recommendations = self._get_recommendations(user_id, similar_users)
        
        return {
            'listened_songs': listened_songs,
            'similar_users': self._get_similar_users_info(user_id, similar_users),
            'recommendations': recommendations
        }
    
    def _get_similar_users_info(self, user_id, similar_users):
        """Mengambil info detail tentang pengguna yang mirip"""
        similar_users_info = []
        for user in similar_users:
            common_songs = np.where((self.matrix[user_id] > 0) & (self.matrix[user] > 0))[0]
            similar_users_info.append({
                'user_id': user,
                'similarity': self.cos_sim[user_id][user],
                'common_songs': len(common_songs),
                'common_songs_list': common_songs.tolist()
            })
        return similar_users_info
    
    def _get_recommendations(self, user_id, similar_users, top_n=10):
        """Rekomendasi dengan skor yang dinormalisasi"""
        unrated_songs = np.where(self.matrix[user_id] == 0)[0]
        recommendations = []
        
        for song in unrated_songs:
            total_similarity = 0
            weighted_sum = 0
            supporters = []
            
            for similar_user in similar_users:
                if self.matrix[similar_user][song] > 0:
                    similarity = self.cos_sim[user_id][similar_user]
                    rating = self.matrix[similar_user][song]
                    
                    # Akumulasi jumlah berbobot dan total kesamaan
                    weighted_sum += rating * similarity
                    total_similarity += similarity
                    
                    supporters.append({
                        'user_id': similar_user,
                        'rating': rating,
                        'similarity': similarity
                    })
            
            if supporters:
                # Normalisasi skor dengan membaginya dengan total kesamaan
                normalized_score = (weighted_sum / total_similarity) if total_similarity > 0 else 0
                
                recommendations.append({
                    'song_id': song,
                    'score': normalized_score,
                    'raw_score': weighted_sum,  # Simpan skor mentah untuk referensi
                    'num_supporters': len(supporters),
                    'supporters': supporters
                })
        
        # Urutkan berdasarkan skor yang dinormalisasi dan kembalikan top N
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        return recommendations[:top_n]