import numpy as np
import pandas as pd
from implementation import *

def create_test_data():
    """
    Membuat data uji dengan pola :
    - Pengguna 0 suka lagu rock (rating tinggi untuk lagu 0-4)
    - Pengguna 1 suka lagu rock (mirip dengan Pengguna 0)
    - Pengguna 2 suka lagu pop (rating tinggi untuk lagu 5-9)
    - Pengguna 3 memiliki selera campuran tetapi sedikit mirip dengan Pengguna 0
    - Pengguna 4 memiliki selera yang sangat berbeda
    """
    # Matriks kosong: 5 pengguna x 10 lagu
    matrix = np.zeros((5, 10))
    
    # Pengguna 0 (pengguna target) - Penggemar rock
    matrix[0] = [5, 5, 0, 5, 0, 1, 0, 1, 0, 1]  # Suka rock, tidak suka pop
    
    # Pengguna 1 - Mirip dengan Pengguna 0 (penggemar rock)
    matrix[1] = [5, 4, 5, 5, 5, 1, 1, 0, 1, 0]  # Selera sangat mirip dengan Pengguna 0
    
    # Pengguna 2 - Selera berbeda (penggemar pop)
    matrix[2] = [1, 1, 0, 1, 0, 5, 5, 5, 5, 5]  # Selera berlawanan dengan Pengguna 0
    
    # Pengguna 3 - Selera campuran tetapi cenderung ke rock
    matrix[3] = [4, 4, 0, 3, 0, 2, 3, 0, 2, 0]  # Agak mirip dengan Pengguna 0
    
    # Pengguna 4 - Selera sangat berbeda
    matrix[4] = [2, 1, 0, 1, 0, 5, 5, 5, 4, 5]  # Sangat berbeda dari Pengguna 0
    
    # Membuat metadata lagu
    songs = pd.DataFrame({
        'song_id': range(10),
        'genre': ['Rock', 'Rock', 'Rock', 'Rock', 'Rock', 
                 'Pop', 'Pop', 'Pop', 'Pop', 'Pop'],
        'title': [f'Lagu Rock {i+1}' if i < 5 else f'Lagu Pop {i-4}' 
                 for i in range(10)]
    })
    
    return matrix, songs

def verify_recommendations(matrix, user_matrix, item_matrix, analyzer, songs_df):
    """Menguji dan memverifikasi rekomendasi untuk Pengguna 0"""
    analysis = analyzer.analyze_user_taste(0)
    
    print("\n=== Verifikasi Uji ===")
    
    # 1. Verifikasi Kesamaan Pengguna
    print("\n1. Kesamaan Pengguna (seharusnya paling mirip dengan Pengguna 1, lalu Pengguna 3):")
    for user in analysis['similar_users']:
        print(f"Pengguna {user['user_id']}: Kesamaan = {user['similarity']:.3f}")
        print(f"Lagu yang disukai bersama (rating >= 4): {[songs_df.iloc[song]['title'] for song in user['common_songs_list'] if matrix[0][song] >= 4 and matrix[user['user_id']][song] >= 4]}")
    
    # 2. Verifikasi Rekomendasi Lagu
    print("\n2. Rekomendasi Lagu (seharusnya lebih memilih lagu rock yang belum dirating):")
    for rec in analysis['recommendations'][:5]:
        song_info = songs_df.iloc[rec['song_id']]
        print(f"\nDirekomendasikan: {song_info['title']} (Genre: {song_info['genre']})")
        print(f"Skor: {rec['score']:.3f}")
        print("Didukung oleh:")
        for supporter in rec['supporters']:
            print(f"- Pengguna {supporter['user_id']} memberi rating {supporter['rating']} "
                  f"(kesamaan: {supporter['similarity']:.3f})")
    
    # 3. Verifikasi Preferensi Pengguna 0
    user0_ratings = matrix[0]
    print("\n3. Preferensi Dikenal Pengguna 0:")
    liked_songs = [i for i, rating in enumerate(user0_ratings) if rating >= 4]
    disliked_songs = [i for i, rating in enumerate(user0_ratings) if 0 < rating <= 2]
    
    print("Lagu yang Sangat Disukai:")
    for song_id in liked_songs:
        print(f"- {songs_df.iloc[song_id]['title']} ({songs_df.iloc[song_id]['genre']}): {user0_ratings[song_id]}")
    
    print("\nLagu yang Tidak Disukai:")
    for song_id in disliked_songs:
        print(f"- {songs_df.iloc[song_id]['title']} ({songs_df.iloc[song_id]['genre']}): {user0_ratings[song_id]}")

def main():
    # Membuat data uji
    matrix, songs_df = create_test_data()
    print(matrix)

    # Menyimpan data uji ke CSV
    pd.DataFrame(matrix).to_csv('test_matrix.csv', index=True)
    
    # Mendapatkan dimensi matriks
    num_users, num_items = matrix.shape
    
    # Melakukan faktorisasi matriks
    user_matrix, item_matrix = matrix_factorization(matrix, num_users, num_items)
    
    print("User Matrix:")
    print(user_matrix)
    print()
    print("Item Matrix:")
    print(item_matrix)
    # Membuat analyzer
    analyzer = MusicTasteAnalyzer(matrix, user_matrix, item_matrix)
    
    # Memverifikasi rekomendasi
    verify_recommendations(matrix, user_matrix, item_matrix, analyzer, songs_df)

if __name__ == "__main__":
    main()