# Laporan Proyek Machine Learning - Dhi'fan Razaqa

## Project Overview
Industri perfilman merupakan salah satu sektor hiburan yang terus berkembang secara global dan berkontribusi besar terhadap ekonomi kreatif. Menurut laporan dari UNESCO Institute for Statistics, industri film global memproduksi lebih dari 7.000 film setiap tahun dengan pendapatan box office mencapai miliaran dolar \[1]. Dalam konteks tersebut, pemahaman terhadap preferensi audiens dan respons pasar menjadi sangat krusial. Hal ini dikarenakan persaingan yang semakin ketat di tengah perubahan perilaku konsumen, terutama dengan maraknya platform streaming digital seperti Netflix, Disney+, dan Amazon Prime Video.

Salah satu cara untuk memahami respons audiens terhadap suatu film adalah dengan menganalisis data penilaian (rating) dan ulasan pengguna. Rating film dari platform seperti IMDb menjadi indikator penting dalam menilai kualitas dan popularitas sebuah film. Penilaian ini tidak hanya mempengaruhi keputusan calon penonton, tetapi juga berdampak pada pendapatan film dan reputasi para sineas di industri.

Namun, dengan volume data yang sangat besar dan variatif, muncul tantangan dalam hal bagaimana memproses, menganalisis, dan mengambil wawasan yang bermakna dari data tersebut. Oleh karena itu, analisis data dan penerapan *data science* menjadi solusi strategis dalam membantu pelaku industri, mulai dari produser, sutradara, hingga platform distribusi, untuk mengambil keputusan yang lebih akurat dan berbasis data (*data-driven decision making*).

Dataset “Movie Rating” dari Kaggle menyediakan informasi penting seperti nama film, tahun rilis, genre, durasi, rating IMDb, dan jumlah suara. Melalui dataset ini, kita dapat mengeksplorasi berbagai pertanyaan, seperti bagaimana tren rating film dari waktu ke waktu, genre apa yang paling disukai penonton, atau apakah durasi film memengaruhi kepuasan audiens. Analisis semacam ini tidak hanya memberikan wawasan deskriptif, tetapi juga membuka peluang untuk membangun sistem rekomendasi film atau model prediktif yang mampu memperkirakan rating film sebelum dirilis.

Sebuah penelitian terdahulu menunjukkan bahwa penggunaan fitur konten seperti genre, durasi, dan metadata lainnya mampu meningkatkan performa sistem rekomendasi film \[2]. Dengan demikian, proyek ini juga relevan untuk mendukung inovasi teknologi dalam ekosistem perfilman digital.

### Referensi

[1] UNESCO Institute for Statistics, *The Global Film Industry in the Digital Age: Emerging Trends and Challenges*. [Online]. Available: http://uis.unesco.org. Accessed: May 19, 2025.

[2] Y. Deldjoo, M. Elahi, P. Cremonesi, and M. Quadrana, "Using visual features and side information for movie recommendation," *Int. J. Multimedia Inf. Retrieval*, vol. 5, no. 2, pp. 83–95, 2016, doi: 10.1007/s13735-016-0100-6.

## Business Understanding
### Problem Statements
- **Bagaimana cara menyajikan rekomendasi film yang relevan dan personal di tengah maraknya volume film dan perbedaan preferensi audiens?**  
Dengan lebih dari 7.000 film yang diproduksi setiap tahun dan dominasi platform streaming, pengguna dihadapkan pada terlalu banyak pilihan. Tanpa bantuan sistem cerdas, mereka kesulitan menemukan film yang sesuai dengan preferensi mereka secara efisien.

- **Bagaimana memanfaatkan data rating, genre, dan metadata film lainnya untuk membangun sistem rekomendasi yang akurat dan berbasis data?**  
Informasi dari rating IMDb, genre, durasi, dan jumlah suara (votes) dapat dimanfaatkan untuk membangun sistem yang mampu belajar dari pola penilaian dan minat pengguna.

- **Bagaimana mengintegrasikan pendekatan berbasis konten (content-based) dan perilaku pengguna (collaborative filtering) dalam sistem rekomendasi agar hasilnya lebih optimal?**
Kedua pendekatan memiliki kelebihan masing-masing, dan integrasi keduanya dapat membantu menangani masalah seperti cold-start serta meningkatkan akurasi rekomendasi.

### Goals
- Mengembangkan sistem rekomendasi film yang mampu menyaring dan menyajikan daftar film yang sesuai dengan selera pengguna secara otomatis.

- Memanfaatkan data rating, genre, dan metadata lainnya dalam model rekomendasi untuk memahami karakteristik film yang disukai pengguna.

- Membangun sistem rekomendasi hybrid yang mengombinasikan pendekatan content-based dan collaborative filtering untuk meningkatkan relevansi dan akurasi rekomendasi.

### Solution statements
- **Content-Based Filtering**  
Membangun sistem yang merekomendasikan film berdasarkan kemiripan fitur-fitur film yang telah disukai oleh pengguna. Kemiripan antar film dihitung menggunakan cosine similarity, lalu film-film dengan skor kemiripan tertinggi akan direkomendasikan.

- **Collaborative Filtering**  
Menerapkan teknik item-based collaborative filtering untuk menemukan film dengan pola rating yang mirip. Dengan pendekatan ini adalah pengguna bisa mendapatkan rekomendasi film yang disukai oleh pengguna lain dengan preferensi serupa.

## Data Understanding
Dataset yang digunakan dalam proyek ini berasal dari **Kaggle** dengan judul **"Movie Name and Review Dataset"**, yang dapat diakses melalui tautan berikut [Movie Name and Review Dataset – Kaggle](https://www.kaggle.com/datasets/meetnagadia/movie-rating)

Dataset ini terdiri dari dua file utama, yaitu:

1. `movies.csv` — berisi informasi metadata film seperti ID, judul, dan genre.
2. `ratings.csv` — berisi data penilaian (rating) dari pengguna terhadap film tertentu.

Kedua file ini dapat digabungkan melalui atribut **`movieid`**, yang berperan sebagai primary key pada tabel film dan foreign key pada tabel rating. Dataset ini cocok untuk membangun sistem rekomendasi karena mengandung dua elemen penting: **metadata film** dan **rating pengguna**. Kondisi dataset sudah dalam keadaan yang baik, tidak memiliki missing value dan data yang duplikat. Namun, ada genre yang bernilai "(no genres listed)" yang dapat dihapus untuk mendukung sistem rekomendasi lebih baik.


Dataset ini memiliki total:

* **`movies.csv`**: 10329 entri film dengan 3 kolom.
* **`ratings.csv`**: 105339 entri penilaian dari pengguna dengan 4 kolom.

### **Variabel pada `movies.csv`**

* **movieid** : ID unik untuk tiap film, digunakan sebagai penghubung dengan file `ratings.csv`.
* **title** : Judul film.
* **genres** : Daftar genre film yang dipisahkan oleh tanda pipe (`|`) seperti `Action|Adventure|Comedy`.

---

### **Variabel pada `ratings.csv`**

* **userid** : ID pengguna yang memberikan rating.
* **movieid** : ID film yang dinilai (relasi ke `movies.csv`).
* **rating** : Skor penilaian yang diberikan oleh pengguna terhadap film (skala 0.5 hingga 5).
* **timestamp** : Waktu saat rating diberikan dalam format UNIX timestamp.


### **Exploratory Data Analysis (EDA)**
* **Distribusi Jumlah Film per Genre**:  
   ![download](https://github.com/user-attachments/assets/ff0f7858-872b-4b48-b801-2126eba861ef)  
  *Insight:* Dapat dilihat genre Drama memiliki jumlah film terbanyak yang disusul dengan Comedy serta Thriller

* **Distribusi Jumlah Film dengan Rating Tertentu**:
  ![download](https://github.com/user-attachments/assets/cbab0874-f094-4164-af75-77b170e85497)  
  *Insight:* Rating 3 sampai dengan 4 mendominasi grafik, hal ini menunjukkan pengguna cenderung memberikan nilai tengah-tengah tidak terlalu baik maupun terlalu buruk

* **Distribusi Rata-rata Rating per Film**:
  ![download](https://github.com/user-attachments/assets/c9c3daa3-8af4-4829-9a07-e8cd53d2d7d6)  
  *Insight:* Rata-rata Rating 3 sampai dengan 4 mendominasi grafik, hal ini menunjukkan pengguna cenderung memberikan nilai tengah-tengah tidak terlalu baik maupun terlalu buruk

## Data Preparation
Proses data preparation yang digunakan untuk dataset ini tidak terlalu banyak. Hanya ada dua proses yang digunakan sebagai berikut:
- **Menghapus Genre yang Tidak Relevan**  
  Proses yang dilakukan pada tahap ini adalah menghapus kategori genre yang tidak relevan. Saat proses pemahaman data, ditemukan ada satu kategori genre yang tidak relevan bernilai "(no genres listed)". Hal ini dapat mengganggu sistem rekomendasi yang akan dibuat karena nanti hasil rekomendasinya bisa saja terpengaruh oleh genre yang tidak relevan sehingga genre tersebut perlu dihapus.
- **Mengubah Separator Untuk Genre**  
  Proses yang dilakukan pada tahap ini adalah mengubah separator yang digunakan untuk data genre yang tadinya "|" menjadi hanya spasi. Hal ini dilakukan agar TF-IDF Vectorizer dapat secara otomatis memisahkan kategori genre yang banyak menjadi hanya satu kata berdasarkan spasi sebagai pemisahnya.

## Modeling
### **Content Based Filtering**  
Content-based filtering adalah metode sistem rekomendasi yang menyarankan item (dalam hal ini film) berdasarkan kemiripan karakteristik kontennya. Pada project ini, pendekatan dilakukan dengan merepresentasikan fitur *genre* dari setiap film ke dalam bentuk vektor menggunakan **TF-IDF (Term Frequency-Inverse Document Frequency)**, yang menangkap seberapa penting kata (genre) tertentu dalam keseluruhan koleksi film. Kemudian, kemiripan antar film dihitung menggunakan **cosine similarity**, yaitu ukuran sejauh mana dua film memiliki genre yang serupa. Fungsi `content_based_recommendations()` akan menerima judul film sebagai input, menghitung kemiripan antara film tersebut dengan seluruh film lain, lalu mengembalikan daftar film dengan genre yang paling mirip. Dengan pendekatan ini, pengguna akan mendapatkan rekomendasi yang relevan berdasarkan preferensi film yang telah mereka sukai sebelumnya, tanpa memerlukan data dari pengguna lain.

#### **Kelebihan:**  
- **Tidak bergantung pada pengguna lain:** Sistem membuat rekomendasi hanya berdasarkan preferensi pengguna itu sendiri dan karakteristik konten film.
- **Efektif untuk pengguna baru:** Jika pengguna telah memberikan sedikit rating, sistem tetap bisa memberikan rekomendasi berdasarkan metadata film seperti genre.  

#### **Kekurangan:**
- **Rekomendasi cenderung terbatas dan serupa:** Karena fokus pada film yang mirip dengan yang telah disukai, sistem cenderung memberikan rekomendasi dengan genre atau fitur serupa, sehingga tidak memberikan pilihan lain yang lebih beragam untuk pengguna.

- **Keterbatasan fitur:** Dataset movies.csv hanya memiliki fitur dasar (title, genres). Tidak tersedia sinopsis, sutradara, atau aktor—yang membatasi konteks konten.

#### **Top-N Recommendation**  
**Showing recommendations for: Toy Story (1995)**  
Toy Story (1995)  Adventure|Animation|Children|Comedy|Fantasy  
**Top 10 movie recommendation**  
- Antz (1998)  
- Toy Story 2 (1999)  
- Adventures of Rocky and Bullwinkle, The (2000)  
- Emperor's New Groove, The (2000)  
- Monsters, Inc. (2001)  
- DuckTales: The Movie - Treasure of the Lost Lamp (1990)  
- Wild, The (2006)  
- Shrek the Third (2007)  
- Tale of Despereaux, The (2008)  
- Asterix and the Vikings (Astérix et les Vikings) (2006)  

### **Collaborative Filtering**  
Collaborative filtering adalah pendekatan sistem rekomendasi yang memanfaatkan pola interaksi antar pengguna dan item untuk memberikan rekomendasi, tanpa perlu mengetahui konten dari item tersebut. Pada project ini, collaborative filtering diimplementasikan menggunakan **neural network dengan embedding layer** melalui arsitektur `RecommenderNet`. Model ini memetakan setiap pengguna dan film ke dalam representasi vektor berdimensi tetap (**embedding**), lalu menghitung **dot product** antara vektor pengguna dan film sebagai estimasi kecocokan (rating). Nilai akhir dilewatkan ke fungsi aktivasi **sigmoid** agar berada dalam rentang 0–1. Model dilatih menggunakan fungsi loss *binary crossentropy* dan dioptimasi dengan Adam Optimizer. Pendekatan ini efektif karena mampu menangkap preferensi laten dari pengguna berdasarkan pola rating secara kolektif, bahkan ketika item-item yang direkomendasikan tidak mirip secara konten.

#### **Kelebihan:**  
- **Rekomendasi berdasarkan komunitas:** Menggunakan pola rating dari pengguna lain untuk memberikan rekomendasi yang lebih kontekstual, bahkan jika film tersebut memiliki genre yang berbeda.

- **Mampu menemukan “hidden gems”:** Film yang mungkin tidak serupa secara konten namun disukai oleh pengguna dengan preferensi yang sama akan tetap direkomendasikan.  

#### **Kekurangan:**
- **Cold start problem (pengguna/film baru):** Jika pengguna belum memberikan rating atau film belum pernah dirating, sistem sulit memberikan rekomendasi.

- **Kompleksitas komputasi:** Teknik seperti matrix factorization memerlukan proses komputasi yang lebih tinggi, terutama untuk dataset besar.

#### **Top-N Recommendation**  
**Showing recommendations for users: 451**  
**movie with high ratings from user**  
- Welcome to the Dollhouse (1995) : Comedy|Drama  
- Brother (2000) : Action|Crime|Thriller  
- Slumdog Millionaire (2008) : Crime|Drama|Romance  
- Milk (2008) : Drama  
- Guardians of the Galaxy (2014) : Action|Adventure|Sci-Fi  

**Top 10 movie recommendation**  
- Citizen Kane (1941) : Drama|Mystery  
- Cinema Paradiso (Nuovo cinema Paradiso) (1989) : Drama  
- Paths of Glory (1957) : Drama|War  
- Third Man, The (1949) : Film-Noir|Mystery|Thriller  
- Ran (1985) : Drama|War  
- Touch of Evil (1958) : Crime|Film-Noir|Thriller  
- Character (Karakter) (1997) : Drama  
- Boy in the Striped Pajamas, The (Boy in the Striped Pyjamas, The) (2008) : Drama|War  
- Cosmos (1980) : Documentary  
- Batman: Under the Red Hood (2010) : Action|Animation

## Evaluation
Metrik evaluasi yang digunakan dalam kasus ini adalah Root Mean Squared Error (RMSE). Metrik ini sangat sesuai digunakan dengan konteks data, problem statement, dan solusi yang diinginkan untuk kasus sistem rekomendasi film.
### Root Mean Squared Error
![image](https://raw.githubusercontent.com/dhifanrazaqa/Predictive_analysis/refs/heads/main/Screenshot%202025-04-04%20061425.png)    
Root Mean Squared Error (RMSE) adalah akar dari MSE, yang mengembalikan kesalahan ke skala asli variabel target sehingga lebih mudah diinterpretasikan dibandingkan MSE. Karena RMSE masih berbasis kuadrat, ia tetap sensitif terhadap outlier, tetapi lebih mudah dalam memahami seberapa besar rata-rata kesalahan prediksi dalam satuan yang sama dengan target. RMSE sering digunakan ketika model harus menangani variasi data yang besar dan dimana penalti untuk kesalahan yang lebih besar perlu diperhatikan.

### Hasil Evaluasi
Setelah proses evaluasi dilakukan, didapatkan hasil sebagai berikut:    
![download](https://github.com/user-attachments/assets/796355c2-a8fb-470f-987d-7b19e2c40758)  
Hasil **RMSE** pada **validation** bernilai **0.1951** dan pada **training** bernilai **0.1777** Terlihat bahwa hasil training masih menunjukkan **overfitting**, yang artinya model akan kesulitan untuk memprediksi data baru. Tetapi **overfitting** yang terjadi disini tidak terlalu parah dan model masih bisa memberikan prediksi. 

## **Evaluasi Dampak Model terhadap Business Understanding**

### **Apakah model sudah menjawab setiap problem statement?**
Model yang dibangun telah menjawab setiap problem statement secara relevan. Dengan menerapkan pendekatan content-based filtering, sistem mampu menyajikan rekomendasi film yang relevan berdasarkan preferensi pengguna individu, yang menjawab tantangan banyaknya pilihan film. Penggunaan metadata seperti genre dan rating dalam proses ekstraksi fitur menunjukkan pemanfaatan data secara efektif. Selain itu, penerapan collaborative filtering berbasis model juga memberikan personalisasi berdasarkan pola perilaku pengguna lain. Dengan demikian, integrasi kedua pendekatan (hybrid) telah menjawab kebutuhan untuk mengatasi cold-start dan meningkatkan akurasi rekomendasi secara menyeluruh.

### **Apakah model berhasil mencapai goals yang diharapkan?**
Model berhasil memenuhi sebagian besar tujuan yang ditetapkan. Sistem rekomendasi dapat secara otomatis menyaring dan menyarankan film yang sesuai dengan minat pengguna, memanfaatkan data rating, genre, dan informasi film lainnya untuk membangun pemahaman terhadap preferensi pengguna. Evaluasi menunjukkan bahwa meskipun terdapat indikasi overfitting kecil (RMSE training: 0.1777, validation: 0.1951), performa model masih tergolong baik dan mampu menghasilkan prediksi yang cukup akurat. Hal ini menunjukkan bahwa sistem dapat dijadikan dasar untuk menyusun daftar film yang relevan bagi pengguna dengan personalisasi yang memadai.

### **Apakah setiap solution statement berdampak?**
Ya, kedua pendekatan content-based dan collaborative filtering berkontribusi positif terhadap sistem rekomendasi secara keseluruhan. Content-based filtering berdampak dalam memberikan hasil rekomendasi yang dapat dijelaskan dan sesuai dengan preferensi film pengguna berdasarkan genre, sedangkan collaborative filtering memungkinkan sistem menangkap preferensi melalui pola rating gabungan antar pengguna. Meskipun collaborative filtering menunjukkan sedikit overfitting, dampaknya tetap signifikan karena mampu memberikan personalisasi dan rekomendasi yang sesuai. Secara keseluruhan, keduanya saling melengkapi dan memperkuat kinerja sistem dalam menghadirkan rekomendasi yang akurat dan relevan.

## Conclusion & Recommendation
Hasil akhir dari evaluasi adalah sebagai berikut:    
**RMSE: 0.1951**      

**Recommendation:**   
Berdasarkan hasil pembahasan, rekomendasi yang dapat diberikan adalah mencoba mengembangkan sistem rekomendasi film berbasis hybrid dengan menggabungkan content-based filtering dan collaborative filtering secara lebih terintegrasi untuk meningkatkan akurasi dan relevansi hasil. Untuk memperkaya fitur content-based, metadata seperti aktor, sutradara, atau sinopsis dapat ditambahkan dari sumber eksternal seperti IMDb atau TMDb. Mengingat masih terdapat overfitting ringan pada collaborative filtering, penggunaan teknik regulasi seperti dropout, early stopping, dan validasi silang dapat membantu meningkatkan generalisasi model. Selain itu, untuk menangani cold-start pengguna baru, sistem dapat mengandalkan rekomendasi awal menggunakan content based filtering. Evaluasi sistem juga perlu diperluas dengan metrik seperti Precision\@K dan Recall\@K untuk merefleksikan performa lebih baik. Melihat tingginya volume produksi film dan beragamnya preferensi penonton, pengembangan sistem rekomendasi yang mampu menyajikan film secara personal dan efisien dapat sangat membantu. Dengan mengintegrasikan pendekatan content-based dan collaborative filtering, pengguna dapat memperoleh saran film yang relevan baik berdasarkan genre favorit maupun pola perilaku pengguna lain. Pendekatan ini akan membantu mengatasi masalah kelebihan pilihan (information overload) dan meningkatkan kepuasan pengguna dalam menjelajahi platform film digital.

**---Ini adalah bagian akhir laporan---**
