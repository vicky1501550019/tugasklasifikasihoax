library(caret)
library(tm)
library(SnowballC)
library(arm)
# Training data.
data <- c('Putusan MAHKAMAH AGUNG Nomor 2183 K/Pdt/2017 Tahun 2018','Putusan MAHKAMAH AGUNG Nomor 2829 K/Pdt/2017 Tahun 2017',
          'Putusan MAHKAMAH AGUNG Nomor 1553 K/Pdt/2017 Tahun 2017','Putusan MAHKAMAH AGUNG Nomor 1447 K/Pdt/2017 Tahun 2017',
          'Putusan MAHKAMAH AGUNG Nomor 1443 K/Pdt/2017 Tahun 2017','Putusan MAHKAMAH AGUNG Nomor 1442 K/Pdt/2017 Tahun 2017',
          'Putusan MAHKAMAH AGUNG Nomor 1441 K/Pdt/2017 Tahun 2017','Putusan MAHKAMAH AGUNG Nomor 1410 K/Pdt/2017 Tahun 2017',
          'Putusan MAHKAMAH AGUNG Nomor 1485 K/Pdt/2016 Tahun 2016','Putusan MAHKAMAH AGUNG Nomor 1403 K/Pdt/2017 Tahun 2017',
          'Putusan MAHKAMAH AGUNG Nomor 3406 K/Pdt/2017 Tahun 2017','Putusan MAHKAMAH AGUNG Nomor 1951 K/Pdt/2017 Tahun 2017',
          'Anggota: Tujuan Awal MCA Bela Ulama Saat Pilkada DKI Jakarta','Kapolri Tak Nyaman Ada Frasa 'Muslim' di Grup Penyebar Hoax MCA',
          'Kunci Setang Motor ke Kanan, Cara Ampuh Cegah Motor Dicuri?','Ramai Bocah Sebatang Kara Huni Gubuk Reyot di Karawang, Masa Sih?',
          'Beredar Broadcast WhatsApp hingga Facebook Bakal Dipantau, Apa Iya?','Salah Kaprah Pesan Berantai Tentang Manfaat Petai',
          'Pesan Berantai Kecelakaan Tewaskan 130 Jemaah Umrah, Ini Faktanya','Geger Motor 'Hantu' Terparkir 2 Tahun di Jalan dan Tak Bisa Dipindah',
          'Sendok Disebut Bisa Digunakan untuk Check Up Medis')
corpus <- VCorpus(VectorSource(data))

# Create a document term matrix.
tdm <- DocumentTermMatrix(corpus, list(removePunctuation = TRUE, stopwords = TRUE, stemming = TRUE, removeNumbers = TRUE))

# Convert to a data.frame for training and assign a classification (factor) to each document.
train <- as.matrix(tdm)
train <- cbind(train, c(0, 1))
colnames(train)[ncol(train)] <- 'y'
train <- as.data.frame(train)
train$y <- as.factor(train$y)
data
train
# Train.
fit <- train(y ~ ., data = train, method = 'bayesglm')

# Check accuracy on training.
predict(fit, newdata = train)

# Test data.
data2 <- c('Putusan MAHKAMAH AGUNG Nomor 2183 K/Pdt/2017 Tahun 2018','Putusan MAHKAMAH AGUNG Nomor 2829 K/Pdt/2017 Tahun 2017',
           'Putusan MAHKAMAH AGUNG Nomor 1553 K/Pdt/2017 Tahun 2017','Putusan MAHKAMAH AGUNG Nomor 1447 K/Pdt/2017 Tahun 2017',
           'Putusan MAHKAMAH AGUNG Nomor 1443 K/Pdt/2017 Tahun 2017','Putusan MAHKAMAH AGUNG Nomor 1442 K/Pdt/2017 Tahun 2017',
           'Putusan MAHKAMAH AGUNG Nomor 1441 K/Pdt/2017 Tahun 2017','Putusan MAHKAMAH AGUNG Nomor 1410 K/Pdt/2017 Tahun 2017',
           'Putusan MAHKAMAH AGUNG Nomor 1485 K/Pdt/2016 Tahun 2016','Putusan MAHKAMAH AGUNG Nomor 1403 K/Pdt/2017 Tahun 2017',
           'Putusan MAHKAMAH AGUNG Nomor 3406 K/Pdt/2017 Tahun 2017','Putusan MAHKAMAH AGUNG Nomor 1951 K/Pdt/2017 Tahun 2017',)
corpus <- VCorpus(VectorSource(data2))
tdm <- DocumentTermMatrix(corpus, control = list(dictionary = Terms(tdm), removePunctuation = TRUE, stopwords = TRUE, stemming = TRUE, removeNumbers = TRUE))
test <- as.matrix(tdm)

# Check accuracy on test.
predict(fit, newdata = test)

