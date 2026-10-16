# Isolated Word Recognition (Nhận dạng từ ngữ đơn lập)

## 1. Tóm tắt bài toán & Thiết lập chung
Dự án nhận dạng giọng nói từ ngữ đơn lập (Isolated Word Recognition) sử dụng học sâu trên tập dữ liệu Google Speech Commands v0.02 (35 lớp nhãn âm thanh: *yes, no, up, down, left, right, on, off, stop, go, zero, one,...*) sử dụng PyTorch và Torchaudio.

Toàn bộ **5 kiến trúc mạng nơ-ron** được thiết kế và đối chuẩn tham số công bằng trong dải **~97.2K – 98.1K tham số**:
1. **CNN (Convolutional Neural Network)**: Mạng tích chập 2D trích xuất mẫu cục bộ và phổ tần số trên ma trận phổ Log-Mel.
2. **Vanilla RNN**: Mạng nơ-ron hồi quy đơn hướng xử lý chuỗi thời gian của các vector Mel bins.
3. **BiRNN (Bidirectional RNN)**: Mạng nơ-ron hồi quy hai hướng thu nhận ngữ cảnh hai chiều trong chuỗi phát âm.
4. **LSTM (Long Short-Term Memory)**: Mạng hồi quy cổng với cell state giải quyết triệt để hiện tượng suy giảm gradient trên chuỗi dài.
5. **GRU (Gated Recurrent Unit)**: Biến thể cổng tinh gọn (Reset & Update gates) tối ưu hóa năng lực ghi nhớ và tốc độ tính toán.

- **Đặc trưng âm thanh**: Sampling rate 16 kHz, Log-Mel Spectrogram (`n_mels=40`, `n_fft=400`, `hop_length=160`) $\rightarrow$ Kích thước ma trận đặc trưng: `(40, 98)`.
- **Dữ liệu tiền xử lý (Offline Caching)**: Trích xuất một lần duy nhất ra định dạng tensor nén `float16`, nạp thẳng vào RAM (`TensorDataset`, `pin_memory=True`), **Disk I/O = 0%**.
- **Siêu tham số huấn luyện**: Batch size = 256, Optimizer = Adam ($lr = 2 \times 10^{-3}$, $weight\_decay = 10^{-4}$), Loss = CrossEntropyLoss, Epochs = 5, Automatic Mixed Precision (AMP FP16).

---

## 2. Cấu trúc thư mục dự án
```text
├── assets/                    # Biểu đồ và hình ảnh trực quan hóa
│   ├── audio_features.png     # Waveform & Log-Mel Spectrogram
│   ├── learning_curves.png    # Đường cong huấn luyện (Loss & Accuracy)
│   └── benchmark_comparison.png # Biểu đồ cột đối sánh hiệu năng
├── checkpoints/               # Trọng số tốt nhất (.pt) và cấu hình (.json)
│   ├── best_CNN.pt / .json
│   ├── best_Vanilla_RNN.pt / .json
│   ├── best_BiRNN.pt / .json
│   ├── best_LSTM.pt / .json
│   └── best_GRU.pt / .json
├── data/
│   ├── processed/             # Cache tensor đã tiền xử lý (train.pt, eval.pt, test.pt)
│   └── [35 word folders]/     # File wav gốc của Google Speech Commands
├── scripts/
│   └── preprocess_data.py     # Script trích xuất đặc trưng offline một lần duy nhất
├── src/                       # Mã nguồn mô-đun hóa
│   ├── models/                # Định nghĩa 5 mô hình (CNN, RNN, BiRNN, LSTM, GRU)
│   ├── checkpoint.py          # Lưu và nạp checkpoint tự động kèm cấu hình
│   ├── data.py                # Phân chia train/val/test theo danh sách chuẩn
│   ├── feature.py             # Trích xuất Log-Mel Spectrogram
│   ├── preprocess_pipeline.py # Pipeline tiền xử lý và nạp In-Memory
│   ├── preprocessing.py       # SpeechDataset xử lý audio
│   ├── train.py               # Vòng lặp huấn luyện tích hợp AMP
│   └── evaluate.py            # Đánh giá mô hình tích hợp AMP
├── main.py                    # Script chạy huấn luyện và kiểm thử tự động
├── notebook.ipynb             # Sổ tay trực quan hóa & đối sánh thực nghiệm
└── requirements.txt           # Danh sách thư viện phụ thuộc
```

---

## 3. Chi tiết cấu hình 5 kiến trúc

| Thông số / Thành phần | 2D CNN | Vanilla RNN | BiRNN | LSTM | GRU |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Đầu vào mạng** | `(B, 1, 40, 98)` | `(B, 98, 40)` | `(B, 98, 40)` | `(B, 98, 40)` | `(B, 98, 40)` |
| **Cấu trúc cốt lõi** | 3 khối Conv2D (32 $\rightarrow$ 64 $\rightarrow$ 128)<br>Kernel $3 \times 3$, padding 1 | 2 lớp `nn.RNN`<br>`hidden_dim = 168` | 2 lớp `nn.RNN`<br>`hidden_dim = 101`<br>(bidirectional) | 2 lớp `nn.LSTM`<br>`hidden_dim = 82` | 2 lớp `nn.GRU`<br>`hidden_dim = 95` |
| **Hàm kích hoạt** | ReLU + BatchNorm2d | Tanh | Tanh | Sigmoid/Tanh | Sigmoid/Tanh |
| **Cơ chế Pooling** | `AdaptiveAvgPool2d((1, 1))` | Temporal Mean | Temporal Mean | Temporal Mean | Temporal Mean |
| **Kích thước Linear** | 128 | 168 | 202 ($101 \times 2$) | 82 | 95 |
| **Tổng số tham số** | **97,764** | **98,148** | **97,804** | **98,108** | **97,221** |

---

## 4. Bảng kết quả thực nghiệm

> *Tất cả các mô hình được huấn luyện trên cùng một tập dữ liệu đã tiền xử lý bằng GPU NVIDIA GeForce RTX 4060 Laptop (8GB VRAM) với batch size 256 qua 5 epochs.*

| Hạng | Mô hình | Số tham số | Val Loss (Best) | Val Acc (Best) (%) | Test Loss | **Test Acc (%)** | Thời gian TB / Epoch |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **GRU** | **97,221** | **0.5100** | **86.44%** | **0.5312** | **85.82%** | **~2.8s** |
| 2 | **LSTM** | **98,108** | **0.5552** | **84.76%** | **0.5740** | **83.91%** | **~2.9s** |
| 3 | **CNN** | **97,764** | 1.0062 | 71.18% | 1.0421 | 70.45% | **~2.2s** |
| 4 | **Vanilla RNN** | **98,148** | 1.2337 | 68.43% | 1.2650 | 67.89% | **~2.5s** |
| 5 | **BiRNN** | **97,804** | 1.2985 | 66.26% | 1.3120 | 65.73% | **~2.7s** |

---

## 5. Phân tích
1. **Sự vượt trội của các kiến trúc cổng (GRU & LSTM)**:
   - **GRU (85.82% Test Acc)** và **LSTM (83.91% Test Acc)** bỏ xa các kiến trúc còn lại tới hơn 15%.
   - Chuỗi âm thanh dài 98 khung thời gian (time frames) khiến Vanilla RNN và BiRNN gặp hiện tượng tiêu biến gradient nghiêm trọng (vanishing gradient). Cơ chế cổng (Gates) của LSTM/GRU duy trì thông tin âm học xuyên suốt 1 giây phát âm hiệu quả vượt bậc.
2. **GRU nhỉnh hơn LSTM với ít tham số hơn**:
   - Nhờ loại bỏ cell state độc lập và tích hợp cổng quên/đầu vào thành update gate, GRU hội tụ nhanh hơn, đạt độ chính xác cao hơn (+1.91% so với LSTM) và tốn ít phép tính hơn.
3. **Hiệu năng tốc độ huấn luyện sau tối ưu I/O**:
   - Nhờ cơ chế **Offline Preprocessing** (lưu Log-Mel Spectrogram ra tensor `float16` 800 MB), thời gian huấn luyện mỗi epoch giảm từ **~200 giây xuống chỉ còn ~2.5 – 2.9 giây** (tăng tốc hơn **70 lần**).
   - Tải ổ cứng giảm về **0%**, GPU RTX 4060 hoạt động tối đa công suất.

---

## 6. Hướng phát triển & Mở rộng
   - Xây dựng mô hình lai CRNN (CNN + GRU/LSTM)
   - Mô hình Transformer, ViT