# Isolated Word Recognition (Nhận dạng từ ngữ đơn lập)

## Tóm tắt bài toán & Thiết lập chung
Dự án nhận dạng giọng nói từ ngữ đơn lập (Isolated Word Recognition) sử dụng học sâu trên tập dữ liệu Google Speech Commands sử dụng PyTorch và Torchaudio.

Dự án so sánh hiệu năng giữa 3 kiến trúc mạng thần kinh nhân tạo chính:
1. **CNN (Convolutional Neural Network)**: Mạng tích chập trích xuất mẫu đặc trưng cục bộ và phổ tần số trên Log-Mel Spectrogram.
2. **Vanilla RNN**: Mạng nơ-ron hồi quy đơn hướng xử lý chuỗi thời gian của các vector Mel bins.
3. **BiRNN (Bidirectional RNN)**: Mạng nơ-ron hồi quy hai hướng thu nhận ngữ cảnh trước và sau trong chuỗi phát âm.

- Tập dữ liệu: Google Speech Commands Dataset v0.02 (phân loại tập 35 từ, ví dụ: yes, no, dog, cat,...)
- Đặc trưng âm thanh: Sampling rate (16kHz), Log-Mel Spectrogram (`n_mels=40`, `n_fft=400`,`hop_length=160`)
- Hyperparameters cố định: Batch size (32), Optimizer (Adam, $lr = 10^{-3}$, $weight_decay=10^{-4}$), Loss (CrossEntropyLoss), Epochs (5)

## Cấu trúc thư mục dự án
```text
├── assets/             # Hình ảnh biểu đồ báo cáo
├── data/               # Tập dữ liệu Google Speech Commands
├── src/                # Mã nguồn xử lý
│   ├── models/         # Kiến trúc CNN, RNN, BiRNN
│   ├── data.py         # Phân chia tập dữ liệu
│   ├── preprocessing.py# Tiền xử lý âm thanh
│   ├── feature.py      # Trích xuất phổ Log-Mel
│   ├── train.py        # Huấn luyện mô hình
│   └── evaluate.py     # Đánh giá mô hình
├── main.py             # Script chạy huấn luyện & benchmark
├── notebook.ipynb      # Sổ tay trực quan hóa & phân tích
└── requirements.txt    # Danh sách thư viện
```

## Chi tiết cấu hình
| Thông số / Thành phần       | 2D CNN                                                                                            | Vanilla RNN                                      | BiRNN (Hai hướng)                                |
| :-------------------------- | :------------------------------------------------------------------------------------------------ | :----------------------------------------------- | :----------------------------------------------- |
| **Đầu vào mạng**            | `(B, 1, 40, 98)`                                                                                  | `(B, 98, 40)`                                    | `(B, 98, 40)`                                    |
| **Cấu trúc cốt lõi**        | 3 khối Conv2D (32 $\rightarrow$ 64 $\rightarrow$ 128)<br>Kernel $3 \times 3$, stride 1, padding 1 | 2 lớp `nn.RNN` (Đơn hướng)<br>`hidden_dim = 168` | 2 lớp `nn.RNN` (Hai hướng)<br>`hidden_dim = 101` |
| **Hàm kích hoạt**           | ReLU                                                                                              | Tanh                                             | Tanh                                             |
| **Chuẩn hóa (Norm)**        | `BatchNorm2d` sau mỗi Conv block                                                                  | Không                                            | Không                                            |
| **Cơ chế Pooling**          | `AdaptiveAvgPool2d((1, 1))`                                                                       | Temporal Mean Pooling                            | Temporal Mean Pooling                            |
| **Kích thước trước Linear** | 128                                                                                               | 168                                              | 202 ($101 \times 2$)                             |
| **Tổng số tham số**         | **97,635**                                                                                        | **97,979**                                       | **97,601**                                       |

## Bảng kết quả thực nghiệm
| Mô hình         | Số tham số | Train Acc (%) | Val Acc (Best) (%) | Val Loss (Best) | **Test Acc (%)** | **Test Loss** | Thời gian TB / Epoch |
| :-------------- | :--------: | :-----------: | :----------------: | :-------------: | :--------------: | :-----------: | :------------------: |
| **CNN**         |   97,635   |  **84.36%**   |     **84.52%**     |   **0.5945**    |    **82.39%**    |  **0.6200**   |        ~50.6s        |
| **Vanilla RNN** |   97,979   |    68.54%     |       71.26%       |     1.1088      |    **69.59%**    |  **1.1723**   |        ~49.9s        |
| **BiRNN**       |   97,601   |    68.69%     |       70.67%       |     1.1352      |    **69.12%**    |  **1.1953**   |        ~51.0s        |

## Phân tích & Nhận định
- **CNN đạt độ chính xác cao hơn Vanilla RNN và BiRNN**: Đang làm
- **BiRNN không có sự cải thiện đáng kể so với Vanilla RNN**: Đang làm