# OCR_MSC-Team

Code này được chạy trên Python 3.9.17 Anaconda

## Install
Để clone và cài đặt các thư viên cần thiết, chạy lệnh sau:

```bash
git clone https://github.com/quangtien19999/OCR_MSC-Team.git
cd OCR_MSC-Team
pip install -r requirements.txt  # install
```

## Run
Download file weight tại [đây](https://drive.google.com/drive/folders/1t9e_Bet6D1CTga_yj7I5WCiiV7w4TOYI?usp=sharing) và copy vào thư mực weights.
Tập ảnh private test copy vào thư mục private_test.
Để chạy, dùng lênh:

```bash
./run_all.sh
```

Sau khi chạy xong, kết quả ảnh được lưu ở folder visualize và thông tin bbounding box được lưu dưới file .csv.

Dữ liệu train custom (row, col, span) có thể tải tại [đây](https://drive.google.com/drive/folders/1bIIyck7Bk6fGDkmy5tNxew9XPPvJw6s3?usp=sharing).
