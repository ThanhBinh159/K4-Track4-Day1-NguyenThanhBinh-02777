# Báo cáo Lab Day 1 — Nguyễn Thanh Bình — 2A202602777

## 1. Thiết lập

- Môi trường: Google Colab, PyTorch 2.11.0+cu130, CUDA trên GPU Tesla T4.
- Dữ liệu: Forest CoverType; 464 809 mẫu train và 116 203 mẫu eval theo metadata cố định. Tách validation phân tầng 20%, seed 42: 371 847 train và 92 962 validation. Chuẩn hóa 10 đặc trưng số chỉ fit trên phần train; 44 cột one-hot giữ nguyên.
- Mô hình: M-base, 54 → 256 → 128 → 7, 47 879 tham số; ReLU, khởi tạo He, Cross-Entropy, SGD momentum 0,9, lr=0,1, batch=512, weight decay=0, 20 epoch, FP32. Baseline cuối dùng seed 1 (base-s1).
- Accuracy mốc luôn đoán lớp đa số trên validation là 0,4876 (lớp 1), từ bước chuẩn bị dữ liệu trong notebook.
- Đã thử: loss, optimizer, batch size, dropout, gradient clipping, mixed precision và khởi tạo.

## 2. Kiểm tra ban đầu và độ nhiễu

| Kiểm tra | Kết quả |
|---|---|
| Số tham số / shape logits | 47 879 / (B, 7) |
| Loss bước 0 của health check (ln 7 = 1,9459) | 2,3776 |
| Overfit 20 mẫu: loss cuối / accuracy | 0,000081 / 1,000 |
| Gradient khác 0 ở mọi tensor tham số | Có |
| Baseline | 3 seed: base-s1, base-s2, base-s3 |
| Validation accuracy, trung bình ± độ lệch chuẩn mẫu | 0,9082 ± 0,0024 |
| Validation macro-F1, trung bình ± độ lệch chuẩn mẫu | 0,8535 ± 0,0126 |

Ngưỡng nhiễu tham khảo là 2σ = 0,0251 macro-F1 validation, tính từ ba baseline seed. Đây là ước lượng thô, không phải kiểm định thống kê; các thí nghiệm khác chủ yếu chỉ chạy một seed. Health check, shape, loss ban đầu và overfit 20 mẫu được lưu trong output notebook và figures/health_overfit_20.png; các run huấn luyện còn lại có exp_id trong workbook và JSON tương ứng.

## 3. Kết quả theo chủ đề

Các dự đoán dưới đây là giả thuyết ghi trong notebook trước khi huấn luyện. Kết quả so sánh dùng validation; eval chỉ được chấm sau khi đã chọn cấu hình.

### 3.1 Hàm mất mát — Cross-Entropy và MSE

Dự đoán: MSE có thể cập nhật kém hữu ích hơn khi phân lớp sai nhưng logits đã tự tin. Với cùng seed 1 và các thiết lập baseline, base-s1 (CE) đạt macro-F1 0,8390 ở epoch tốt nhất 18; loss-mse đạt 0,7277 ở epoch 20, thấp hơn 0,1114, lớn hơn ngưỡng 2σ. Accuracy tương ứng là 0,9054 và 0,8705. Xem figures/compare_loss.png, figures/compare_loss_loss.png, figures/base-s1.png và figures/loss-mse.png.

CE tối ưu trực tiếp log-likelihood của nhãn đúng. Với MSE áp lên xác suất softmax, gradient còn đi qua Jacobian softmax nên có thể nhỏ khi dự đoán đã bão hòa; vì thế sửa lỗi phân lớp tự tin có thể chậm hơn. Giá trị loss CE và MSE có thang đo khác nhau, nên kết luận dựa vào macro-F1/accuracy chứ không so trực tiếp loss. Đây là một lần chạy mỗi cấu hình.

![So sánh Cross-Entropy và MSE trên validation](figures/compare_loss.png)

### 3.2 Bộ tối ưu hóa

Mỗi bộ tối ưu được dò ba learning rate trong năm epoch (các run opt-*); điểm dưới đây là macro-F1 tốt nhất trên validation trong sweep, không phải kết quả huấn luyện đủ 20 epoch.

| Optimizer | exp_id tốt nhất | LR | Val macro-F1 | Epoch tốt nhất |
|---|---|---:|---:|---:|
| SGD | opt-sgd-lr0p1 | 0,1 | 0,6179 | 5 |
| SGD momentum 0,9 | opt-sgd_momentum-lr0p1 | 0,1 | 0,7825 | 5 |
| Adam | opt-adam-lr0p003 | 0,003 | 0,8111 | 5 |
| AdamW | opt-adamw-lr0p003 | 0,003 | 0,8111 | 5 |

Trong sweep ngắn, Adam và AdamW đồng hạng cao nhất; weight decay đặt bằng 0 nên hai thuật toán cho kết quả bằng nhau là phù hợp. Ở ba LR chung của SGD và SGD momentum (0,01/0,03/0,1), momentum cao hơn SGD lần lượt 0,1731/0,1730/0,1646 macro-F1. LR cao nhất được thử là tốt nhất trong mỗi sweep, nhưng chưa chứng minh đó là tối ưu toàn cục. So sánh trực tiếp Adam với SGD ở cùng LR không đầy đủ vì hai nhóm dùng các dải LR khác nhau. Baseline 20 epoch được chọn là SGD momentum, lr=0,1; do đó không thể tuyên bố Adam thắng khi huấn luyện dài hơn nếu chưa chạy đối chứng 20 epoch và nhiều seed. Xem figures/compare_optimizer.png và figures/compare_optimizer_loss.png.

![So sánh optimizer trên validation](figures/compare_optimizer.png)

### 3.3 Hyper-parameter: batch size

Dự đoán: batch lớn giảm nhiễu gradient nhưng cùng 20 epoch sẽ có ít lần cập nhật hơn. Đúng theo hướng đó, batch-2048 đạt macro-F1 0,7970, thấp hơn base-s1 0,0421 (lớn hơn 2σ); accuracy giảm từ 0,9054 xuống 0,8852. Batch 512 cần xấp xỉ 727 update/epoch, còn batch 2048 khoảng 182; tổng update trong 20 epoch vì vậy giảm khoảng bốn lần. Thời gian mỗi epoch cũng giảm từ 1,272 giây (base-s1) xuống 0,349 giây (batch-2048), nhưng đây không phải so sánh với cùng số bước cập nhật. Kết quả có thể do under-training chứ không chỉ do batch lớn tự thân. Xem figures/compare_hparam.png và figures/compare_hparam_loss.png.

![So sánh batch size trên validation](figures/compare_hparam.png)

### 3.4 Dropout

Dự đoán: dropout chỉ có lợi nếu baseline overfit; nếu hai đường loss gần nhau thì dropout có thể làm chậm học. Ở epoch cuối, base-s1 có train/val loss 0,2223/0,2426 (khoảng cách 0,0202); dropout-0p3 có 0,3109/0,3151 (khoảng cách 0,0042). Tuy khoảng cách nhỏ hơn, macro-F1 giảm từ 0,8390 xuống 0,7875 (−0,0515, lớn hơn 2σ). Baseline không cho thấy khoảng cách loss lớn; dropout 0,3 ở đây có vẻ regularize quá mạnh hoặc làm học chậm, chứ không cải thiện tổng quát hóa. Xem figures/compare_dropout.png và figures/compare_dropout_loss.png.

![So sánh dropout trên validation](figures/compare_dropout.png)

### 3.5 Gradient clipping

Ngưỡng clip c=0,7371 lấy từ trung vị các thống kê p95 grad norm theo batch của ba seed baseline. Ở LR baseline 0,1, các p95 theo epoch vượt ngưỡng trong một số epoch đầu; tuy vậy không có đối chứng clip/bỏ clip riêng ở LR bình thường, nên chưa thể kết luận tác động của clipping ở mức này.

Ở LR cao gấp 10 lần (1,0), clip-high-lr-none đạt macro-F1 0,7732 còn clip-high-lr-p95 đạt 0,8202, tăng 0,0470 (lớn hơn 2σ). Cả hai run đều không bị đánh dấu diverged; vì vậy clipping cải thiện kết quả trong phép thử này nhưng không phải cứu một run đã diverge. Run có clipping ghi nhận grad norm cực đại trước clip 4,183 ở epoch đầu, cao hơn ngưỡng 0,7371. Xem figures/compare_clipping.png và figures/compare_clipping_loss.png.

![So sánh gradient clipping trên validation](figures/compare_clipping.png)

### 3.6 Mixed precision

| Cấu hình | exp_id | Val macro-F1 | Giây/epoch | Peak memory (MB) | Diverged |
|---|---|---:|---:|---:|---|
| FP32 | base-s1 | 0,8390 | 1,272 | 160,941 | Không |
| FP16 | amp-fp16 | 0,8237 | 1,582 | 160,940 | Có |
| BF16 | amp-bf16 | 0,8366 | 1,560 | 160,939 | Không |

Trên workload MLP này, hai chế độ mixed precision chậm hơn FP32 khoảng 23–24% mỗi epoch và không giảm bộ nhớ đo được. FP16 bị đánh dấu diverged do loss/gradient không hữu hạn và dừng sớm theo quy tắc của vòng train; BF16 hoàn tất. Chênh lệch macro-F1 so với baseline seed 1 nhỏ hơn ngưỡng 2σ, nhưng kết luận vẫn hạn chế bởi một seed mỗi chế độ. Kích thước mô hình/mini-batch và overhead chuyển kiểu có thể lấn át lợi ích tăng tốc; không nên giả định mixed precision luôn nhanh hơn. Xem figures/compare_amp.png và figures/compare_amp_loss.png.

![So sánh mixed precision trên validation](figures/compare_amp.png)

### 3.7 Khởi tạo tham số

activation_stats ghi độ lệch chuẩn sau từng lớp Linear/ReLU theo thứ tự Linear 54→256, ReLU, Linear 256→128, ReLU, logits 7 (1024 mẫu validation). Các vector quan sát được:

| exp_id | Activation std theo thứ tự trên | Step-0 loss | Val macro-F1 |
|---|---|---:|---:|
| init-zeros | [0, 0, 0, 0, 0] | 1,9459 | 0,0936 |
| init-normal | [0,035640; 0,021794; 0,004242; 0,002509; 0,000375] | 1,9460 | 0,8449 |
| init-xavier | [0,287649; 0,173320; 0,236466; 0,148530; 0,166152] | 2,0222 | 0,8514 |
| He baseline | Không được in trong log activation | 2,2691 (base-s1) | 0,8390 |

Khởi tạo toàn 0 làm các neuron cùng lớp đối xứng và không học được biểu diễn phân biệt; accuracy 0,4876 và macro-F1 0,0936 gần mốc đoán lớp đa số. He dùng phương sai trọng số xấp xỉ 2/fan-in, phù hợp với ReLU; Xavier dùng 2/(fan-in+fan-out), hướng tới cân bằng phương sai qua lớp. Xavier/normal cao hơn He 0,0124/0,0059 macro-F1 trong lần chạy này, đều nhỏ hơn ngưỡng 2σ nên chưa đủ bằng chứng rằng chúng tốt hơn. Activation std của He không được lưu, vì vậy không suy diễn số còn thiếu. Xem figures/compare_init.png và figures/compare_init_loss.png.

![So sánh các cách khởi tạo trên validation](figures/compare_init.png)

## 4. Đánh giá cuối trên tập eval

Sau khi so sánh bằng validation, lựa chọn tốt nhất trong các run hợp lệ 20 epoch là base-s3 (val macro-F1 0,8614). Cấu hình được giữ nguyên là CE + SGD momentum 0,9, lr 0,1, batch 512, He, FP32; run nộp final-s1 được huấn luyện lại với seed 1. Baseline eval cũng dùng base-s1. Không dùng eval để chọn cấu hình.

| Cấu hình | Run/seed chấm | Val macro-F1 | Eval macro-F1 | Eval accuracy |
|---|---|---:|---:|---:|
| Baseline | base-s1 / 1 | 0,8390 | 0,8410 | 0,9033 |
| Cuối cùng | final-s1 / 1; cấu hình được chọn từ base-s3 | 0,8390 | 0,8410 | 0,9033 |

Eval có 116 203 mẫu. Chênh lệch cuối so với baseline là 0,0000 ở cả macro-F1 lẫn accuracy: không có cải thiện vì cấu hình cuối cùng thực chất là baseline được train lại cùng seed 1. Macro-F1 validation của seed 1 là 0,8390, gần eval 0,8410 (chênh khoảng 0,0020). Run base-s3 đạt validation cao hơn nhưng thuộc một seed khác; độ dao động ba seed nhắc rằng không nên diễn giải riêng điểm cao nhất như một nâng cấp cấu hình. Chỉ có một seed được evaluator chấm cho baseline và final, nên chưa có độ lệch chuẩn eval.

### 4.1 Phân tích lỗi theo lớp

| Lớp | Support | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|
| 0 | 42 368 | 0,9016 | 0,9029 | 0,9022 |
| 1 | 56 661 | 0,9093 | 0,9304 | 0,9197 |
| 2 | 7 151 | 0,8972 | 0,8839 | 0,8905 |
| 3 | 549 | 0,7618 | 0,7923 | 0,7768 |
| 4 | 1 899 | 0,8161 | 0,6403 | 0,7176 |
| 5 | 3 473 | 0,8489 | 0,7328 | 0,7866 |
| 6 | 4 102 | 0,9436 | 0,8484 | 0,8935 |

Lớp khó nhất theo F1 là lớp 4 (0,7176): trong 1 899 mẫu thật, 1 216 được phân loại đúng; 600 bị nhầm thành lớp 1 và 64 thành lớp 0. Lớp 3 có F1 0,7768 với 549 mẫu; 92 mẫu lớp 3 bị nhầm thành lớp 2. Tập eval mất cân bằng rõ (lớp 1 có gần 30 lần số mẫu lớp 4), nhưng confusion matrix không đủ để khẳng định đặc trưng của các lớp giống nhau. Thử tiếp theo hợp lý là weighted cross-entropy hoặc weighted sampler, chọn hệ số chỉ bằng validation và theo dõi macro-F1 cùng precision lớp 1 để tránh đổi một lỗi lấy lỗi khác.

Confusion matrix từ eval_result.json, hàng là lớp thật, cột là dự đoán:

| Thật / Dự đoán | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 38255 | 3897 | 0 | 0 | 23 | 5 | 188 |
| 1 | 3557 | 52719 | 84 | 1 | 191 | 89 | 20 |
| 2 | 4 | 359 | 6321 | 88 | 48 | 331 | 0 |
| 3 | 0 | 0 | 92 | 435 | 0 | 22 | 0 |
| 4 | 64 | 600 | 13 | 0 | 1216 | 6 | 0 |
| 5 | 9 | 325 | 535 | 47 | 12 | 2545 | 0 |
| 6 | 543 | 79 | 0 | 0 | 0 | 0 | 3480 |

## 5. Trả lời câu hỏi dẫn dắt

1. **Optimizer:** Trong sweep 5 epoch với LR chọn riêng, Adam và AdamW cùng đạt 0,8111; momentum đạt 0,7825. Tuy nhiên đây chưa phải kết luận cho 20 epoch, vì baseline đa seed là SGD momentum và các dải LR không trùng hoàn toàn. Ở LR chung của SGD và momentum, momentum tốt hơn. Nếu cố định LR mà không dò, kết quả phụ thuộc mạnh vào độ phù hợp của LR với optimizer; thí nghiệm hiện tại chưa so đủ bốn optimizer tại cùng mọi LR.
2. **Dropout:** Với baseline chưa có khoảng cách train–val lớn, dropout 0,3 làm macro-F1 giảm 0,0515 dù khoảng cách loss nhỏ hơn. Nên cân nhắc dropout khi có dấu hiệu overfit ổn định qua nhiều seed; cần dò mức dropout nhẹ hơn trên validation.
3. **Gradient clipping:** Clip giới hạn norm gradient trước bước cập nhật, giúp giảm ảnh hưởng của batch có gradient lớn. Ở LR=1, clip cải thiện macro-F1 0,0470 và run có norm cực đại 4,183 trước clip với ngưỡng 0,7371. Cả hai run không diverged nên bằng chứng ở đây là cải thiện, không phải cứu run đã hỏng.
4. **Mixed precision:** Không nhanh hơn trong phép thử này: FP16/BF16 mất 1,582/1,560 giây mỗi epoch so với FP32 1,272 giây, bộ nhớ đều khoảng 160,94 MB. Quy mô MLP nhỏ khiến overhead chuyển kiểu/scale có thể lấn át; cần đo trên mô hình và batch lớn hơn trước khi kết luận tổng quát.
5. **Khởi tạo:** Toàn 0 làm các neuron đồng nhất nên gradient không phá được đối xứng; mô hình gần như chỉ đoán lớp đa số. He tăng phương sai theo 2/fan-in cho ReLU, còn Xavier cân bằng dựa trên cả fan-in và fan-out. Điểm Xavier cao hơn He trong một seed nhưng chưa vượt ngưỡng nhiễu 2σ; lợi ích khởi tạo cần đánh giá qua nhiều seed.
6. **Loss không giảm sau 2 000 bước:** (a) kiểm tra batch dữ liệu, dtype/shape, nhãn và NaN/chuẩn hóa để loại trừ lỗi input; (b) kiểm tra logits, loss ban đầu, gradient hữu hạn/khác 0 và một bước optimizer thực sự cập nhật tham số; (c) thử overfit một tập nhỏ cố định rồi xem grad norm, learning rate và đường train/validation. Ở lab này health check đã overfit 20 mẫu xuống loss 0,000081, còn khởi tạo zero cho kết quả gần majority baseline; hai phép thử giúp phân biệt lỗi pipeline/cập nhật với vấn đề tổng quát hóa hoặc khởi tạo.

## 6. Hạn chế và điều bất ngờ

Các so sánh ngoài baseline chủ yếu chỉ có một seed; ngưỡng 2σ từ ba baseline seed chỉ là mốc tham khảo. Optimizer probe chỉ chạy năm epoch; batch 2048 có ít update hơn ở cùng số epoch; chưa có so sánh clipping tại LR bình thường; grid LR còn thưa. FP16 bị diverged, và độ lệch chuẩn activation của He không được log. Điểm bất ngờ là cấu hình được chọn từ seed 3 không tạo mức eval cao hơn baseline seed 1 sau khi final được huấn luyện lại cùng cấu hình/seed 1; kết quả trùng nhau đúng với việc cấu hình không thay đổi.

Nếu có thêm thời gian, ưu tiên chạy nhiều seed cho dropout/clipping/khởi tạo, so optimizer ở thời lượng và bước cập nhật tương đương, dò LR dày hơn, rồi thử xử lý mất cân bằng lớp. Mọi lựa chọn tiếp tục dựa trên validation; chỉ chấm eval cho cấu hình cuối.

## 7. Phụ lục

- Hồ sơ gồm REPORT.md, experiments.xlsx, predictions_eval.csv, eval_result.json, figures/, results/ và code/. Có thêm eval_result_baseline.json để đối chiếu baseline.
- results/ có 26 JSON experiment; figures/ có 26 hình theo exp_id, 14 hình so sánh và health_overfit_20.png.
- code/ có notebook đã lưu output và các module data.py, model.py, optimizer.py, train.py, plots.py, results_table.py, lab_workflow.py.
- Tổng thời gian epoch được ghi trong 26 JSON là khoảng 424 giây (7,1 phút); chưa tính khởi động Colab, nạp dữ liệu và các bước ngoài vòng epoch.
