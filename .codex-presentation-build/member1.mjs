import { slide, text, rows, table, node, connector, C } from './layout.mjs';
import { cohort, splits } from './data.mjs';

export function member1(p) {
  let s = slide(p, 1, 'AI Clinician', 30,
    'Giới thiệu bài tập lớn AIH301m. Mục tiêu: triển khai và phân tích một agent RL cổ điển trên dữ liệu mô phỏng.\nGiang phụ trách dữ liệu, Sơn phụ trách RL, Cường phụ trách đánh giá.\nPhạm vi: minh họa kỹ thuật, chưa hỗ trợ quyết định lâm sàng.\nChuyển: lựa chọn điều trị mà agent đang mô hình hóa.',
    'AIH301m • Tài liệu dự án và cohort_manifest.json');
  text(s, 'Học chính sách dịch truyền và thuốc vận mạch\ncho bài toán sepsis trong ICU', 64, 166, 1120, 100, 30, C.teal, true);
  table(s, 1, [
    ['Thành viên', 'Phần phụ trách', 'Slide'],
    ['Bùi Đức Giang', 'Member 1: dữ liệu', '1–9'],
    ['Nguyễn Huy Sơn', 'Member 2: học tăng cường (RL)', '10–20'],
    ['Cường', 'Member 3: đánh giá', '21–28'],
  ], { y: 322, h: 240, widths: [350, 622, 180], pt: 21 });
  text(s, 'Bản triển khai mô phỏng phục vụ học tập', 64, 596, 1100, 40, 21, C.red);

  s = slide(p, 2, 'Bài toán điều trị sepsis trong ICU', 60,
    'Sepsis: rối loạn chức năng cơ quan đe dọa tính mạng do đáp ứng mất điều hòa với nhiễm trùng, theo phần mở đầu bài báo.\nICU là đơn vị hồi sức tích cực. Dịch truyền và thuốc vận mạch là hai chiều hành động nhóm mô hình hóa.\nLiều phù hợp phụ thuộc trạng thái hiện tại và diễn biến tiếp theo. Agent chỉ xuất bin liều trong mô phỏng.\nChuyển: bài báo gốc và phần nhóm thực sự triển khai.',
    'Komorowski và cs. (2018), tr. 1716–1717, DOI: 10.1038/s41591-018-0213-5');
  rows(s, [
    ['Đầu vào', 'Các chỉ số sinh lý và xét nghiệm mô tả trạng thái bệnh nhân.'],
    ['Quyết định mỗi bốn giờ', 'Chọn mức dịch truyền và mức thuốc vận mạch.'],
    ['Mục tiêu của mô hình', 'Tối đa hóa tổng phần thưởng chiết khấu theo diễn biến episode.'],
  ], { gap: 123 });
  text(s, 'Khuyến nghị mô phỏng không phải đơn thuốc hay chỉ định điều trị', 64, 570, 1140, 65, 22, C.red);

  s = slide(p, 3, 'Nghiên cứu gốc và phạm vi của nhóm', 70,
    'Bài báo Nature Medicine năm 2018 dùng dữ liệu ICU quan sát thực tế.\nMIMIC-III: 17.083 lượt nhập ICU trong tập phát triển. eRI: 79.073 lượt trong tập kiểm tra.\n750 trạng thái và 48 biến trong bài báo so với 75 trạng thái và 30 đặc trưng ở nhóm. Cả hai dùng 25 hành động.\nMIMIC dùng tử vong 90 ngày, eRI dùng tử vong bệnh viện. Nhóm chỉ có outcome mô phỏng, không đồng nhất với hai endpoint đó.\nChuyển: lý do dùng quyết định tuần tự.',
    'Komorowski và cs. (2018), Methods • artifacts/cohort_manifest.json • ai_policy.json');
  table(s, 3, [
    ['Khía cạnh', 'Komorowski (2018)', 'Bản triển khai của nhóm'],
    ['Dữ liệu', 'MIMIC-III và eRI thực tế', '600 bệnh nhân mô phỏng'],
    ['Biểu diễn', '48 biến, 750 trạng thái', '30 đặc trưng, 75 trạng thái'],
    ['Thuật toán', 'MDP, Policy Iteration', 'MDP, Policy Iteration, TD'],
    ['Giá trị bằng chứng', 'Nghiên cứu dữ liệu quan sát', 'Minh họa kỹ thuật RL'],
  ], { y: 158, h: 350, widths: [230, 443, 479], pt: 20 });
  text(s, 'Nhóm chưa tái lập toàn bộ nghiên cứu và chưa có kiểm chứng lâm sàng', 64, 555, 1130, 70, 22, C.red);

  s = slide(p, 4, 'Quyết định tuần tự và kết quả đến muộn', 65,
    'Đọc sơ đồ từ trái sang phải: trạng thái trước điều trị, hành động, trạng thái ở bước sau.\ns_t là trạng thái, a_t là hành động, r_t là phần thưởng cho chuyển trạng thái.\nMột hành động có thể ảnh hưởng khả năng lựa chọn tiếp theo. Outcome chỉ xuất hiện ở cuối episode.\nGiả định Markov: trạng thái nén chứa đủ thông tin cho dự báo bước kế tiếp. K-means không chứng minh giả định này đúng trên lâm sàng.\nChuyển: ba module hiện thực luồng đó.',
    'Komorowski và cs. (2018), mô hình MDP • member_1_data/synthetic_data.py');
  node(s, 'Trạng thái sₜ\ntrước điều trị', 64, 203);
  node(s, 'Hành động aₜ\nhai mức liều', 475, 203);
  node(s, 'Trạng thái sₜ₊₁\nsau tác động', 886, 203, 330);
  connector(s, 394, 253, 475, 253);
  connector(s, 805, 253, 886, 253);
  text(s, 'Phản hồi điều trị đi vào bước tiếp theo', 64, 359, 1140, 50, 24, C.teal, true);
  text(s, 'Các bước giữa có r = 0.\nKết quả kết thúc episode tạo r = +100 hoặc −100.', 64, 439, 1130, 104, 24);
  text(s, 'Biểu diễn Markov là giả định của mô hình', 64, 579, 1140, 40, 21, C.red);

  s = slide(p, 5, 'Luồng xử lý và trách nhiệm ba module', 45,
    'Module dữ liệu tạo trajectories. Module RL chỉ học từ train và lưu policy. Module đánh giá đọc policy và dữ liệu giữ lại.\nĐiểm bàn giao là các CSV và artifacts đã lưu. Không cần chạy notebook khi thuyết trình.\nGiang: member_1_data. Sơn: member_2_rl. Cường: member_3_evaluation.\nChuyển: đặc điểm cụ thể của dữ liệu đang có.',
    'README.md • member_1_data • member_2_rl • member_3_evaluation');
  rows(s, [
    ['01  Bùi Đức Giang: dữ liệu và biểu diễn', 'Mô phỏng, chia tập, xử lý thiếu, chuẩn hóa, mã hóa trạng thái và hành động.'],
    ['02  Nguyễn Huy Sơn: agent RL', 'Tạo MDP, action masking, Policy Iteration và TD. Lưu chính sách.'],
    ['03  Cường: đánh giá và diễn giải', 'Ước lượng hành vi, WIS, bootstrap, ESS và phân tích giới hạn.'],
  ], { gap: 137 });

  s = slide(p, 6, 'Dữ liệu mô phỏng hiện có', 60,
    'Đọc cohort_manifest: synthetic-v2, 600 bệnh nhân, 30 đặc trưng, độ phân giải bốn giờ, cửa sổ tối đa 72 giờ.\nVí dụ đặc trưng: tuổi, MAP (huyết áp động mạch trung bình), lactate, creatinine, urine_output.\nKhông xem đây là cohort Sepsis-3 trích xuất từ hồ sơ thật. Chưa có ETL MIMIC hoàn chỉnh.\nETL là trích xuất, chuyển đổi và nạp dữ liệu.\nChuyển: chia tập và ngăn rò rỉ dữ liệu.',
    'Mô phỏng: artifacts/cohort_manifest.json • member_1_data/synthetic_data.py');
  text(s, String(cohort.counts.total_patients), 64, 170, 300, 90, 58, C.teal, true);
  text(s, 'bệnh nhân mô phỏng', 64, 273, 530, 46, 23);
  text(s, '30 đặc trưng\nBước thời gian: 4 giờ\nCửa sổ tối đa: 72 giờ', 660, 182, 550, 190, 26);
  text(s, 'Đặc trưng gồm chỉ số sinh lý, xét nghiệm\nvà thông tin nền của bệnh nhân', 64, 398, 1140, 100, 23);
  text(s, 'Chưa có ETL MIMIC hoàn chỉnh hoặc cohort Sepsis-3 thực tế', 64, 561, 1140, 72, 22, C.red);

  s = slide(p, 7, 'Chia theo bệnh nhân, fit chỉ trên train', 70,
    'Mỗi bệnh nhân chỉ thuộc một tập, để các lần quan sát không rò giữa train và test.\nImputation bằng median, StandardScaler và K-means chỉ fit trên train. Validation/test chỉ transform bằng đối tượng đã lưu.\nTrain dùng học MDP và chính sách. Validation dùng phân tích độ nhạy. Test dùng báo cáo cuối.\nSố hàng là số chuyển trạng thái, không phải số bệnh nhân.\nChuyển: mã hóa thành trạng thái và hành động rời rạc.',
    'Mô phỏng: data/trajectories_{train,val,test}.csv • member_1_data/preprocessor.py');
  table(s, 7, [
    ['Tập', 'Bệnh nhân', 'Chuyển trạng thái', 'Vai trò'],
    ...splits.map((r, i) => [[ 'Train', 'Validation', 'Test' ][i], String(r.patients), r.rows.toLocaleString('vi-VN'), ['Học biểu diễn và policy', 'Phân tích độ nhạy', 'Báo cáo giữ lại'][i]]),
  ], { y: 160, h: 294, widths: [205, 200, 280, 467], pt: 21 });
  text(s, 'Median, scaler và K-means: fit trên 420 bệnh nhân train', 64, 507, 1140, 60, 24, C.teal, true);
  text(s, 'Validation và test chỉ dùng transform', 64, 582, 1140, 40, 22);

  s = slide(p, 8, '75 trạng thái và 25 hành động', 75,
    'K-means gom vector 30 đặc trưng thành 75 cụm, mã trạng thái từ 0 đến 74. Các cụm không phải chẩn đoán y khoa.\nMỗi chiều điều trị có năm bin: không dùng và bốn mức liều dương. Ngưỡng bin chỉ học từ train.\nMã hành động a = 5 × bin_dịch + bin_vận_mạch. Hai bin chạy từ 0 đến 4.\nVí dụ bin dịch 2 và vận mạch 3 tạo hành động 13. Đây chỉ là ví dụ mã hóa.\nChuyển: cách biến một ICU stay thành episode.',
    'member_1_data/state_action_encoder.py • artifacts/action_bins.json • features_schema.json');
  text(s, 'Trạng thái', 64, 167, 500, 42, 26, C.teal, true);
  text(s, '30 đặc trưng sau xử lý\nK-means với 75 cụm\ns ∈ {0, …, 74}', 64, 237, 500, 180, 24);
  text(s, 'Hành động', 660, 167, 550, 42, 26, C.teal, true);
  text(s, '5 bin dịch × 5 bin vận mạch\n25 tổ hợp liều\na ∈ {0, …, 24}', 660, 237, 550, 180, 24);
  equationLocal(s, 'a = 5 × bin_dịch + bin_vận_mạch', 'Ví dụ mã hóa: (2, 3) tương ứng hành động 13');

  s = slide(p, 9, 'Một episode kết thúc theo từng ICU stay', 65,
    'Mỗi hàng chứa state, action, reward, next_state, done và định danh bệnh nhân/ICU stay.\nstate đo trước điều trị hiện tại. Tác động của a_t đi vào state ở bước kế tiếp, tránh đưa thông tin tương lai vào đầu vào hiện tại.\nKhóa episode gồm subject_id, hadm_id, icustay_id khi có. Mỗi ICU stay có một terminal, không nối qua stay khác.\nHiện dữ liệu test có 90 bệnh nhân và 90 episode, nhưng hợp đồng vẫn hỗ trợ nhiều stay/bệnh nhân.\nChuyển thành viên: Giang kết: “Từ những trajectories này, Sơn sẽ trình bày cách xây MDP và học chính sách.”',
    'member_1_data/trajectory_validation.py • synthetic_data.py • data/trajectories_train.csv');
  table(s, 9, [
    ['Thời điểm', 'Thông tin dùng', 'Sau bước hiện tại'],
    ['t', 'sₜ trước điều trị, chọn aₜ', 'rₜ và sₜ₊₁'],
    ['t + 1', 'sₜ₊₁ trước điều trị mới', 'Tiếp tục episode'],
    ['Bước cuối', 'Outcome mô phỏng của stay', 'done = True, vào terminal'],
  ], { y: 180, h: 300, widths: [210, 460, 482], pt: 21 });
  text(s, 'Không nối chuyển trạng thái giữa hai ICU stay khác nhau', 64, 539, 1140, 74, 24, C.teal, true);
}

function equationLocal(s, formula, caption) {
  text(s, formula, 64, 477, 1130, 65, 26, C.teal, true);
  text(s, caption, 64, 562, 1130, 60, 22);
}
