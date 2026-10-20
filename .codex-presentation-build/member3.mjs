import { slide, text, rows, table, equation, C } from './layout.mjs';
import { metrics as m, mortality, sensitivity, fmt } from './data.mjs';

export function member3(p) {
  let s = slide(p, 21, 'Chính sách hành vi và chính sách đích', 65,
    'Cường nhận phần đánh giá. π_b: xác suất hành vi ước lượng từ train, Laplace smoothing α = 0,05 trên 25 hành động.\nπ_e: policy đích mềm. Hành động PI đề xuất có xác suất 0,99, mỗi hành động khác có xác suất 0,01/24.\nPI khi train là xác định. WIS đánh giá bản làm mềm, khác policy xác định.\nSmoothing tránh số 0 do đếm nhưng không tạo bằng chứng cho hành động hiếm.\nChuyển: tỷ số xác suất theo episode trong WIS.',
    'member_3_evaluation/behavior_policy.py, off_policy_wis.py • configs/evaluation.yaml');
  text(s, 'π̂ᵦ(a|s) = [N(s, a) + 0,05] / [N(s) + 25 × 0,05]', 64, 169, 1150, 80, 25, C.teal, true);
  text(s, 'Hành vi mô phỏng, ước lượng từ train với Laplace smoothing', 64, 269, 1140, 65, 22);
  text(s, 'πₑ(a|s) = 0,99 nếu a = πPI(s)\nπₑ(a|s) = 0,01 / 24 với hành động còn lại', 64, 371, 1150, 118, 26, C.teal, true);
  text(s, 'Smoothing không chứng minh dữ liệu có đủ độ phủ', 64, 561, 1140, 68, 23, C.red);

  s = slide(p, 22, 'WIS theo episode và log-weight', 80,
    'G_i: tổng reward chiết khấu của episode i, t bắt đầu từ 0. ρ_i: tích tỷ số xác suất đích / hành vi.\nℓ_i: tổng log xác suất đích trừ log xác suất hành vi. Chặn trên ℓ_i ở log(500) theo cấu hình, không đặt sàn.\nw_i: trọng số chuẩn hóa. Trừ max ℓ trước exp giúp tránh lỗi số học. WIS = tổng w_i × G_i.\nTarget có xác suất 0 cho weight 0. Hành vi có xác suất 0 trên hành động quan sát là lỗi.\nTest không có episode chạm clip. Return không phải tỷ lệ sống.\nChuyển: bất định của ước lượng và bootstrap theo bệnh nhân.',
    'member_3_evaluation/off_policy_wis.py: evaluate_trajectories(), normalize_log_weights()');
  text(s, 'Gᵢ = Σₜ γᵗrᵢₜ       ρᵢ = ∏ₜ πₑ(aᵢₜ|sᵢₜ) / π̂ᵦ(aᵢₜ|sᵢₜ)', 64, 157, 1150, 75, 25, C.teal, true);
  text(s, 'ℓᵢ = Σₜ [log πₑ(aᵢₜ|sᵢₜ) − log π̂ᵦ(aᵢₜ|sᵢₜ)]', 64, 269, 1150, 70, 25);
  text(s, 'wᵢ = exp(ℓᵢ − max ℓ) / Σⱼ exp(ℓⱼ − max ℓ)', 64, 376, 1150, 65, 26, C.teal, true);
  text(s, 'V̂WIS = Σᵢ wᵢGᵢ', 64, 483, 1150, 65, 28, C.teal, true);
  text(s, 'i: episode, t: bước thời gian. Chặn trên ℓ ở log(500) trước chuẩn hóa.', 64, 580, 1150, 65, 21);

  s = slide(p, 23, 'Bootstrap 500 lần theo bệnh nhân', 60,
    'Lấy mẫu subject_id có hoàn lại. Giữ tất cả episode của mỗi bệnh nhân được chọn. Bệnh nhân xuất hiện k lần thì toàn bộ episode của họ có bội số k.\nMỗi lần lấy mẫu: chuẩn hóa lại trọng số WIS và tính lại trung bình return quan sát.\nPhân vị 2,5% và 97,5% tạo khoảng 95%. Phân biệt trung vị bootstrap với point toàn test.\nTest hiện tại: 90 bệnh nhân, 90 episode, 500 lần, seed 42. Không có bootstrap thiếu support.\nBootstrap không sửa thiếu overlap hoặc sai mô hình.\nChuyển: cách đọc bảng kết quả test.',
    'member_3_evaluation/bootstrap_evaluator.py • configs/evaluation.yaml • evaluation_metrics.json');
  rows(s, [
    ['Đơn vị lấy mẫu: bệnh nhân', 'Lấy có hoàn lại subject_id, giữ toàn bộ ICU stay tương ứng.'],
    ['Tính lại trên mỗi mẫu', 'WIS chuẩn hóa lại trọng số. Return quan sát dùng trung bình episode.'],
    ['Khoảng 95% theo percentile', 'Phân vị 2,5% và 97,5% của 500 lần bootstrap, seed = 42.'],
  ], { gap: 122 });
  text(s, 'Bootstrap không bù được thiếu độ phủ giữa hai chính sách', 64, 578, 1140, 66, 22, C.red);

  s = slide(p, 24, 'Kết quả trên 90 episode test mô phỏng', 80,
    `Point toàn test: WIS ${fmt(m.point_wis)}, return hành vi ${fmt(m.point_clinician)}. Trung vị bootstrap có nhãn riêng, dù WIS làm tròn trùng point.\nKhoảng 95% từ 500 patient-cluster bootstrap: WIS ${fmt(m.wis_ci_lower)} đến ${fmt(m.wis_ci_upper)}, hành vi ${fmt(m.clin_ci_lower)} đến ${fmt(m.clin_ci_upper)}.\nESS ${fmt(m.ess)}/90, low_ess = true. Khoảng nhìn hẹp không khắc phục việc vài episode chi phối.\nKTC là khoảng tin cậy. Đơn vị là return chiết khấu, không phải phần trăm sống hoặc số mạng sống cứu được.\nChuyển: ESS và độ nhạy cho biết giới hạn cụ thể.`,
    'Mô phỏng: artifacts/evaluation_metrics.json • 500 patient-cluster bootstrap, percentile 95%');
  table(s, 24, [
    ['Ước lượng return', 'Toàn test', 'Trung vị\nbootstrap', 'KTC 95%\nbootstrap'],
    ['Chính sách đích mềm (WIS)', fmt(m.point_wis), fmt(m.wis_median), `${fmt(m.wis_ci_lower)}–${fmt(m.wis_ci_upper)}`],
    ['Hành vi mô phỏng quan sát', fmt(m.point_clinician), fmt(m.clin_median), `${fmt(m.clin_ci_lower)}–${fmt(m.clin_ci_upper)}`],
  ], { y: 183, h: 252, widths: [435, 180, 215, 322], pt: 20 });
  text(s, `ESS = ${fmt(m.ess)} / ${m.n_episodes}: so sánh thiếu độ tin cậy`, 64, 480, 1140, 60, 27, C.red, true);
  text(s, 'Chỉ minh họa kỹ thuật. Return không phải tỷ lệ sống.', 64, 571, 1140, 65, 23, C.red);

  const ranges = ['alpha', 'weight_clip', 'soft_prob'].map(key => {
    const values = sensitivity.filter(x => x.param_type === key);
    return [key === 'alpha' ? 'Smoothing α: 0,01–0,20' : key === 'weight_clip' ? 'Clip: 100–1.000' : 'Đích mềm: 0,95–0,99',
      `${fmt(Math.min(...values.map(x => +x.wis_value)))}–${fmt(Math.max(...values.map(x => +x.wis_value)))}`,
      `${fmt(Math.min(...values.map(x => +x.ess)))}–${fmt(Math.max(...values.map(x => +x.ess)))}`];
  });
  s = slide(p, 25, 'ESS thấp và độ nhạy trên validation', 85,
    'ESS: effective sample size = 1 / tổng w_i bình phương, với w đã chuẩn hóa. Test có ESS 1,896, tương đương xấp xỉ hai episode đều trọng số theo chẩn đoán này.\nNgưỡng cảnh báo max(10, 0,1 × số episode) chỉ là quy tắc thực nghiệm.\nBảng độ nhạy chỉ dùng validation. WIS quanh 93 không trộn vào WIS test 91,48.\nESS validation đều thấp. Clip 100, 250, 500, 1.000 cho cùng kết quả vì không episode bị clip. Điều đó chưa chứng minh tính vững.\nMô phỏng, thiếu overlap và mô hình hành vi ước lượng khiến nhóm chưa thể kết luận AI vượt bác sĩ.\nChuyển: liên hệ chênh lệch liều và tử vong.',
    'Mô phỏng: artifacts/evaluation_metrics.json • data/sensitivity_analysis.csv (validation)');
  text(s, 'ESS = 1 / Σᵢ wᵢ²', 64, 155, 730, 70, 30, C.teal, true);
  text(s, `${fmt(m.ess)} / 90 trên test`, 64, 238, 720, 55, 26, C.red, true);
  table(s, 25, [['Thay đổi trên validation', 'WIS', 'ESS'], ...ranges], { y: 327, h: 228, widths: [570, 300, 282], pt: 20 });
  text(s, 'ESS thấp ở mọi cấu hình. Chưa thể kết luận AI vượt bác sĩ.', 64, 590, 1140, 62, 22, C.red, true);

  s = slide(p, 26, 'Chênh lệch bin liều và tử vong mô phỏng', 75,
    'Mỗi episode: lấy trung bình (bin quan sát − bin AI). Đây không phải chênh lệch liều theo mL hoặc microgram.\nBiên phân nhóm trong code: −5, −1, −0,2, 0,2, 1, 5. Các khoảng mở trái, đóng phải. Khớp nghĩa là chênh lệch trong (−0,2 đến 0,2].\nDịch khớp: 2/9 tử vong, 22,2%, KTC Wilson 6,3–54,7%. Vận mạch nhóm > AI: 4/44, 9,1%, KTC 3,6–21,2%.\nKhông có tử vong trong mẫu nhỏ không có nghĩa nguy cơ bằng 0. Trạng thái bệnh ảnh hưởng cả liều và outcome, nên không suy ra nhân quả.\nChuyển: kết luận và các bước còn thiếu.',
    'Mô phỏng: figures/mortality_bin_summary.csv • member_3_evaluation/visualizer.py');
  text(s, 'Dịch truyền', 64, 149, 550, 40, 24, C.teal, true);
  text(s, 'Thuốc vận mạch', 668, 149, 550, 40, 24, C.teal, true);
  for (const [i, treatment] of ['Fluid', 'Vasopressor'].entries()) {
    const values = mortality.filter(r => r.treatment === treatment).map((r, j) => [
      ['<< AI', '< AI', 'Khớp', '> AI', '>> AI'][j], `${r.sum}/${r.count}`,
      `${fmt(+r.mortality * 100, 1)} [${fmt(+r.ci_lower * 100, 1)}–${fmt(+r.ci_upper * 100, 1)}]`,
    ]);
    table(s, 26, [['Nhóm', 'd/n', '% [KTC 95%]'], ...values], { x: i ? 668 : 64, y: 210, w: 548, h: 313, widths: [116, 96, 336], pt: 20 });
  }
  text(s, 'd/n: số tử vong / số episode. KTC 95% theo Wilson.', 64, 547, 1140, 40, 21);
  text(s, 'Nhóm dựa trên chênh lệch bin quan sát so với AI. Liên hệ không chứng minh nhân quả.', 64, 601, 1140, 55, 20, C.red);

  s = slide(p, 27, 'Kết luận và giới hạn ứng dụng', 65,
    'ML: biểu diễn K-means và tối ưu policy trong MDP. Thống kê sinh học: bootstrap theo bệnh nhân, ESS, Wilson và giới hạn nhân quả. Lập trình: hợp đồng trajectory, masking, kiểm tra hội tụ và artifacts.\nNhóm có pipeline RL cổ điển minh họa kỹ thuật. Chưa có clinical trial, kiểm chứng ngoài bệnh viện hoặc bảo đảm HCOPE.\nHCOPE: High Confidence Off-Policy Evaluation. Nhóm chưa chứng minh các giả định cho bảo đảm đánh giá này.\nBước tiếp: cohort thật và ETL có thẩm định, đánh giá thiên lệch / overlap, xác nhận ngoài mẫu và quy trình thử nghiệm an toàn với chuyên gia lâm sàng.\nChuyển: tài liệu tham khảo và kết thúc phần 30 phút.',
    'README.md • artifacts/cohort_manifest.json, evaluation_metrics.json • Komorowski và cs. (2018)');
  table(s, 27, [
    ['Lĩnh vực', 'Vai trò trong dự án'],
    ['Machine Learning', 'K-means, MDP và Policy Iteration'],
    ['Thống kê sinh học', 'Bootstrap, ESS, bất định và giới hạn nhân quả'],
    ['Lập trình', 'Hợp đồng dữ liệu, kiểm thử, lưu artifacts'],
  ], { y: 163, h: 269, widths: [360, 792], pt: 21 });
  text(s, 'Cần dữ liệu thật được thẩm định và đánh giá ngoài mẫu\ntrước quy trình kiểm chứng lâm sàng có giám sát', 64, 477, 1140, 105, 24);
  text(s, 'Chưa có clinical trial hoặc bảo đảm HCOPE', 64, 608, 1140, 37, 22, C.red, true);

  s = slide(p, 28, 'Tài liệu tham khảo và nguồn kết quả', 30,
    'Nguồn chính: Komorowski và cộng sự, Nature Medicine 2018. DOI trên slide có thể mở.\nNguồn nhóm: manifest, policy JSON, evaluation metrics, trajectory CSV, sensitivity CSV và mortality summary CSV trong thư mục dự án.\nYêu cầu bài: docs/AI_in_Healthcare_Assignment 2 Brief.docx, môn AIH301m.\nKết: “Nhóm xin kết thúc phần trình bày và sẵn sàng trao đổi về công thức, code xử lý và giới hạn đánh giá.”\nChỉ mở phụ lục khi cần trả lời câu hỏi.',
    'Bài báo gốc trong docs • artifacts và CSV hiện có của nhóm');
  rows(s, [
    ['Nghiên cứu gốc', 'Komorowski và cs. (2018). The Artificial Intelligence Clinician learns\noptimal treatment strategies for sepsis in intensive care.\nNature Medicine 24, 1716–1720.'],
    ['Số liệu và code của nhóm', 'artifacts/ai_policy.json, evaluation_metrics.json, cohort_manifest.json\ndata/trajectories_*.csv, sensitivity_analysis.csv\nfigures/mortality_bin_summary.csv'],
  ], { y: 158, gap: 190, pt: 20 });
  const doi = text(s, 'https://doi.org/10.1038/s41591-018-0213-5', 64, 572, 1150, 50, 21, C.teal);
  doi.text.get('https://doi.org/10.1038/s41591-018-0213-5').link = { uri: 'https://doi.org/10.1038/s41591-018-0213-5', isExternal: true };
}
