import { slide, text, table, C } from './layout.mjs';

export function appendix(p) {
  let s = slide(p, 29, 'Phụ lục A: ký hiệu và nghiệm Bellman', 0,
    'Dùng để giải thích lại slide 17 khi hỏi đáp. A tự lặp 50% với reward 0, vào sống 50% với reward 100.\nV_A = 0,5 × 0,99V_A + 50, suy ra 0,505V_A = 50 và V_A ≈ 99,01.\nSlide 17 dùng V cũ = 60 để tính một lần Q(A) = 79,7. Phân biệt một lần cập nhật với nghiệm cố định.\nNghiệm của B là 60 do chuyển thẳng sang terminal. Đây là ví dụ toán học giả định.',
    'Ví dụ toán học giả định • member_2_rl/policy_iteration.py');
  table(s, 29, [
    ['Ký hiệu', 'Diễn giải'],
    ['s, a, s′', 'Trạng thái, hành động, trạng thái kế tiếp'],
    ['T, R, γ', 'Xác suất chuyển, reward, hệ số chiết khấu'],
    ['Vπ(s), Qπ(s,a)', 'Giá trị trạng thái và giá trị thử hành động'],
  ], { y: 162, h: 228, widths: [320, 832], pt: 21 });
  text(s, 'V_A = 0,5 × 0,99V_A + 50\nV_A = 50 / (1 − 0,495) ≈ 99,01', 64, 437, 1150, 112, 28, C.teal, true);
  text(s, 'Q(A) = 79,7 ở slide 17 là một lần cập nhật từ V cũ = 60', 64, 586, 1140, 60, 22);

  s = slide(p, 30, 'Phụ lục B: thuật toán và vị trí trong code', 0,
    'Giang giải thích module dữ liệu, Sơn giải thích MDP / PI / TD, Cường giải thích WIS / bootstrap.\nCác đường dẫn nằm trong repo. Tìm trực tiếp tên hàm khi cần đối chiếu.\nTrích lược ở slide 20 lấy từ RLTrainer.train. RLTrainer.export lưu artifacts đã dùng.',
    'Source code hiện có trong member_1_data, member_2_rl và member_3_evaluation');
  table(s, 30, [
    ['Thuật toán / bước', 'File và hàm chính'],
    ['Mã hóa trạng thái / hành động', 'member_1_data/state_action_encoder.py'],
    ['Ước lượng MDP và masking', 'member_2_rl/mdp_builder.py: build_from_trajectories'],
    ['Bellman / policy improvement', 'member_2_rl/policy_iteration.py: evaluate_policy, improve_policy'],
    ['TD batch theo hành vi', 'member_2_rl/td_evaluator.py: fit'],
    ['WIS và log-weight', 'member_3_evaluation/off_policy_wis.py: evaluate_trajectories'],
    ['Bootstrap theo bệnh nhân', 'member_3_evaluation/bootstrap_evaluator.py'],
  ], { y: 160, h: 448, widths: [390, 762], pt: 20 });

  s = slide(p, 31, 'Phụ lục C: kiểm thử và hợp đồng dữ liệu', 0,
    'Trajectories cần subject_id, time_step, state, action, next_state, reward và done. Các khóa stay dùng thêm nếu có.\nValidator kiểm tra số hữu hạn, mã nguyên đúng miền, bước thời gian không âm và liên tiếp, next_state nối đúng, chỉ một done ở cuối và next_state cuối là terminal.\nLưu ý: validator hiện không bắt buộc bước đầu bằng 0, cũng không đối chiếu reward ±100 với outcome. Không trình bày các điều đó như kiểm tra đã có.\nBộ test_regressions.py có 19 ca unittest, gồm TD/Bellman, WIS log-weight, bootstrap nhiều stay, hợp đồng dữ liệu và luồng tích hợp nhỏ.\nKiểm thử phần mềm không chứng minh hiệu quả lâm sàng.',
    'member_1_data/trajectory_validation.py • test_regressions.py');
  table(s, 31, [
    ['Lớp kiểm tra', 'Ràng buộc cần giữ'],
    ['Định danh', 'Không nối chuyển giữa hai ICU stay'],
    ['Miền giá trị', 's: 0–74, a: 0–24, terminal: 75/76'],
    ['Trình tự', 'time_step liên tiếp, next_state nối đúng'],
    ['Kết thúc', 'Một done cuối episode, next_state cuối là terminal'],
    ['Thuật toán', '19 ca hồi quy trong test_regressions.py'],
  ], { y: 162, h: 372, widths: [310, 842], pt: 21 });
  text(s, 'Kiểm thử phần mềm không thay thế kiểm chứng lâm sàng', 64, 583, 1140, 65, 23, C.red);

  s = slide(p, 32, 'Phụ lục D: câu hỏi phản biện', 0,
    'Phân công trả lời: Giang về dữ liệu và chia tập, Sơn về PI / TD / reward, Cường về ESS / WIS / nhân quả.\nTriển khai bệnh viện: nhóm chưa có bằng chứng cho việc đó.\nWIS cao, ESS thấp: nhấn mạnh trọng số tập trung. Bootstrap không sửa thiếu overlap.\nHCOPE: nhóm chưa chứng minh các giả định hay bảo đảm. Không dùng cận dưới bootstrap làm cam kết an toàn.',
    'Đối chiếu README.md, code và artifacts của nhóm');
  table(s, 32, [
    ['Câu hỏi', 'Ý trả lời ngắn'],
    ['Đây có phải dữ liệu MIMIC?', 'Chưa. Bản hiện tại dùng synthetic-v2.'],
    ['Tại sao terminal có V = 0?', 'Reward ±100 đã nhận khi đi vào terminal.'],
    ['TD có phải Q-learning?', 'Không. TD đánh giá hành vi thực nghiệm.'],
    ['WIS cao nghĩa là AI tốt hơn?', 'ESS 1,90/90 khiến so sánh thiếu tin cậy.'],
    ['Có thể dùng tại bệnh viện?', 'Chưa có kiểm chứng lâm sàng hoặc HCOPE.'],
  ], { y: 164, h: 405, widths: [430, 722], pt: 21 });
  text(s, 'Giang: dữ liệu. Sơn: RL. Cường: đánh giá.', 64, 606, 1140, 42, 22, C.teal, true);
}
