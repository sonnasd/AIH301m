import { slide, text, rows, table, chart, equation, C } from './layout.mjs';
import { policy, fmt } from './data.mjs';

export function member2(p) {
  let s = slide(p, 10, 'MDP của bản triển khai', 60,
    'Sơn nhận phần từ Giang. MDP là Markov Decision Process.\nS: 75 trạng thái cụm và hai terminal. A: 25 tổ hợp liều. T: chuyển trạng thái ước lượng từ train. R: phần thưởng theo chuyển. Gamma bằng 0,99.\nChính sách pi ánh xạ mỗi trạng thái thường sang một hành động khi train.\nChuyển: T được ước lượng như thế nào.',
    'member_2_rl/mdp_builder.py, policy_iteration.py • artifacts/ai_policy.json');
  text(s, 'M = (S, A, T, R, γ)', 64, 157, 1140, 76, 32, C.teal, true);
  table(s, 10, [
    ['Thành phần', 'Ý nghĩa trong dự án'],
    ['S, A', '75 + 2 trạng thái, 25 tổ hợp liều'],
    ['T(s, a, s′)', 'Xác suất đến s′ khi chọn a tại s'],
    ['R(s, a, s′)', 'Phần thưởng trung bình của chuyển trạng thái'],
    ['γ = 0,99', 'Chiết khấu phần thưởng ở các bước sau'],
  ], { y: 260, h: 318, widths: [300, 852], pt: 21 });

  s = slide(p, 11, 'Ước lượng chuyển trạng thái từ số đếm', 65,
    'N(s,a,s′) là số chuyển trong train. N(s,a) là tổng theo trạng thái kế tiếp.\nChỉ chuẩn hóa các cặp hành động cho phép. Các cặp không hợp lệ không dùng trong argmax.\nTensor 77 × 25 × 77 có trục đầu là s, giữa là a, cuối là s′. 77 bằng 75 cộng hai terminal.\nMỗi hàng T(s,a,:) hợp lệ có tổng bằng một. Reward cùng kích thước, lưu trung bình reward của các chuyển cùng bộ ba.\nChuyển: với số đếm nhỏ, action masking xử lý ra sao.',
    'Mô phỏng: member_2_rl/mdp_builder.py • artifacts/mdp_matrices.npz');
  equation(s, 'T(s, a, s′) = N(s, a, s′) / N(s, a)',
    'N(s, a) = Σₛ′ N(s, a, s′)\nN là số lần quan sát trong trajectories train.', 166, 27);
  text(s, '77 × 25 × 77', 64, 437, 680, 90, 40, C.teal, true);
  text(s, 'trạng thái hiện tại × hành động × trạng thái kế tiếp', 64, 544, 1140, 65, 23);

  s = slide(p, 12, 'Action masking và năm trạng thái fallback', 65,
    'Mục đích: tránh tối ưu trên các hành động có quá ít dữ liệu tại trạng thái đó.\nThông thường A_valid(s) chứa các a có N(s,a) ít nhất năm.\nNếu không a nào đạt năm nhưng có dữ liệu, code giữ hành động có số đếm lớn nhất. Năm trạng thái cụ thể: 27,28,52,58,71.\nCó tổng 327 cặp được phép trong 75×25 cặp, tính cả fallback. Ngưỡng năm là quy tắc thực nghiệm, không bảo đảm an toàn.\nNếu trạng thái hoàn toàn chưa thấy, code có self-loop action0. Artifact hiện tại không có trạng thái loại này.\nChuyển: reward và terminal.',
    'Mô phỏng: artifacts/ai_policy.json, mdp_matrices.npz • member_2_rl/mdp_builder.py');
  rows(s, [
    ['Quy tắc chính', 'Cho phép hành động khi N(s, a) ≥ 5.'],
    ['Fallback khi mọi hành động đều dưới ngưỡng', 'Giữ hành động quan sát nhiều nhất ở trạng thái đó.'],
  ], { gap: 125 });
  text(s, policy.below_threshold_fallback_states.join(', '), 64, 437, 1130, 64, 32, C.teal, true);
  text(s, '5 trạng thái fallback trong bản chạy đã lưu', 64, 511, 1130, 48, 22);
  text(s, 'Masking giảm ngoại suy nhưng không chứng minh đủ độ phủ hay an toàn', 64, 580, 1130, 62, 21, C.red);

  s = slide(p, 13, 'Phần thưởng và hai trạng thái hấp thụ', 55,
    'Bước chưa kết thúc nhận reward0. Chuyển cuối sang alive nhận+100, dead nhận−100.\nTrong mã: alive=75, dead=76. Tự lặp terminal có xác suất một và reward0.\nV terminal được giữ bằng0 vì phần thưởng đã nhận khi đi vào terminal. Không cộng lặp +100 ở các bước sau.\nOutcome là nhãn mô phỏng, không phải tử vong90ngày quan sát thực tế.\nChuyển: Bellman kết hợp reward và giá trị tiếp diễn.',
    'member_2_rl/mdp_builder.py, policy_iteration.py • member_1_data/synthetic_data.py');
  table(s, 13, [
    ['Chuyển trạng thái', 'r', 'Giá trị tiếp diễn'],
    ['Chưa kết thúc episode', '0', 'γV(s′)'],
    ['Vào alive (75)', '+100', 'V(alive) = 0'],
    ['Vào dead (76)', '−100', 'V(dead) = 0'],
  ], { y: 176, h: 300, widths: [500, 170, 482], pt: 22 });
  text(s, 'Terminal tự lặp với reward 0', 64, 532, 1130, 70, 26, C.teal, true);

  s = slide(p, 14, 'Bellman policy evaluation', 85,
    'Giữ chính sách pi cố định khi đánh giá. V_pi(s) là kỳ vọng return bắt đầu tại s và theo pi.\ns′ chạy trên tất cả trạng thái kế tiếp. T là xác suất, R là reward của chuyển, gamma=0,99.\nV_new(s) bằng tổng có trọng số của reward cộng giá trị tiếp diễn.\nTrong code, cập nhật từng s ngay trên mảng V (Gauss-Seidel). Công thức trên slide biểu diễn phép gán Bellman, không khẳng định cập nhật đồng bộ toàn bộ vector.\nDelta là thay đổi tuyệt đối lớn nhất trong một lượt. Dừng khi delta<10^-5. V terminal luôn bằng0.\nChuyển: sau khi đánh giá pi, cải thiện lựa chọn a.',
    'member_2_rl/policy_iteration.py: evaluate_policy()');
  text(s, 'V_mới(s) = Σₛ′ T(s, π(s), s′)', 64, 176, 1150, 65, 27, C.teal, true);
  text(s, '                  × [R(s, π(s), s′) + γV(s′)]', 64, 244, 1150, 70, 27, C.teal, true);
  table(s, 14, [
    ['Ký hiệu', 'Ý nghĩa'],
    ['π(s)', 'Hành động hiện tại của chính sách tại s'],
    ['T và R', 'Xác suất chuyển và reward từ train'],
    ['γV(s′)', 'Giá trị tương lai sau chiết khấu'],
  ], { y: 355, h: 220, widths: [250, 902], pt: 21 });
  text(s, 'Lặp đến khi Δ = maxₛ |V_mới(s) − V_cũ(s)| < 10⁻⁵', 64, 600, 1150, 47, 22);

  s = slide(p, 15, 'Policy improvement trong tập được phép', 65,
    'Q_pi(s,a) là giá trị thử hành động a trước, rồi theo pi ở những bước sau.\nDùng V đã hội tụ từ evaluation. Tính Q trên từng hành động hợp lệ. Chọn argmax.\nTrong code, hành động không hợp lệ có Q âm vô cùng, nên không thể thắng argmax.\nNếu nhiều hành động bằng nhau, numpy.argmax chọn chỉ số đầu. Đây là lựa chọn xác định.\nChuyển: hai phép evaluation và improvement tạo vòng Policy Iteration.',
    'member_2_rl/policy_iteration.py: improve_policy(), compute_all_q_values()');
  equation(s, 'Qπ(s, a) = Σₛ′ T(s, a, s′)[R(s, a, s′) + γVπ(s′)]',
    'Qπ(s, a): kỳ vọng return nếu thử a tại s, rồi tiếp tục theo π.\nVπ là giá trị đã hội tụ ở bước đánh giá.', 170, 25);
  text(s, 'π_mới(s) = argmax {Qπ(s, a) : a ∈ A_valid(s)}', 64, 439, 1150, 80, 27, C.teal, true);
  text(s, 'A_valid(s): các hành động còn lại sau masking và fallback', 64, 553, 1140, 68, 22);

  s = slide(p, 16, 'Vòng lặp Policy Iteration và điều kiện dừng', 65,
    'Bắt đầu từ một hành động hợp lệ đầu tiên tại mỗi trạng thái.\nBước1 evaluation: giữ pi, lặp Bellman đến delta<1e-5. Bước2 improvement: giữ V, cập nhật pi bằng Q lớn nhất.\nLặp hai bước nếu còn trạng thái đổi hành động. Khi chính sách ổn định thì dừng và đánh giá lại lần cuối.\nGiới hạn 1.000 lượt evaluation và100 vòng PI. Code báo lỗi nếu chưa hội tụ, không âm thầm xuất kết quả.\nChuyển: một ví dụ số để nhìn rõ phép so sánh Q.',
    'member_2_rl/policy_iteration.py: fit(), evaluate_policy(), improve_policy()');
  table(s, 16, [
    ['Bước', 'Giữ cố định', 'Cập nhật', 'Điều kiện'],
    ['1. Evaluation', 'Chính sách π', 'Giá trị V', 'Δ < 10⁻⁵'],
    ['2. Improvement', 'Giá trị V', 'Chính sách π', 'Chọn Q lớn nhất'],
    ['3. Kiểm tra', 'Chính sách mới', 'Số hành động đổi', 'Bằng 0: dừng'],
  ], { y: 180, h: 300, widths: [270, 260, 270, 352], pt: 21 });
  text(s, 'Giới hạn: 1.000 lượt đánh giá, 100 vòng Policy Iteration', 64, 534, 1150, 60, 23);
  text(s, 'Vượt giới hạn mà chưa hội tụ: báo lỗi', 64, 602, 1150, 42, 21, C.red);

  s = slide(p, 17, 'Ví dụ Bellman với hai hành động', 75,
    'Ghi rõ ví dụ toán học giả định, không phải kết quả điều trị bệnh nhân.\nMột trạng thái s, V cũ(s)=60 và gamma0,99. Terminal có V0.\nA có50% quay lại s với reward0 và50% vào alive với reward100. B có80% vào alive,20% vào dead.\nQ(A)=0,5×0,99×60+0,5×100=79,7. Q(B)=0,8×100−0,2×100=60.\nGreedy improvement chọnA trong lần so sánh này. V=60 chỉ là đầu vào của phép cập nhật, không phải nghiệm cố định của A.\nPhụ lục29: nghiệm của A là50/(1−0,495), xấp xỉ99,01.\nChuyển: so sánh với chính sách hành vi bằng TD.',
    'Ví dụ toán học giả định của nhóm • công thức: member_2_rl/policy_iteration.py');
  text(s, 'Giả định: γ = 0,99 và V cũ(s) = 60', 64, 148, 1140, 50, 23, C.red);
  table(s, 17, [
    ['Hành động', 'Các chuyển có thể xảy ra', 'Q hiện tại'],
    ['A', '0,5 về s và 0,5 sống', '79,7'],
    ['B', '0,8 sống và 0,2 tử vong', '60'],
  ], { y: 224, h: 213, widths: [220, 680, 252], pt: 22 });
  text(s, 'Q(A) = 0,5 × 0,99 × 60 + 0,5 × 100 = 79,7\nQ(B) = 0,8 × 100 − 0,2 × 100 = 60', 64, 480, 1140, 110, 24, C.teal, true);
  text(s, 'Lần cải thiện này chọn A', 64, 610, 1140, 36, 22);

  s = slide(p, 18, 'Batch empirical-policy TD(0)', 80,
    'TD là Temporal Difference. Đây là đánh giá batch chính sách hành vi thực nghiệm, không phải Q-learning.\nTính target từng hàng bằng reward cộng gammaV ở next_state, triệt phần tiếp diễn nếu done.\nLấy trung bình các target cho mỗi cặp(s,a) đã quan sát, rồi cập nhật Q với alpha=1.\nV(s) là trung bình Q có trọng số pi_b thực nghiệm=N(s,a)/N(s). Không dùng trung bình đều25hành động.\nLặp đến max|deltaV|<1e-8, tối đa10.000 lượt. Pi_b thực nghiệm ở đây không phải bản smoothing dùng trong WIS.\nChuyển: kết quả hội tụ đã lưu của Policy Iteration.',
    'member_2_rl/td_evaluator.py: fit() • artifacts/ai_policy.json');
  text(s, 'yᵢ = rᵢ + γ(1 − doneᵢ)V(s′ᵢ)', 64, 163, 1140, 65, 28, C.teal, true);
  text(s, 'Q(s, a) = trung bình yᵢ của các hàng có cùng (s, a)', 64, 263, 1140, 70, 24);
  text(s, 'V(s) = Σₐ π̂ᵦ(a|s)Q(s, a)', 64, 370, 1140, 65, 28, C.teal, true);
  text(s, 'π̂ᵦ(a|s) = N(s, a) / N(s) là tần suất hành động trên train.\nCập nhật batch với α = 1, chỉ trên cặp (s, a) đã quan sát.', 64, 472, 1140, 105, 22);
  text(s, 'doneᵢ = 1 ở bước cuối nên target chỉ còn reward', 64, 602, 1140, 40, 21);

  s = slide(p, 19, 'Chính sách ổn định sau ba vòng', 60,
    'Đọc history đã lưu: số hành động đổi là53,8,0. Vòng3 stable=true, converged=true.\nBiểu đồ là số trạng thái đổi hành động trong mỗi vòng, không phải số bệnh nhân.\nGiá trị AI trung bình trên75trạng thái train là86,28. TD là80,99. Đây là trung bình trạng thái không trọng số, không phải return test hay OPE.\nKhông dùng chênh lệch train để kết luận hiệu quả lâm sàng.\nChuyển: nhìn vào code thật tạo ra các artifacts.',
    'Mô phỏng: artifacts/ai_policy.json, history và value_scope');
  chart(s, 19, policy.history.map(x => `Vòng ${x.iteration}`), [{ name: 'Trạng thái đổi hành động', values: policy.history.map(x => x.actions_changed), fill: C.teal }], { w: 730, h: 390, max: 60, title: 'Số trạng thái đổi hành động' });
  text(s, 'V trung bình train', 850, 193, 360, 48, 24, C.teal, true);
  text(s, `Policy Iteration\n${fmt(policy.mean_v)}\n\nTD hành vi\n${fmt(policy.td_clinician_mean_v)}`, 850, 262, 360, 260, 24);
  text(s, 'Trung bình trạng thái train không phải OPE hoặc return test', 64, 594, 1150, 54, 22, C.red);

  s = slide(p, 20, 'Code và đầu ra đã chuẩn bị sẵn', 45,
    'Trích nguyên các lệnh cốt lõi từ RLTrainer.train: MDPBuilder.build_from_trajectories và PolicyIteration.fit. Dấu ba chấm chỉ lược phần truyền tham số, không trình bày như chương trình hoàn chỉnh.\nĐọc đầu ra từ ai_policy.json: converged=true,75mục policy,history3vòng.\nRLTrainer.export lưu policyJSON và tensorNPZ. Trình chiếu file đã lưu, không cần chạy notebook.\nChuyển thành viên: “Cường sẽ kiểm tra chính sách này trên dữ liệu giữ lại và giải thích vì sao WIS cao vẫn chưa đủ để kết luận.”',
    'member_2_rl/rl_trainer.py: train(), export() • artifacts/ai_policy.json');
  text(s, 'Trích lược RLTrainer.train()', 64, 157, 1130, 43, 22, C.teal, true);
  text(s, 'self.mdp = MDPBuilder(...)\nself.mdp.build_from_trajectories(train_traj_df)\nself.pi = PolicyIteration(mdp=self.mdp, ...)\nself.pi.fit()', 64, 220, 1150, 187, 22);
  table(s, 20, [
    ['Đầu ra đã lưu', 'Giá trị'],
    ['converged', String(policy.converged)],
    ['policy / history', `${policy.policy.length} mục policy / ${policy.history.length} vòng`],
  ], { y: 450, h: 150, widths: [490, 662], pt: 21 });
  text(s, 'Xuất bằng RLTrainer.export(): ai_policy.json và mdp_matrices.npz', 64, 616, 1150, 35, 20);
}
