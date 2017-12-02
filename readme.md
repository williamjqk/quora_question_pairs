## code
- quora_q_pairs_v0_1.py: kaggle上的一个kernal的原始代码
- quora_q_pairs_v0_2.py: 使用dot代替concatenate + 多层dense
- quora_q_pairs_v0_3.py: 试试bidirectional lstm + dropout
- quora_q_pairs_v0_4.py: deep bidirectional lstm + no dropout
- quora_q_pairs_v0_5.py: deep bidirectional lstm(shared both lstm and bidirection) + dropout + concatenate
- quora_q_pairs_v0_6.py: Pool 预处理text
- quora_q_pairs_v0_7.py: name_scope + GRU + attention
- quora_q_pairs_v0_8.py: bidirectional GRU + attention + stack

- predict_quora_q_pairs_v0_1.py: 没有model.fit只是预测

- test_build_model_v0_1.py: 用随机数测试模型 - 3层bilstm + concate + 2fc
- test_build_model_v0_2.py: 用Sequential自定义的layer的tensorboard test
- test_build_model_v0_3.py: tensorboard测试, 清晰定义name_scope

- XGB_handcrafted_leaky.py: kaggle上0.158得分的XGB程序

## input
数据

## output
输出
